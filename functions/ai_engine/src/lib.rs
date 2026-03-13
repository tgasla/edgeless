use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::depth_anything_v2::{DepthAnythingV2, DepthAnythingV2Config};
use candle_transformers::models::dinov2;
use edgeless_function::*;
use std::sync::{Arc, OnceLock};

struct AIEngine;

/// JSON wrapper for image data — the dataplane treats all payloads as UTF-8 strings.
#[derive(serde::Serialize, serde::Deserialize)]
struct ImagePayload {
    frame_id: u32,
    png_data: Vec<u8>,
}

/// Global model state — loaded once in handle_init, reused across frames.
struct ModelState {
    model: DepthAnythingV2,
    device: Device,
}

static MODEL: OnceLock<ModelState> = OnceLock::new();
static FRAME_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

const DINO_IMG_SIZE: usize = 518;
const MAGIC_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const MAGIC_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Spectral_r-like colormap: maps a [0,1] gray value to RGB.
fn spectral_color(t: f32) -> (u8, u8, u8) {
    // 9-stop gradient from dark blue to dark red
    let stops: [(f32, f32, f32); 9] = [
        (0.369, 0.310, 0.635), // Dark blue
        (0.196, 0.533, 0.741), // Blue
        (0.400, 0.761, 0.647), // Cyan
        (0.671, 0.867, 0.643), // Green
        (0.902, 0.961, 0.596), // Yellow
        (0.996, 0.878, 0.545), // Orange
        (0.992, 0.682, 0.380), // Red-orange
        (0.957, 0.428, 0.263), // Dark red
        (0.835, 0.243, 0.310), // Dark purple
    ];
    let t = t.clamp(0.0, 1.0);
    let idx = (t * 8.0).min(7.999);
    let i = idx as usize;
    let frac = idx - i as f32;
    let (r0, g0, b0) = stops[i];
    let (r1, g1, b1) = stops[(i + 1).min(8)];
    let r = r0 + (r1 - r0) * frac;
    let g = g0 + (g1 - g0) * frac;
    let b = b0 + (b1 - b0) * frac;
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

impl EdgeFunction for AIEngine {
    fn handle_init(_payload: Option<&[u8]>, _init_metadata: Option<&[u8]>) {
        edgeless_function::init_logger();
        log::info!("AI Engine: Initializing Depth Anything V2...");

        let device = Device::new_cuda(0).unwrap_or_else(|e| {
            log::warn!("AI Engine: CUDA not available ({e}), falling back to CPU");
            Device::Cpu
        });
        log::info!("AI Engine: Using device: {:?}", device);

        // Download model weights from HuggingFace (cached after first download)
        let api = hf_hub::api::sync::Api::new().expect("Failed to create HF API");

        log::info!("AI Engine: Downloading DINOv2 ViT-S weights...");
        let dinov2_path = api
            .model("lmz/candle-dino-v2".into())
            .get("dinov2_vits14.safetensors")
            .expect("Failed to download dinov2_vits14.safetensors");
        log::info!("AI Engine: DINOv2 weights at {:?}", dinov2_path);

        log::info!("AI Engine: Downloading Depth Anything V2 weights...");
        let dav2_path = api
            .model("jeroenvlek/depth-anything-v2-safetensors".into())
            .get("depth_anything_v2_vits.safetensors")
            .expect("Failed to download depth_anything_v2_vits.safetensors");
        log::info!("AI Engine: DAv2 weights at {:?}", dav2_path);

        // Build DINOv2 backbone
        let vb_dino = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dinov2_path], DType::F32, &device)
                .expect("Failed to load DINOv2 safetensors")
        };
        let dinov2_model = dinov2::vit_small(vb_dino).expect("Failed to build DINOv2");
        log::info!("AI Engine: DINOv2 model built");

        // Build Depth Anything V2 head
        let vb_dav2 = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dav2_path], DType::F32, &device)
                .expect("Failed to load DAv2 safetensors")
        };
        let config = DepthAnythingV2Config::vit_small();
        let model = DepthAnythingV2::new(Arc::new(dinov2_model), config, vb_dav2)
            .expect("Failed to build Depth Anything V2");
        log::info!("AI Engine: Depth Anything V2 model ready!");

        MODEL.set(ModelState { model, device }).ok();
    }

    fn handle_cast(_src: InstanceId, msg: &[u8]) {
        let frame_id = FRAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        log::info!("AI Engine: Frame {} ({} bytes received)", frame_id, msg.len());

        let state = match MODEL.get() {
            Some(s) => s,
            None => {
                log::error!("AI Engine: Model not initialized!");
                return;
            }
        };

        // 1. Parse the JSON HTTP response to extract the image body
        let msg_str = core::str::from_utf8(msg).unwrap_or("");
        let input_bytes: Vec<u8> = match serde_json::from_str::<serde_json::Value>(msg_str) {
            Ok(http_resp) => {
                if let Some(body) = http_resp.get("body") {
                    serde_json::from_value::<Vec<u8>>(body.clone()).unwrap_or_default()
                } else {
                    log::error!("AI Engine: No 'body' field in HTTP response");
                    return;
                }
            }
            Err(e) => {
                log::error!("AI Engine: Failed to parse JSON: {e}");
                return;
            }
        };
        log::info!("AI Engine: Decoded input image ({} bytes)", input_bytes.len());

        // 2. Decode the image and get original dimensions
        let img = match image::load_from_memory(&input_bytes) {
            Ok(i) => i,
            Err(e) => {
                log::error!("AI Engine: Failed to decode image: {e}");
                return;
            }
        };
        let original_width = img.width() as usize;
        let original_height = img.height() as usize;
        log::info!("AI Engine: Input image {}x{}", original_width, original_height);

        // 3. Resize to DINO_IMG_SIZE and convert to tensor
        let resized = img.resize_exact(
            DINO_IMG_SIZE as u32,
            DINO_IMG_SIZE as u32,
            image::imageops::FilterType::Triangle,
        );
        let rgb = resized.to_rgb8();
        let raw: Vec<f32> = rgb.as_raw().iter().map(|&b| b as f32).collect();

        // Convert HWC to CHW tensor: [1, 3, 518, 518]
        let tensor = match Tensor::from_vec(raw, (DINO_IMG_SIZE, DINO_IMG_SIZE, 3), &state.device) {
            Ok(t) => t,
            Err(e) => {
                log::error!("AI Engine: Failed to create tensor: {e}");
                return;
            }
        };
        let tensor = tensor
            .permute((2, 0, 1)).unwrap()        // HWC -> CHW
            .unsqueeze(0).unwrap()               // CHW -> BCHW
            .to_dtype(DType::F32).unwrap();

        // Normalize: (pixel / 255 - mean) / std
        let max_val = Tensor::try_from(255.0f32).unwrap()
            .to_device(&state.device).unwrap()
            .broadcast_as(tensor.shape()).unwrap();
        let tensor = (tensor / max_val).unwrap();

        let mean = Tensor::from_vec(MAGIC_MEAN.to_vec(), (3, 1, 1), &state.device).unwrap()
            .broadcast_as(tensor.shape()).unwrap();
        let std_t = Tensor::from_vec(MAGIC_STD.to_vec(), (3, 1, 1), &state.device).unwrap()
            .broadcast_as(tensor.shape()).unwrap();
        let tensor = tensor.sub(&mean).unwrap().div(&std_t).unwrap();

        log::info!("AI Engine: Running Depth Anything V2 inference...");

        // 4. Run forward pass
        let depth = match state.model.forward(&tensor) {
            Ok(d) => d,
            Err(e) => {
                log::error!("AI Engine: Inference failed: {e}");
                return;
            }
        };
        log::info!("AI Engine: Depth output shape: {:?}", depth.shape());

        // 5. Post-process: resize back to original dims, scale to [0,1], colorize
        let depth = depth
            .interpolate2d(original_height, original_width).unwrap();

        // Scale to [0, 1]
        let flat: Vec<f32> = depth.flatten_all().unwrap().to_vec1().unwrap();
        let min_val = flat.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_val - min_val).max(1e-6);

        // Build colorized RGB image
        let mut img_buf = image::RgbImage::new(original_width as u32, original_height as u32);
        for (i, pixel) in img_buf.pixels_mut().enumerate() {
            let normalized = (flat[i] - min_val) / range;
            let (r, g, b) = spectral_color(normalized);
            *pixel = image::Rgb([r, g, b]);
        }

        // 6. Encode as PNG
        let mut png_bytes: Vec<u8> = Vec::new();
        {
            let encoder = image::codecs::png::PngEncoder::new(&mut png_bytes);
            image::ImageEncoder::write_image(
                encoder,
                img_buf.as_raw(),
                original_width as u32,
                original_height as u32,
                image::ColorType::Rgb8,
            ).unwrap();
        }

        // 7. Wrap in JSON and send
        let payload = ImagePayload {
            frame_id,
            png_data: png_bytes.clone(),
        };
        let json_payload = serde_json::to_string(&payload).unwrap();

        log::info!("AI Engine: Frame {} complete. Depth map: {} byte PNG -> {} byte JSON",
            frame_id, png_bytes.len(), json_payload.len());
        cast("transformed_image_channel", json_payload.as_bytes());
    }

    fn handle_call(_src: InstanceId, _msg: &[u8]) -> CallRet {
        CallRet::NoReply
    }

    fn handle_stop() {
        log::info!("AI Engine: Shutting down");
    }
}

edgeless_function::export!(AIEngine);
