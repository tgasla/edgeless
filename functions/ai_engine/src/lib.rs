use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::depth_anything_v2::{DepthAnythingV2, DepthAnythingV2Config};
use candle_transformers::models::dinov2;
use candle_transformers::models::stable_diffusion;
use edgeless_function::*;
use std::sync::{Arc, Mutex};

struct AIEngine;

/// JSON wrapper for image data — the dataplane treats all payloads as UTF-8 strings.
#[derive(serde::Serialize, serde::Deserialize)]
struct ImagePayload {
    frame_id: u32,
    png_data: Vec<u8>,
}

/// Global model state — loaded once in handle_init, reused across frames.
struct ModelState {
    depth_model: DepthAnythingV2,
    device: Device,
    // SDXL Turbo components
    unet: stable_diffusion::unet_2d::UNet2DConditionModel,
    vae: stable_diffusion::vae::AutoEncoderKL,
    text_embeddings: Tensor,
    sd_config: stable_diffusion::StableDiffusionConfig,
}

// DepthAnythingV2 contains Box<dyn Module> which isn't Sync.
// The edgeless runtime calls handle_cast sequentially, so this is safe.
unsafe impl Send for ModelState {}
unsafe impl Sync for ModelState {}

static MODEL: Mutex<Option<ModelState>> = Mutex::new(None);
static FRAME_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

const DINO_IMG_SIZE: usize = 518;
const MAGIC_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const MAGIC_STD: [f32; 3] = [0.229, 0.224, 0.225];

// SDXL Turbo constants
const SD_HEIGHT: usize = 512;
const SD_WIDTH: usize = 512;
const VAE_SCALE: f64 = 0.13025;
const IMG2IMG_STRENGTH: f64 = 0.5;
const PROMPT: &str = "a beautiful cinematic fantasy landscape, vibrant colors, epic volumetric lighting, detailed, ultra high quality, 4k";

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
        log::info!("AI Engine: Initializing Depth Anything V2 + SDXL Turbo...");

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
        let depth_model = DepthAnythingV2::new(Arc::new(dinov2_model), config, vb_dav2)
            .expect("Failed to build Depth Anything V2");
        log::info!("AI Engine: Depth Anything V2 model ready!");

        // Download SDXL Turbo models
        log::info!("AI Engine: Downloading SDXL Turbo UNet weights...");
        let sdxl_unet_path = api
            .model("stabilityai/sdxl-turbo".into())
            .get("sd_xl_turbo_1.0.safetensors")
            .expect("Failed to download SDXL Turbo UNet");
        log::info!("AI Engine: SDXL UNet weights at {:?}", sdxl_unet_path);

        log::info!("AI Engine: Downloading SDXL VAE weights...");
        let sdxl_vae_path = api
            .model("madebyollin/sdxl-vae-fp16-fix".into())
            .get("sdxl_vae.safetensors")
            .expect("Failed to download SDXL VAE");
        log::info!("AI Engine: SDXL VAE weights at {:?}", sdxl_vae_path);

        log::info!("AI Engine: Downloading CLIP ViT-L/14 weights...");
        let clip_path = api
            .model("openai/clip-vit-large-patch14".into())
            .get("model.safetensors")
            .expect("Failed to download CLIP ViT-L/14");
        log::info!("AI Engine: CLIP weights at {:?}", clip_path);

        // Build SDXL config
        let sd_config = stable_diffusion::StableDiffusionConfig::sdxl();
        log::info!("AI Engine: SDXL config built");

        // Build VAE
        let vb_vae = unsafe {
            VarBuilder::from_mmaped_safetensors(&[sdxl_vae_path], DType::F16, &device)
                .expect("Failed to load VAE safetensors")
        };
        let vae = stable_diffusion::vae::AutoEncoderKL::new(
            sd_config.vae.clone(),
            vb_vae,
            4,
            false,
        ).expect("Failed to build VAE");
        log::info!("AI Engine: VAE ready");

        // Build UNet (Turbo model)
        let vb_unet = unsafe {
            VarBuilder::from_mmaped_safetensors(&[sdxl_unet_path], DType::F16, &device)
                .expect("Failed to load UNet safetensors")
        };
        let unet = stable_diffusion::unet_2d::UNet2DConditionModel::new(
            sd_config.clone(),
            vb_unet,
        ).expect("Failed to build UNet");
        log::info!("AI Engine: UNet ready");

        // Build CLIP text encoder
        let vb_clip = unsafe {
            VarBuilder::from_mmaped_safetensors(&[clip_path], DType::F32, &device)
                .expect("Failed to load CLIP safetensors")
        };
        let text_encoder = stable_diffusion::clip::ClipTextTransformer::new(
            sd_config.clone(),
            vb_clip,
            77,
            false,
        ).expect("Failed to build CLIP text encoder");
        log::info!("AI Engine: CLIP text encoder ready");

        // Tokenize and encode prompt
        log::info!("AI Engine: Encoding prompt: \"{}\"", PROMPT);
        let tokenizer = tokenizers::Tokenizer::from_pretrained("openai/clip-vit-large-patch14", Default::default())
            .expect("Failed to load tokenizer");
        let tokens = tokenizer.encode(PROMPT, true).expect("Failed to encode prompt");
        let mut input_ids = vec![0u32; 77];
        for (i, &id) in tokens.get_ids().iter().take(77).enumerate() {
            input_ids[i] = id;
        }
        let input_ids = Tensor::from_vec(input_ids, (1, 77), &device)
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap();

        // Compute text embeddings
        let text_embeddings = text_encoder.forward(&input_ids).expect("Failed to encode text");
        log::info!("AI Engine: Text embeddings shape: {:?}", text_embeddings.shape());

        log::info!("AI Engine: All models ready!");

        *MODEL.lock().unwrap() = Some(ModelState {
            depth_model,
            device,
            unet,
            vae,
            text_embeddings,
            sd_config,
        });
    }

    fn handle_cast(_src: InstanceId, msg: &[u8]) {
        let frame_id = FRAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        log::info!("AI Engine: Frame {} ({} bytes received)", frame_id, msg.len());

        let guard = MODEL.lock().unwrap();
        let state = match guard.as_ref() {
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
        let depth = match state.depth_model.forward(&tensor) {
            Ok(d) => d,
            Err(e) => {
                log::error!("AI Engine: Depth inference failed: {e}");
                return;
            }
        };
        log::info!("AI Engine: Depth output shape: {:?}", depth.shape());

        // 5. Prepare depth for SDXL img2img: resize to 512x512 and normalize to [-1, 1]
        let depth_512 = depth.interpolate2d(SD_HEIGHT, SD_WIDTH).unwrap();
        let flat: Vec<f32> = depth_512.flatten_all().unwrap().to_vec1().unwrap();
        let min_val = flat.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_val - min_val).max(1e-6);

        // Normalize to [0, 1] then to [-1, 1]
        let normalized: Vec<f32> = flat.iter()
            .map(|&v| ((v - min_val) / range * 2.0 - 1.0) as f64)
            .collect();

        // Convert to tensor [1, 3, 512, 512] - grayscale to RGB by repeating
        let depth_tensor = Tensor::from_vec(
            normalized.iter().flat_map(|&v| vec![v; 3]).collect(),
            (1, 3, SD_HEIGHT, SD_WIDTH),
            &state.device,
        ).unwrap().to_dtype(DType::F32).unwrap();
        log::info!("AI Engine: Depth tensor for SDXL: {:?}", depth_tensor.shape());

        // 6. VAE encode depth to latent
        log::info!("AI Engine: Encoding depth to latent with VAE...");
        let depth_encoded = state.vae.encode(&depth_tensor).expect("Failed to encode with VAE");
        let latents = depth_encoded.scale(VAE_SCALE);
        log::info!("AI Engine: Latent shape: {:?}", latents.shape());

        // 7. Add noise for img2img (timestep based on strength)
        // For SDXL Turbo, we use a single timestep. 50% strength ≈ timestep 500 (out of 1000)
        let timestep = ((1.0 - IMG2IMG_STRENGTH) * 1000.0) as u32;
        log::info!("AI Engine: Adding noise at timestep {}", timestep);

        // Generate random noise
        let noise = Tensor::randn(0f64, 1.0, latents.shape(), &state.device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        // Simple linear interpolation: latents_noisy = latents * (1 - strength) + noise * strength
        let latents_noisy = latents
            .mul(1.0 - IMG2IMG_STRENGTH)
            .unwrap()
            .add(&noise.mul(IMG2IMG_STRENGTH).unwrap())
            .unwrap();
        log::info!("AI Engine: Noised latent shape: {:?}", latents_noisy.shape());

        // 8. Run UNet for 1 step (Turbo mode)
        log::info!("AI Engine: Running SDXL Turbo UNet (1 step)...");
        let time_emb = stable_diffusion::timestep_embedding(
            &[timestep],
            state.sd_config.hidden_size,
            &state.device,
        ).expect("Failed to create timestep embedding");

        // Create context (text embeddings) - expand to batch size
        let context = state.text_embeddings
            .unsqueeze(0)
            .unwrap();

        // Run UNet
        let noise_pred = state.unet
            .forward(&latents_noisy, &time_emb, &context)
            .expect("Failed to run UNet");
        log::info!("AI Engine: UNet output shape: {:?}", noise_pred.shape());

        // 9. Denoise: latents = latents_noisy - noise_pred (simplified for single step)
        let denoised = latents_noisy
            .sub(&noise_pred)
            .unwrap();
        log::info!("AI Engine: Denoised latent shape: {:?}", denoised.shape());

        // 10. VAE decode to RGB
        log::info!("Decoding latent with VAE...");
        let scale = Tensor::try_from(1.0f64 / VAE_SCALE).unwrap()
            .to_device(&state.device)
            .unwrap();
        let scaled_latents = denoised.mul(&scale).unwrap();
        let rgb_output = state.vae.decode(&scaled_latents).expect("Failed to decode with VAE");
        log::info!("AI Engine: VAE output shape: {:?}", rgb_output.shape());

        // 11. Convert tensor to image
        // Output is [1, 3, 512, 512], convert to RGB
        let rgb_tensor = rgb_output.squeeze(0).unwrap();
        let rgb_data: Vec<u8> = rgb_tensor
            .permute((1, 2, 0)).unwrap()
            .flatten_all().unwrap()
            .to_vec1::<f32>().unwrap()
            .iter()
            .map(|&v| ((v.clamp(0.0, 1.0) * 255.0) as u8))
            .collect();

        let img_buf = image::RgbImage::from_raw(
            SD_WIDTH as u32,
            SD_HEIGHT as u32,
            rgb_data,
        ).expect("Failed to create image buffer");

        // 12. Encode as PNG
        let mut png_bytes: Vec<u8> = Vec::new();
        {
            let encoder = image::codecs::png::PngEncoder::new(&mut png_bytes);
            image::ImageEncoder::write_image(
                encoder,
                img_buf.as_raw(),
                SD_WIDTH as u32,
                SD_HEIGHT as u32,
                image::ColorType::Rgb8,
            ).unwrap();
        }

        // 13. Wrap in JSON and send
        let payload = ImagePayload {
            frame_id,
            png_data: png_bytes.clone(),
        };
        let json_payload = serde_json::to_string(&payload).unwrap();

        log::info!("AI Engine: Frame {} complete. Generated scene: {} byte PNG -> {} byte JSON",
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