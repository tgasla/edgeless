use candle_core::{DType, Device, Module, Tensor};
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

impl EdgeFunction for AIEngine {
    fn handle_init(_payload: Option<&[u8]>, _init_metadata: Option<&[u8]>) {
        edgeless_function::init_logger();
        log::info!("AI Engine: Initializing Depth Anything V2 + SDXL Turbo...");

        let device = Device::new_cuda(0).unwrap_or_else(|e| {
            log::warn!("AI Engine: CUDA not available ({e}), falling back to CPU");
            Device::Cpu
        });
        log::info!("AI Engine: Using device: {:?}", device);

        // Download model weights & configs from HuggingFace
        let api = hf_hub::api::sync::Api::new().expect("Failed to create HF API");

        // --- DINOv2 & Depth Anything V2 ---
        log::info!("AI Engine: Downloading DINOv2 ViT-S weights...");
        let dinov2_path = api
            .model("lmz/candle-dino-v2".into())
            .get("dinov2_vits14.safetensors")
            .expect("Failed to download dinov2_vits14.safetensors");

        log::info!("AI Engine: Downloading Depth Anything V2 weights...");
        let dav2_path = api
            .model("jeroenvlek/depth-anything-v2-safetensors".into())
            .get("depth_anything_v2_vits.safetensors")
            .expect("Failed to download depth_anything_v2_vits.safetensors");

        let vb_dino = unsafe { VarBuilder::from_mmaped_safetensors(&[dinov2_path], DType::F32, &device).expect("Failed to load DINOv2 safetensors") };
        let dinov2_model = dinov2::vit_small(vb_dino).expect("Failed to build DINOv2");

        let vb_dav2 = unsafe { VarBuilder::from_mmaped_safetensors(&[dav2_path], DType::F32, &device).expect("Failed to load DAv2 safetensors") };
        let config = DepthAnythingV2Config::vit_small();
        let depth_model = DepthAnythingV2::new(Arc::new(dinov2_model), config, vb_dav2).expect("Failed to build Depth Anything V2");

        // --- SDXL Turbo ---
        log::info!("AI Engine: Downloading SDXL Turbo weights...");

        let sdxl_unet_path = api
            .model("stabilityai/sdxl-turbo".into())
            .get("sd_xl_turbo_1.0.safetensors")
            .expect("Failed to download SDXL Turbo UNet");

        let sdxl_vae_path = api
            .model("madebyollin/sdxl-vae-fp16-fix".into())
            .get("sdxl_vae.safetensors")
            .expect("Failed to download SDXL VAE");

        // CLIP Weights
        let clip_path = api
            .model("openai/clip-vit-large-patch14".into())
            .get("model.safetensors")
            .expect("Failed to download CLIP ViT-L/14");

        // Build overall SD config first — this contains the UNet and VAE configs!
        let sd_config = stable_diffusion::StableDiffusionConfig::sdxl(None, Some(SD_HEIGHT), Some(SD_WIDTH));

        // Build VAE
        let vb_vae = unsafe { VarBuilder::from_mmaped_safetensors(&[sdxl_vae_path], DType::F16, &device).expect("Failed to load VAE safetensors") };
        let vae = stable_diffusion::vae::AutoEncoderKL::new(
            vb_vae,
            3,
            3,
            sd_config.autoencoder_config().clone(), // <-- Using the getter method
        )
        .expect("Failed to build VAE");

        // Build UNet
        let vb_unet =
            unsafe { VarBuilder::from_mmaped_safetensors(&[sdxl_unet_path], DType::F16, &device).expect("Failed to load UNet safetensors") };
        let unet = stable_diffusion::unet_2d::UNet2DConditionModel::new(
            vb_unet,
            4,
            4,
            false,
            sd_config.unet_config().clone(), // <-- Using the getter method
        )
        .expect("Failed to build UNet");

        // Build CLIP
        let vb_clip = unsafe { VarBuilder::from_mmaped_safetensors(&[clip_path], DType::F32, &device).expect("Failed to load CLIP safetensors") };
        let text_encoder = stable_diffusion::clip::ClipTextTransformer::new(vb_clip, &sd_config.clip).expect("Failed to build CLIP text encoder");

        // Load Tokenizer
        log::info!("AI Engine: Downloading Tokenizer...");
        let tokenizer_path = api
            .model("openai/clip-vit-large-patch14".into())
            .get("tokenizer.json")
            .expect("Failed to download tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");

        log::info!("AI Engine: Encoding prompt: \"{}\"", PROMPT);
        let tokens = tokenizer.encode(PROMPT, true).expect("Failed to encode prompt");
        let mut input_ids = vec![0u32; 77];
        for (i, &id) in tokens.get_ids().iter().take(77).enumerate() {
            input_ids[i] = id;
        }
        let input_ids = Tensor::from_vec(input_ids, (1, 77), &device).unwrap().to_dtype(DType::U32).unwrap();

        let text_embeddings = text_encoder.forward(&input_ids).expect("Failed to encode text");
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
        let guard = MODEL.lock().unwrap();
        let state = match guard.as_ref() {
            Some(s) => s,
            None => {
                log::error!("AI Engine: Model not initialized!");
                return;
            }
        };

        // Parse JSON & Decode Image
        let msg_str = core::str::from_utf8(msg).unwrap_or("");
        let input_bytes: Vec<u8> = match serde_json::from_str::<serde_json::Value>(msg_str) {
            Ok(http_resp) => {
                if let Some(body) = http_resp.get("body") {
                    serde_json::from_value::<Vec<u8>>(body.clone()).unwrap_or_default()
                } else {
                    return;
                }
            }
            Err(_) => return,
        };

        let img = image::load_from_memory(&input_bytes).expect("Failed to decode image");

        // Depth Anything Processing
        let resized = img.resize_exact(DINO_IMG_SIZE as u32, DINO_IMG_SIZE as u32, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();
        let raw: Vec<f32> = rgb.as_raw().iter().map(|&b| b as f32).collect();

        let tensor = Tensor::from_vec(raw, (DINO_IMG_SIZE, DINO_IMG_SIZE, 3), &state.device)
            .unwrap()
            .permute((2, 0, 1))
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        let max_val = Tensor::try_from(255.0f32)
            .unwrap()
            .to_device(&state.device)
            .unwrap()
            .broadcast_as(tensor.shape())
            .unwrap();
        let tensor = (tensor / max_val).unwrap();

        let mean = Tensor::from_vec(MAGIC_MEAN.to_vec(), (3, 1, 1), &state.device)
            .unwrap()
            .broadcast_as(tensor.shape())
            .unwrap();
        let std_t = Tensor::from_vec(MAGIC_STD.to_vec(), (3, 1, 1), &state.device)
            .unwrap()
            .broadcast_as(tensor.shape())
            .unwrap();
        let tensor = tensor.sub(&mean).unwrap().div(&std_t).unwrap();

        let depth = state.depth_model.forward(&tensor).expect("Depth inference failed");

        // Convert depth to [-1, 1] RGB for SD img2img
        let depth_512 = depth.interpolate2d(SD_HEIGHT, SD_WIDTH).unwrap();
        let flat: Vec<f32> = depth_512.flatten_all().unwrap().to_vec1().unwrap();
        let min_val = flat.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_val - min_val).max(1e-6);

        let normalized: Vec<f32> = flat.iter().map(|&v| ((v - min_val) / range * 2.0 - 1.0) as f32).collect();

        let depth_tensor = Tensor::from_vec(
            normalized.iter().flat_map(|&v| vec![v; 3]).collect(),
            (1, 3, SD_HEIGHT, SD_WIDTH),
            &state.device,
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

        // VAE Encode & Sample Latents
        let depth_encoded = state.vae.encode(&depth_tensor).expect("Failed to encode with VAE");
        let latents = depth_encoded.sample().unwrap().affine(VAE_SCALE, 0.0).expect("Failed to scale");

        // Add Noise
        let timestep = ((1.0 - IMG2IMG_STRENGTH) * 1000.0) as u32;
        let noise = Tensor::randn(0f32, 1.0, latents.shape(), &state.device).unwrap();

        let latents_noisy = latents
            .affine(1.0 - IMG2IMG_STRENGTH, 0.0)
            .unwrap()
            .add(&noise.affine(IMG2IMG_STRENGTH, 0.0).unwrap())
            .unwrap();

        // Run UNet (Passing f64 directly for timestep)
        let context = state.text_embeddings.unsqueeze(0).unwrap();
        let noise_pred = state.unet.forward(&latents_noisy, timestep as f64, &context).expect("Failed to run UNet");

        // Denoise
        let denoised = latents_noisy.sub(&noise_pred).unwrap();

        // VAE Decode to RGB
        let scaled_latents = denoised.affine(1.0 / VAE_SCALE, 0.0).unwrap();
        let rgb_output = state.vae.decode(&scaled_latents).expect("Failed to decode with VAE");

        // Convert Tensor -> PNG Bytes
        let rgb_tensor = rgb_output.squeeze(0).unwrap();
        let rgb_data: Vec<u8> = rgb_tensor
            .permute((1, 2, 0))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();

        let img_buf = image::RgbImage::from_raw(SD_WIDTH as u32, SD_HEIGHT as u32, rgb_data).expect("Failed to create image buffer");

        let mut png_bytes: Vec<u8> = Vec::new();
        {
            let encoder = image::codecs::png::PngEncoder::new(&mut png_bytes);
            image::ImageEncoder::write_image(encoder, img_buf.as_raw(), SD_WIDTH as u32, SD_HEIGHT as u32, image::ColorType::Rgb8).unwrap();
        }

        // Send via Edgeless Output Channel
        let payload = ImagePayload {
            frame_id,
            png_data: png_bytes.clone(),
        };
        let json_payload = serde_json::to_string(&payload).unwrap();

        cast("transformed_image_channel", json_payload.as_bytes());
    }

    fn handle_call(_src: InstanceId, _msg: &[u8]) -> CallRet {
        CallRet::NoReply
    }

    fn handle_stop() {}
}

edgeless_function::export!(AIEngine);
