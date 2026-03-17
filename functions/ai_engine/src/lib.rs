use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::{self, StableDiffusionConfig};
use edgeless_function::*;
use std::sync::Mutex;

struct AIEngine;

/// JSON wrapper for image data
#[derive(serde::Serialize, serde::Deserialize)]
struct ImagePayload {
    frame_id: u32,
    png_data: Vec<u8>,
}

/// Global model state
struct ModelState {
    device: Device,
    unet: stable_diffusion::unet_2d::UNet2DConditionModel,
    vae: stable_diffusion::vae::AutoEncoderKL,
    text_embeddings: Tensor,
}

unsafe impl Send for ModelState {}
unsafe impl Sync for ModelState {}

static MODEL: Mutex<Option<ModelState>> = Mutex::new(None);
static FRAME_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

const SD_HEIGHT: usize = 512;
const SD_WIDTH: usize = 512;
const VAE_SCALE: f64 = 0.13025;
const PROMPT: &str = "a beautiful cinematic fantasy landscape, vibrant colors, epic volumetric lighting, detailed, ultra high quality, 4k";

impl EdgeFunction for AIEngine {
    fn handle_init(_payload: Option<&[u8]>, _init_metadata: Option<&[u8]>) {
        edgeless_function::init_logger();
        log::info!("AI Engine: Initializing SDXL Turbo Image-to-Image...");

        let device = Device::new_cuda(0).unwrap_or_else(|e| {
            log::warn!("AI Engine: CUDA not available ({e}), falling back to CPU");
            Device::Cpu
        });
        log::info!("AI Engine: Using device: {:?}", device);

        let api = hf_hub::api::sync::Api::new().expect("Failed to create HF API");

        // --- SDXL Turbo ---
        log::info!("AI Engine: Downloading SDXL Turbo weights...");

        let sdxl_unet_path = api
            .model("stabilityai/sdxl-turbo".into())
            .get("unet/diffusion_pytorch_model.safetensors")
            .expect("Failed to download SDXL Turbo UNet");

        let sdxl_vae_path = api
            .model("madebyollin/sdxl-vae-fp16-fix".into())
            .get("diffusion_pytorch_model.safetensors")
            .expect("Failed to download SDXL VAE");

        log::info!("AI Engine: Downloading SDXL Text Encoders...");
        let clip1_path = api
            .model("stabilityai/sdxl-turbo".into())
            .get("text_encoder/model.safetensors")
            .expect("Failed to download text_encoder");

        let clip2_path = api
            .model("stabilityai/sdxl-turbo".into())
            .get("text_encoder_2/model.safetensors")
            .expect("Failed to download text_encoder_2");

        let sd_config = StableDiffusionConfig::sdxl_turbo(None, Some(SD_HEIGHT), Some(SD_WIDTH));

        // VAE built in F32, UNet in F16
        let vae = sd_config.build_vae(&sdxl_vae_path, &device, DType::F32).expect("Failed to build VAE");
        let unet = sd_config
            .build_unet(&sdxl_unet_path, &device, 4, false, DType::F16)
            .expect("Failed to build UNet");

        // Setup the 2 Text Encoders in F32
        let vb_clip1 = unsafe { VarBuilder::from_mmaped_safetensors(&[clip1_path], DType::F32, &device).expect("Failed to load CLIP1 safetensors") };
        let vb_clip2 = unsafe { VarBuilder::from_mmaped_safetensors(&[clip2_path], DType::F32, &device).expect("Failed to load CLIP2 safetensors") };

        let text_encoder_1 = stable_diffusion::clip::ClipTextTransformer::new(vb_clip1, &sd_config.clip).expect("Failed to build CLIP1 text encoder");
        let text_encoder_2 = stable_diffusion::clip::ClipTextTransformer::new(vb_clip2, sd_config.clip2.as_ref().unwrap())
            .expect("Failed to build CLIP2 text encoder");

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

        // Extract and concatenate embeddings (becomes [1, 77, 2048])
        let emb1 = text_encoder_1.forward(&input_ids).expect("Failed to encode text 1");
        let emb2 = text_encoder_2.forward(&input_ids).expect("Failed to encode text 2");
        let text_embeddings = Tensor::cat(&[&emb1, &emb2], 2).expect("Failed to concat embeddings");

        log::info!("AI Engine: All models ready!");

        *MODEL.lock().unwrap() = Some(ModelState {
            device,
            unet,
            vae,
            text_embeddings,
        });
    }

    fn handle_cast(_src: InstanceId, msg: &[u8]) {
        // Define an inner function so we can use the '?' operator
        let mut process = || -> Result<(), Box<dyn std::error::Error>> {
            let frame_id = FRAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let guard = MODEL.lock().unwrap();
            let state = guard.as_ref().ok_or("AI Engine: Model not initialized!")?;

            // 1. Parse Input
            let msg_str = core::str::from_utf8(msg)?;
            let http_resp: serde_json::Value = serde_json::from_str(msg_str)?;
            let body_val = http_resp.get("body").ok_or("No body in request")?;
            let input_bytes: Vec<u8> = serde_json::from_value(body_val.clone())?;

            let img = image::load_from_memory(&input_bytes)?;
            let resized_sd = img.resize_exact(SD_WIDTH as u32, SD_HEIGHT as u32, image::imageops::FilterType::Triangle);
            let rgb_sd = resized_sd.to_rgb8();

            // Normalize to [-1.0, 1.0]
            let raw_sd: Vec<f32> = rgb_sd.as_raw().iter().map(|&b| (b as f32 / 127.5) - 1.0).collect();

            let img_tensor = Tensor::from_vec(raw_sd, (SD_HEIGHT, SD_WIDTH, 3), &state.device)?
                .permute((2, 0, 1))?
                .unsqueeze(0)?
                .to_dtype(DType::F32)?;

            // Encode to latents
            let encoded = state.vae.encode(&img_tensor)?;
            let latents = encoded.sample()?.affine(VAE_SCALE, 0.0)?.to_dtype(DType::F16)?;

            // --- 2. DYNAMIC MATH CALCULATION ---
            let timestep = 250u32;

            // Calculate the exact SDXL noise schedule dynamically to bypass the private Scheduler API
            let mut alpha_cumprod = 1.0f64;
            let beta_start = 0.00085f64.sqrt();
            let beta_end = 0.012f64.sqrt();

            for t in 0..timestep {
                let step_f = t as f64 / 999.0;
                let beta = (beta_start + step_f * (beta_end - beta_start)).powi(2);
                alpha_cumprod *= 1.0 - beta;
            }

            let alpha_root = alpha_cumprod.sqrt();
            let noise_root = (1.0 - alpha_cumprod).sqrt();

            let noise = Tensor::randn(0f32, 1.0, latents.shape(), &state.device)?.to_dtype(DType::F16)?;

            // Mix: x_t = (x_0 * alpha_root) + (noise * noise_root)
            let latents_noisy = latents.affine(alpha_root, 0.0)?.add(&noise.affine(noise_root, 0.0)?)?;

            // --- 3. UNET INFERENCE ---
            let context = state.text_embeddings.to_dtype(DType::F16)?;
            let noise_pred = state.unet.forward(&latents_noisy, timestep as f64, &context)?;

            // --- 4. RECOVER IMAGE ---
            // x_0 = (x_t - predicted_noise * noise_root) / alpha_root
            let noise_scaled = noise_pred.affine(noise_root, 0.0)?;
            let denoised = latents_noisy.sub(&noise_scaled)?.affine(1.0 / alpha_root, 0.0)?;

            let scaled_latents = denoised.affine(1.0 / VAE_SCALE, 0.0)?;
            let rgb_output = state.vae.decode(&scaled_latents.to_dtype(DType::F32)?)?;
            let rgb_tensor = rgb_output.squeeze(0)?;

            let rgb_data: Vec<u8> = rgb_tensor
                .permute((1, 2, 0))?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|&v| {
                    // Shift from [-1, 1] to [0, 1] then to [0, 255]
                    let pixel = (v * 0.5 + 0.5).clamp(0.0, 1.0);
                    (pixel * 255.0) as u8
                })
                .collect();

            // 5. Package and Cast
            let img_buf = image::RgbImage::from_raw(SD_WIDTH as u32, SD_HEIGHT as u32, rgb_data).ok_or("Failed to create image buffer")?;

            let mut png_bytes: Vec<u8> = Vec::new();
            let encoder = image::codecs::png::PngEncoder::new(&mut png_bytes);
            image::ImageEncoder::write_image(encoder, img_buf.as_raw(), SD_WIDTH as u32, SD_HEIGHT as u32, image::ColorType::Rgb8)?;

            let payload = ImagePayload {
                frame_id,
                png_data: png_bytes,
            };
            let json_payload = serde_json::to_string(&payload)?;

            cast("transformed_image_channel", json_payload.as_bytes());
            Ok(())
        };

        if let Err(e) = process() {
            log::error!("AI Engine Error: {}", e);
        }
    }

    fn handle_call(_src: InstanceId, _msg: &[u8]) -> CallRet {
        CallRet::NoReply
    }

    fn handle_stop() {}
}

edgeless_function::export!(AIEngine);
