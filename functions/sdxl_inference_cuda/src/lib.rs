use base64::{Engine as _, engine::general_purpose::STANDARD};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::{self, StableDiffusionConfig};
use edgeless_function::*;
use std::sync::Mutex;

struct AIEngine;

// --- 1. WEB COMMUNICATION CONTRACT ---
#[derive(serde::Deserialize)]
pub struct GenerationRequest {
    pub prompt: String,
    pub timestep: u32,
    pub image_base64: String,
}

#[derive(serde::Serialize)]
pub struct GenerationResponse {
    pub result_base64: String,
}

// --- 2. GLOBAL STATE ---
// Store the text encoders so we can process dynamic prompts from the website!
struct ModelState {
    device: Device,
    unet: stable_diffusion::unet_2d::UNet2DConditionModel,
    vae: stable_diffusion::vae::AutoEncoderKL,
    text_encoder_1: stable_diffusion::clip::ClipTextTransformer,
    text_encoder_2: stable_diffusion::clip::ClipTextTransformer,
    tokenizer: tokenizers::Tokenizer,
}

unsafe impl Send for ModelState {}
unsafe impl Sync for ModelState {}

static MODEL: Mutex<Option<ModelState>> = Mutex::new(None);
static FRAME_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

const SD_HEIGHT: usize = 512;
const SD_WIDTH: usize = 512;
const VAE_SCALE: f64 = 0.13025;

impl EdgeFunction for AIEngine {
    fn handle_init(_payload: Option<&[u8]>, _init_metadata: Option<&[u8]>) {
        edgeless_function::init_logger();
        log::info!("AI Engine: Initializing Dynamic Web Pipeline...");

        let device = Device::new_cuda(0).unwrap_or_else(|e| {
            log::warn!("AI Engine: CUDA not available ({e}), falling back to CPU");
            Device::Cpu
        });

        let api = hf_hub::api::sync::Api::new().expect("Failed to create HF API");

        // Download weights
        log::info!("AI Engine: Downloading SDXL Turbo weights...");
        let sdxl_unet_path = api
            .model("stabilityai/sdxl-turbo".into())
            .get("unet/diffusion_pytorch_model.safetensors")
            .unwrap();
        let sdxl_vae_path = api
            .model("madebyollin/sdxl-vae-fp16-fix".into())
            .get("diffusion_pytorch_model.safetensors")
            .unwrap();
        let clip1_path = api.model("stabilityai/sdxl-turbo".into()).get("text_encoder/model.safetensors").unwrap();
        let clip2_path = api
            .model("stabilityai/sdxl-turbo".into())
            .get("text_encoder_2/model.safetensors")
            .unwrap();

        let sd_config = StableDiffusionConfig::sdxl_turbo(None, Some(SD_HEIGHT), Some(SD_WIDTH));

        // Build models
        let vae = sd_config.build_vae(&sdxl_vae_path, &device, DType::F32).unwrap();
        let unet = sd_config.build_unet(&sdxl_unet_path, &device, 4, false, DType::F16).unwrap();

        let vb_clip1 = unsafe { VarBuilder::from_mmaped_safetensors(&[clip1_path], DType::F32, &device).unwrap() };
        let vb_clip2 = unsafe { VarBuilder::from_mmaped_safetensors(&[clip2_path], DType::F32, &device).unwrap() };

        let text_encoder_1 = stable_diffusion::clip::ClipTextTransformer::new(vb_clip1, &sd_config.clip).unwrap();
        let text_encoder_2 = stable_diffusion::clip::ClipTextTransformer::new(vb_clip2, sd_config.clip2.as_ref().unwrap()).unwrap();

        log::info!("AI Engine: Downloading Tokenizer...");
        let tokenizer_path = api.model("openai/clip-vit-large-patch14".into()).get("tokenizer.json").unwrap();
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).unwrap();

        log::info!("AI Engine: All models ready and waiting for web requests!");

        *MODEL.lock().unwrap() = Some(ModelState {
            device,
            unet,
            vae,
            text_encoder_1,
            text_encoder_2,
            tokenizer,
        });
    }

    fn handle_cast(_src: InstanceId, msg: &[u8]) {
        let mut process = || -> Result<(), Box<dyn std::error::Error>> {
            let _frame_id = FRAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let guard = MODEL.lock().unwrap();
            let state = guard.as_ref().ok_or("AI Engine: Model not initialized!")?;

            // 1. Parse JSON directly into our Web Request struct
            let msg_str = core::str::from_utf8(msg)?;
            let request: GenerationRequest = serde_json::from_str(msg_str)?;

            // 2. Decode the incoming Base64 image
            let image_bytes = STANDARD.decode(&request.image_base64)?;
            let img = image::load_from_memory(&image_bytes)?;
            let resized_sd = img.resize_exact(SD_WIDTH as u32, SD_HEIGHT as u32, image::imageops::FilterType::Triangle);

            // Convert to Tensor
            let raw_sd: Vec<f32> = resized_sd.to_rgb8().as_raw().iter().map(|&b| (b as f32 / 127.5) - 1.0).collect();
            let img_tensor = Tensor::from_vec(raw_sd, (SD_HEIGHT, SD_WIDTH, 3), &state.device)?
                .permute((2, 0, 1))?
                .unsqueeze(0)?
                .to_dtype(DType::F32)?;

            // Encode Image to Latents
            let encoded = state.vae.encode(&img_tensor)?;
            let latents = encoded.sample()?.affine(VAE_SCALE, 0.0)?.to_dtype(DType::F16)?;

            // 3. Encode the DYNAMIC Prompt from the User
            let tokens = state.tokenizer.encode(request.prompt, true).map_err(|e| e.to_string())?;
            let mut input_ids = vec![0u32; 77];
            for (i, &id) in tokens.get_ids().iter().take(77).enumerate() {
                input_ids[i] = id;
            }
            let input_ids = Tensor::from_vec(input_ids, (1, 77), &state.device)?.to_dtype(DType::U32)?;
            let emb1 = state.text_encoder_1.forward(&input_ids)?;
            let emb2 = state.text_encoder_2.forward(&input_ids)?;
            let text_embeddings = Tensor::cat(&[&emb1, &emb2], 2)?;

            // 4. Use the DYNAMIC Timestep from the User
            let timestep = request.timestep;

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

            // Mix
            let latents_noisy = latents.affine(alpha_root, 0.0)?.add(&noise.affine(noise_root, 0.0)?)?;

            // UNet
            let context = text_embeddings.to_dtype(DType::F16)?;
            let noise_pred = state.unet.forward(&latents_noisy, timestep as f64, &context)?;

            // Recover Image
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
                .map(|&v| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8)
                .collect();

            // 5. Package as Base64 and Return
            let img_buf = image::RgbImage::from_raw(SD_WIDTH as u32, SD_HEIGHT as u32, rgb_data).ok_or("Failed to create image buffer")?;

            let mut png_bytes: Vec<u8> = Vec::new();
            let encoder = image::codecs::png::PngEncoder::new(&mut png_bytes);
            image::ImageEncoder::write_image(encoder, img_buf.as_raw(), SD_WIDTH as u32, SD_HEIGHT as u32, image::ColorType::Rgb8)?;

            // Encode to Base64
            let result_base64 = STANDARD.encode(&png_bytes);

            // Create our JSON Response
            let response = GenerationResponse { result_base64 };
            let json_response = serde_json::to_string(&response)?;

            // Send back to the front-end!
            cast("transformed_image_channel", json_response.as_bytes());
            Ok(())
        };

        if let Err(e) = process() {
            log::error!("AI Engine Error: {}", e);
        }
    }

    fn handle_call(_src: InstanceId, msg: &[u8]) -> CallRet {
        // Notice we now return a String (our JSON) instead of ()
        let mut process = || -> Result<String, Box<dyn std::error::Error>> {
            let _frame_id = FRAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let guard = MODEL.lock().unwrap();
            let state = guard.as_ref().ok_or("AI Engine: Model not initialized!")?;

            // 1. Parse JSON
            let msg_str = core::str::from_utf8(msg)?;
            let request: GenerationRequest = serde_json::from_str(msg_str)?;

            // 2. Decode Image
            let image_bytes = STANDARD.decode(&request.image_base64)?;
            let img = image::load_from_memory(&image_bytes)?;
            let resized_sd = img.resize_exact(SD_WIDTH as u32, SD_HEIGHT as u32, image::imageops::FilterType::Triangle);

            let raw_sd: Vec<f32> = resized_sd.to_rgb8().as_raw().iter().map(|&b| (b as f32 / 127.5) - 1.0).collect();
            let img_tensor = Tensor::from_vec(raw_sd, (SD_HEIGHT, SD_WIDTH, 3), &state.device)?
                .permute((2, 0, 1))?
                .unsqueeze(0)?
                .to_dtype(DType::F32)?;

            let encoded = state.vae.encode(&img_tensor)?;
            let latents = encoded.sample()?.affine(VAE_SCALE, 0.0)?.to_dtype(DType::F16)?;

            // 3. Encode DYNAMIC Prompt
            let tokens = state.tokenizer.encode(request.prompt, true).map_err(|e| e.to_string())?;
            let mut input_ids = vec![0u32; 77];
            for (i, &id) in tokens.get_ids().iter().take(77).enumerate() {
                input_ids[i] = id;
            }
            let input_ids = Tensor::from_vec(input_ids, (1, 77), &state.device)?.to_dtype(DType::U32)?;
            let emb1 = state.text_encoder_1.forward(&input_ids)?;
            let emb2 = state.text_encoder_2.forward(&input_ids)?;
            let text_embeddings = Tensor::cat(&[&emb1, &emb2], 2)?;

            // 4. DYNAMIC Timestep & Math
            let timestep = request.timestep;
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

            let latents_noisy = latents.affine(alpha_root, 0.0)?.add(&noise.affine(noise_root, 0.0)?)?;
            let context = text_embeddings.to_dtype(DType::F16)?;
            let noise_pred = state.unet.forward(&latents_noisy, timestep as f64, &context)?;

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
                .map(|&v| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8)
                .collect();

            // 5. Package as Base64
            let img_buf = image::RgbImage::from_raw(SD_WIDTH as u32, SD_HEIGHT as u32, rgb_data).ok_or("Failed to create image buffer")?;
            let mut png_bytes: Vec<u8> = Vec::new();
            let encoder = image::codecs::png::PngEncoder::new(&mut png_bytes);
            image::ImageEncoder::write_image(encoder, img_buf.as_raw(), SD_WIDTH as u32, SD_HEIGHT as u32, image::ColorType::Rgb8)?;

            let result_base64 = STANDARD.encode(&png_bytes);
            let response = GenerationResponse { result_base64 };

            // Return the JSON string so `handle_call` can send it back!
            Ok(serde_json::to_string(&response)?)
        };

        // This is where we answer the "Phone call" and send the JSON back to the browser
        match process() {
            Ok(json_response) => CallRet::Reply(json_response.into_bytes()),
            Err(e) => {
                log::error!("AI Engine Error: {}", e);
                // Even if it fails, send a JSON error back so the browser doesn't freeze
                CallRet::Reply(format!("{{\"error\": \"{}\"}}", e).into_bytes())
            }
        }
    }

    fn handle_stop() {}
}

edgeless_function::export!(AIEngine);
