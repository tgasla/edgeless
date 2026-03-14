use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::depth_anything_v2::{DepthAnythingV2, DepthAnythingV2Config};
use candle_transformers::models::dinov2;
use candle_transformers::models::stable_diffusion;
use edgeless_function::*;
use std::sync::{Arc, Mutex};

struct AIEngine;

#[derive(serde::Serialize, serde::Deserialize)]
struct ImagePayload {
    frame_id: u32,
    png_data: Vec<u8>,
}

/// All loaded models — initialized once in handle_init, reused across frames.
struct ModelState {
    // Stage 1: Depth estimation
    depth_model: DepthAnythingV2,
    // Stage 2: SDXL Turbo scene generation
    text_embeddings: Tensor,
    unet: stable_diffusion::unet_2d::UNet2DConditionModel,
    vae: stable_diffusion::vae::AutoEncoderKL,
    sd_config: stable_diffusion::StableDiffusionConfig,
    device: Device,
}

// DepthAnythingV2 contains Box<dyn Module> which is not Sync.
// The edgeless runtime calls handle_cast sequentially, so this is safe.
unsafe impl Send for ModelState {}
unsafe impl Sync for ModelState {}

static MODEL: Mutex<Option<ModelState>> = Mutex::new(None);
static FRAME_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

// Depth Anything V2 constants
const DINO_IMG_SIZE: usize = 518;
const MAGIC_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const MAGIC_STD: [f32; 3] = [0.229, 0.224, 0.225];

// SDXL Turbo constants
const SD_HEIGHT: usize = 512;
const SD_WIDTH: usize = 512;
const VAE_SCALE: f64 = 0.13025;
const IMG2IMG_STRENGTH: f64 = 0.5;

const PROMPT: &str = "a beautiful cinematic fantasy landscape, vibrant colors, epic volumetric lighting, detailed, ultra high quality, 4k";

/// Spectral colormap (kept for debug/visualization)
fn spectral_color(t: f32) -> (u8, u8, u8) {
    let stops: [(f32, f32, f32); 9] = [
        (0.369, 0.310, 0.635), (0.196, 0.533, 0.741),
        (0.400, 0.761, 0.647), (0.671, 0.867, 0.643),
        (0.902, 0.961, 0.596), (0.996, 0.878, 0.545),
        (0.992, 0.682, 0.380), (0.957, 0.428, 0.263),
        (0.835, 0.243, 0.310),
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

/// Tokenize + pad a prompt for CLIP
fn tokenize_prompt(tokenizer: &tokenizers::Tokenizer, prompt: &str, max_len: usize, pad_id: u32) -> Vec<u32> {
    let mut tokens = tokenizer.encode(prompt, true).expect("tokenize").get_ids().to_vec();
    tokens.truncate(max_len);
    while tokens.len() < max_len { tokens.push(pad_id); }
    tokens
}

/// Build text embeddings for SDXL Turbo (dual CLIP, no classifier-free guidance)
fn build_text_embeddings(
    prompt: &str,
    sd_config: &stable_diffusion::StableDiffusionConfig,
    device: &Device,
    dtype: DType,
) -> Tensor {
    let api = hf_hub::api::sync::Api::new().expect("HF API");
    let turbo_repo = "stabilityai/sdxl-turbo";
    let eot_token = "<|endoftext|>";

    // ---- CLIP encoder 1 ----
    log::info!("AI Engine: Loading CLIP tokenizer 1...");
    let tok1_path = api.model("openai/clip-vit-large-patch14".into()).get("tokenizer.json").expect("tok1");
    let tokenizer1 = tokenizers::Tokenizer::from_file(tok1_path).expect("parse tok1");
    let pad_id1 = match &sd_config.clip.pad_with {
        Some(p) => *tokenizer1.get_vocab(true).get(p.as_str()).unwrap(),
        None => *tokenizer1.get_vocab(true).get(eot_token).unwrap(),
    };
    let tokens1 = tokenize_prompt(&tokenizer1, prompt, sd_config.clip.max_position_embeddings, pad_id1);
    let tokens1 = Tensor::new(tokens1.as_slice(), device).unwrap().unsqueeze(0).unwrap();

    log::info!("AI Engine: Building CLIP transformer 1...");
    let clip1_path = api.model(turbo_repo.into()).get("text_encoder/model.safetensors").expect("clip1");
    let text_model1 = stable_diffusion::build_clip_transformer(&sd_config.clip, clip1_path, device, DType::F32).unwrap();
    let emb1 = text_model1.forward(&tokens1).unwrap();
    log::info!("AI Engine: CLIP 1 embeddings: {:?}", emb1.shape());

    // ---- CLIP encoder 2 ----
    log::info!("AI Engine: Loading CLIP tokenizer 2...");
    let tok2_path = api.model("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k".into()).get("tokenizer.json").expect("tok2");
    let tokenizer2 = tokenizers::Tokenizer::from_file(tok2_path).expect("parse tok2");
    let clip2_config = sd_config.clip2.as_ref().expect("SDXL needs clip2");
    let pad_id2 = match &clip2_config.pad_with {
        Some(p) => *tokenizer2.get_vocab(true).get(p.as_str()).unwrap(),
        None => *tokenizer2.get_vocab(true).get(eot_token).unwrap(),
    };
    let tokens2 = tokenize_prompt(&tokenizer2, prompt, clip2_config.max_position_embeddings, pad_id2);
    let tokens2 = Tensor::new(tokens2.as_slice(), device).unwrap().unsqueeze(0).unwrap();

    log::info!("AI Engine: Building CLIP transformer 2...");
    let clip2_path = api.model(turbo_repo.into()).get("text_encoder_2/model.safetensors").expect("clip2");
    let text_model2 = stable_diffusion::build_clip_transformer(clip2_config, clip2_path, device, DType::F32).unwrap();
    let emb2 = text_model2.forward(&tokens2).unwrap();
    log::info!("AI Engine: CLIP 2 embeddings: {:?}", emb2.shape());

    // Concatenate along last dim: [1, 77, 768] + [1, 77, 1280] -> [1, 77, 2048]
    let text_embeddings = Tensor::cat(&[emb1, emb2], D::Minus1).unwrap();
    text_embeddings.to_dtype(dtype).unwrap()
}

impl EdgeFunction for AIEngine {
    fn handle_init(_payload: Option<&[u8]>, _init_metadata: Option<&[u8]>) {
        edgeless_function::init_logger();
        log::info!("AI Engine: Initializing (Depth Anything V2 + SDXL Turbo)...");

        let device = Device::new_cuda(0).unwrap_or_else(|e| {
            log::warn!("AI Engine: CUDA unavailable ({e}), falling back to CPU");
            Device::Cpu
        });
        log::info!("AI Engine: Using device: {:?}", device);

        let api = hf_hub::api::sync::Api::new().expect("HF API");

        // ============================================================
        // Stage 1: Depth Anything V2
        // ============================================================
        log::info!("AI Engine: Downloading DINOv2 ViT-S weights...");
        let dinov2_path = api.model("lmz/candle-dino-v2".into())
            .get("dinov2_vits14.safetensors").expect("dinov2 weights");
        log::info!("AI Engine: DINOv2 weights at {:?}", dinov2_path);

        log::info!("AI Engine: Downloading Depth Anything V2 weights...");
        let dav2_path = api.model("jeroenvlek/depth-anything-v2-safetensors".into())
            .get("depth_anything_v2_vits.safetensors").expect("dav2 weights");
        log::info!("AI Engine: DAv2 weights at {:?}", dav2_path);

        let vb_dino = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dinov2_path], DType::F32, &device).expect("load dinov2")
        };
        let dinov2_model = dinov2::vit_small(vb_dino).expect("build dinov2");
        log::info!("AI Engine: DINOv2 model built");

        let vb_dav2 = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dav2_path], DType::F32, &device).expect("load dav2")
        };
        let config = DepthAnythingV2Config::vit_small();
        let depth_model = DepthAnythingV2::new(Arc::new(dinov2_model), config, vb_dav2)
            .expect("build dav2");
        log::info!("AI Engine: Depth Anything V2 ready!");

        // ============================================================
        // Stage 2: SDXL Turbo
        // ============================================================
        let dtype = DType::F32;
        let sd_config = stable_diffusion::StableDiffusionConfig::sdxl_turbo(
            None, Some(SD_HEIGHT), Some(SD_WIDTH),
        );

        log::info!("AI Engine: Building SDXL Turbo text embeddings...");
        let text_embeddings = build_text_embeddings(PROMPT, &sd_config, &device, dtype);
        log::info!("AI Engine: Text embeddings: {:?}", text_embeddings.shape());

        log::info!("AI Engine: Loading SDXL Turbo VAE...");
        let vae_path = api.model("madebyollin/sdxl-vae-fp16-fix".into())
            .get("diffusion_pytorch_model.safetensors").expect("vae weights");
        let vae = sd_config.build_vae(vae_path, &device, dtype).expect("build vae");
        log::info!("AI Engine: VAE ready!");

        log::info!("AI Engine: Loading SDXL Turbo UNet (~5GB)");
        let unet_path = api.model("stabilityai/sdxl-turbo".into())
            .get("unet/diffusion_pytorch_model.safetensors").expect("unet weights");
        let unet = sd_config.build_unet(unet_path, &device, 4, false, dtype).expect("build unet");
        log::info!("AI Engine: UNet ready!");

        log::info!("AI Engine: All models loaded successfully!");

        *MODEL.lock().unwrap() = Some(ModelState {
            depth_model,
            text_embeddings,
            unet,
            vae,
            sd_config,
            device,
        });
    }

    fn handle_cast(_src: InstanceId, msg: &[u8]) {
        let frame_id = FRAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        log::info!("AI Engine: === Frame {} === ({} bytes)", frame_id, msg.len());

        let guard = MODEL.lock().unwrap();
        let state = match guard.as_ref() {
            Some(s) => s,
            None => { log::error!("AI Engine: Model not initialized!"); return; }
        };

        // 1. Parse JSON HTTP response to get image bytes
        let msg_str = core::str::from_utf8(msg).unwrap_or("");
        let input_bytes: Vec<u8> = match serde_json::from_str::<serde_json::Value>(msg_str) {
            Ok(v) => match v.get("body") {
                Some(body) => serde_json::from_value::<Vec<u8>>(body.clone()).unwrap_or_default(),
                None => { log::error!("AI Engine: No body field"); return; }
            },
            Err(e) => { log::error!("AI Engine: JSON parse: {e}"); return; }
        };

        let img = match image::load_from_memory(&input_bytes) {
            Ok(i) => i,
            Err(e) => { log::error!("AI Engine: Image decode: {e}"); return; }
        };
        let orig_w = img.width() as usize;
        let orig_h = img.height() as usize;
        log::info!("AI Engine: Input {}x{}", orig_w, orig_h);

        // ============================================================
        // Stage 1: Depth Anything V2
        // ============================================================
        let t0 = std::time::Instant::now();

        let resized = img.resize_exact(
            DINO_IMG_SIZE as u32, DINO_IMG_SIZE as u32,
            image::imageops::FilterType::Triangle,
        );
        let rgb = resized.to_rgb8();
        let raw: Vec<f32> = rgb.as_raw().iter().map(|&b| b as f32).collect();

        let tensor = Tensor::from_vec(raw, (DINO_IMG_SIZE, DINO_IMG_SIZE, 3), &state.device).unwrap()
            .permute((2, 0, 1)).unwrap()
            .unsqueeze(0).unwrap()
            .to_dtype(DType::F32).unwrap();

        let max_v = Tensor::try_from(255.0f32).unwrap()
            .to_device(&state.device).unwrap()
            .broadcast_as(tensor.shape()).unwrap();
        let tensor = (tensor / max_v).unwrap();

        let mean = Tensor::from_vec(MAGIC_MEAN.to_vec(), (3, 1, 1), &state.device).unwrap()
            .broadcast_as(tensor.shape()).unwrap();
        let std_t = Tensor::from_vec(MAGIC_STD.to_vec(), (3, 1, 1), &state.device).unwrap()
            .broadcast_as(tensor.shape()).unwrap();
        let tensor = tensor.sub(&mean).unwrap().div(&std_t).unwrap();

        log::info!("AI Engine: Running depth estimation...");
        let depth = match state.depth_model.forward(&tensor) {
            Ok(d) => d,
            Err(e) => { log::error!("AI Engine: Depth failed: {e}"); return; }
        };
        let dt_depth = t0.elapsed();
        log::info!("AI Engine: Depth took {:.2}s, shape: {:?}", dt_depth.as_secs_f32(), depth.shape());

        // ============================================================
        // Stage 2: SDXL Turbo img2img (depth map -> reimagined scene)
        // ============================================================
        let t1 = std::time::Instant::now();

        // Resize depth to SD dimensions, normalize to [-1, 1], make 3-channel
        let depth_sd = depth.interpolate2d(SD_HEIGHT, SD_WIDTH).unwrap();
        let flat: Vec<f32> = depth_sd.flatten_all().unwrap().to_vec1().unwrap();
        let min_val = flat.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_val - min_val).max(1e-6);

        // Create 3-channel depth image in [-1, 1] for VAE encoding
        let depth_normalized: Vec<f32> = flat.iter()
            .map(|v| ((v - min_val) / range) * 2.0 - 1.0)
            .collect();
        let depth_3ch: Vec<f32> = depth_normalized.iter()
            .flat_map(|&v| [v, v, v])
            .collect();
        let depth_img = Tensor::from_vec(depth_3ch, (SD_HEIGHT, SD_WIDTH, 3), &state.device).unwrap()
            .permute((2, 0, 1)).unwrap()
            .unsqueeze(0).unwrap()
            .to_dtype(DType::F32).unwrap();

        // VAE encode depth image -> latent
        log::info!("AI Engine: VAE encoding depth image...");
        let init_latent_dist = state.vae.encode(&depth_img).unwrap();
        let init_latents = (init_latent_dist.sample().unwrap() * VAE_SCALE).unwrap()
            .to_device(&state.device).unwrap();

        // Build scheduler (1 step for Turbo)
        let n_steps: usize = 1;
        let mut scheduler = state.sd_config.build_scheduler(n_steps).unwrap();
        let timesteps = scheduler.timesteps().to_vec();
        let t_start = n_steps - (n_steps as f64 * IMG2IMG_STRENGTH) as usize;

        // Add noise to latents at starting timestep
        let latents = if t_start < timesteps.len() {
            let noise = init_latents.randn_like(0f64, 1f64).unwrap();
            scheduler.add_noise(&init_latents, noise, timesteps[t_start]).unwrap()
        } else {
            init_latents
        };
        let mut latents = latents.to_dtype(DType::F32).unwrap();

        // Run diffusion (1 step, no guidance)
        log::info!("AI Engine: Running SDXL Turbo diffusion...");
        for (ti, &timestep) in timesteps.iter().enumerate() {
            if ti < t_start { continue; }
            let ts = std::time::Instant::now();

            let input = scheduler.scale_model_input(latents.clone(), timestep).unwrap();
            let noise_pred = state.unet.forward(&input, timestep as f64, &state.text_embeddings).unwrap();
            latents = scheduler.step(&noise_pred, timestep, &latents).unwrap();

            log::info!("AI Engine: Step {}/{} took {:.2}s", ti + 1, n_steps, ts.elapsed().as_secs_f32());
        }

        // VAE decode -> RGB image
        log::info!("AI Engine: VAE decoding...");
        let images = state.vae.decode(&(latents / VAE_SCALE).unwrap()).unwrap();
        let images = ((images / 2.0).unwrap() + 0.5).unwrap()
            .to_device(&Device::Cpu).unwrap();
        let images = images.clamp(0f32, 1.0).unwrap()
            .mul(&Tensor::try_from(255.0f32).unwrap()).unwrap()
            .to_dtype(DType::U8).unwrap();

        let image_t = images.i(0).unwrap().permute((1, 2, 0)).unwrap();
        let (h, w) = (image_t.dim(0).unwrap(), image_t.dim(1).unwrap());
        let pixels: Vec<u8> = image_t.flatten_all().unwrap().to_vec1().unwrap();

        let dt_sd = t1.elapsed();
        log::info!("AI Engine: SDXL Turbo took {:.2}s", dt_sd.as_secs_f32());

        // Encode as PNG
        let mut png_bytes: Vec<u8> = Vec::new();
        {
            let encoder = image::codecs::png::PngEncoder::new(&mut png_bytes);
            image::ImageEncoder::write_image(
                encoder, &pixels, w as u32, h as u32, image::ColorType::Rgb8,
            ).unwrap();
        }

        // Send result
        let payload = ImagePayload { frame_id, png_data: png_bytes.clone() };
        let json_payload = serde_json::to_string(&payload).unwrap();
        log::info!("AI Engine: Frame {} done. Depth={:.2}s SD={:.2}s Total={:.2}s PNG={}B",
            frame_id, dt_depth.as_secs_f32(), dt_sd.as_secs_f32(),
            (dt_depth + dt_sd).as_secs_f32(), png_bytes.len());
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