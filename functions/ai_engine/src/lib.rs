use candle_core::{Device, Tensor};
use edgeless_function::*;

struct AIEngine;

/// Simple JSON wrapper for image data, since the dataplane
/// treats all payloads as strings (UTF-8). Raw PNG bytes would
/// fail, so we JSON-serialize the Vec<u8> (becomes a JSON array of numbers).
#[derive(serde::Serialize, serde::Deserialize)]
struct ImagePayload {
    frame_id: u32,
    png_data: Vec<u8>,
}

static FRAME_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

impl EdgeFunction for AIEngine {
    fn handle_init(_payload: Option<&[u8]>, _init_metadata: Option<&[u8]>) {
        edgeless_function::init_logger();
        log::info!("AI Engine: Initialized");
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        log::info!("AI Engine: Using device: {:?}", device);
    }

    fn handle_cast(_src: InstanceId, msg: &[u8]) {
        let frame_id = FRAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        log::info!("AI Engine: Received message ({} bytes) for frame {}", msg.len(), frame_id);

        // Parse the incoming JSON HTTP response to get the image body
        let msg_str = core::str::from_utf8(msg).unwrap_or("");
        let _input_body: Option<Vec<u8>> = if let Ok(http_resp) = serde_json::from_str::<serde_json::Value>(msg_str) {
            if let Some(body) = http_resp.get("body") {
                if let Ok(bytes) = serde_json::from_value::<Vec<u8>>(body.clone()) {
                    log::info!("AI Engine: Parsed input image ({} bytes)", bytes.len());
                    Some(bytes)
                } else { None }
            } else { None }
        } else { None };

        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        log::info!("AI Engine: Processing on device: {:?}", device);

        // 1. DEPTH ESTIMATION STAGE (MiDaS simulation)
        log::info!("AI Engine [Stage 1]: Running Depth Estimation Model (MiDaS)");
        let weight_1 = Tensor::randn(0f32, 1., (512, 512), &device).unwrap();
        let mut feature_map = Tensor::randn(0f32, 1., (512, 512), &device).unwrap();
        for _ in 0..10 {
            feature_map = feature_map.matmul(&weight_1).unwrap();
        }
        log::info!("AI Engine [Stage 1]: Depth estimation complete");

        // 2. GENERATION STAGE (ControlNet simulation)
        log::info!("AI Engine [Stage 2]: Running SD Inpainting with Depth ControlNet");
        let unet_weight = Tensor::randn(0f32, 1., (1024, 1024), &device).unwrap();
        let mut latent = Tensor::randn(0f32, 1., (1024, 1024), &device).unwrap();
        for step in 1..=20 {
            log::info!("AI Engine: Diffusion Step {}/20", step);
            latent = latent.matmul(&unet_weight).unwrap();
        }

        // 3. Generate an actual output image from tensor data
        log::info!("AI Engine [Stage 3]: Generating output image from tensor");
        let img_width: u32 = 256;
        let img_height: u32 = 256;

        let sample_tensor = Tensor::randn(0f32, 1., (img_height as usize, img_width as usize), &device).unwrap();
        let sample_data: Vec<f32> = sample_tensor.flatten_all().unwrap().to_vec1().unwrap();

        // Build a PNG image
        let mut img_buf = image::RgbImage::new(img_width, img_height);
        for (i, pixel) in img_buf.pixels_mut().enumerate() {
            let val = sample_data[i % sample_data.len()];
            let r = ((val.sin() * 0.5 + 0.5) * 255.0) as u8;
            let g = ((val.cos() * 0.5 + 0.5) * 255.0) as u8;
            let b = (((val * 2.0).sin() * 0.5 + 0.5) * 255.0) as u8;
            *pixel = image::Rgb([r, g, b]);
        }

        // Encode as PNG bytes
        let mut png_bytes: Vec<u8> = Vec::new();
        {
            let encoder = image::codecs::png::PngEncoder::new(&mut png_bytes);
            image::ImageEncoder::write_image(
                encoder,
                img_buf.as_raw(),
                img_width,
                img_height,
                image::ColorType::Rgb8,
            ).unwrap();
        }

        // Wrap in JSON so the payload is UTF-8 safe for the dataplane
        let payload = ImagePayload {
            frame_id,
            png_data: png_bytes.clone(),
        };
        let json_payload = serde_json::to_string(&payload).unwrap();

        log::info!("AI Engine: Generation Complete. Sending {} byte PNG (frame {}) as {} byte JSON to visualizer",
            png_bytes.len(), frame_id, json_payload.len());
        cast("transformed_image_channel", json_payload.as_bytes());
    }

    fn handle_call(_src: InstanceId, _msg: &[u8]) -> CallRet {
        CallRet::NoReply
    }

    fn handle_stop() {}
}

edgeless_function::export!(AIEngine);
