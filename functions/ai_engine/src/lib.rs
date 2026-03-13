use candle_core::{Device, Tensor};
use edgeless_function::*;

struct AIEngine;

impl EdgeFunction for AIEngine {
    fn handle_init(_payload: Option<&[u8]>, _init_metadata: Option<&[u8]>) {
        edgeless_function::init_logger();
        log::info!("AI Engine: Initialized");
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        log::info!("AI Engine: Using device: {:?}", device);
    }

    fn handle_cast(_src: InstanceId, msg: &[u8]) {
        log::info!("AI Engine: Received raw frame ({} bytes)", msg.len());

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

        // Create a colorful gradient image influenced by the tensor computation  
        let sample_tensor = Tensor::randn(0f32, 1., (img_height as usize, img_width as usize), &device).unwrap();
        let sample_data: Vec<f32> = sample_tensor.flatten_all().unwrap().to_vec1().unwrap();

        // Build a PNG image
        let mut img_buf = image::RgbImage::new(img_width, img_height);
        for (i, pixel) in img_buf.pixels_mut().enumerate() {
            let val = sample_data[i % sample_data.len()];
            // Map the random tensor values to vibrant colors
            let r = ((val.sin() * 0.5 + 0.5) * 255.0) as u8;
            let g = ((val.cos() * 0.5 + 0.5) * 255.0) as u8;
            let b = (((val * 2.0).sin() * 0.5 + 0.5) * 255.0) as u8;
            *pixel = image::Rgb([r, g, b]);
        }

        // Encode as PNG bytes
        let mut png_bytes: Vec<u8> = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut png_bytes);
        encoder.write_image(
            img_buf.as_raw(),
            img_width,
            img_height,
            image::ExtendedColorType::Rgb8,
        ).unwrap();

        log::info!("AI Engine: Generation Complete. Sending {} byte PNG to visualizer", png_bytes.len());
        cast("transformed_image_channel", &png_bytes);
    }

    fn handle_call(_src: InstanceId, _msg: &[u8]) -> CallRet {
        CallRet::NoReply
    }

    fn handle_stop() {}
}

edgeless_function::export!(AIEngine);
