use edgeless_function::*;

struct CameraSource;

impl EdgeFunction for CameraSource {
    fn handle_init(payload: Option<&[u8]>, _init_metadata: Option<&[u8]>) {
        edgeless_function::init_logger();
        log::info!("Camera Source: Initialized");
        // Kick off the frame capture loop by sending a delayed message to ourselves
        edgeless_function::delayed_cast(1000, "self", b"capture");
    }

    fn handle_cast(_src: edgeless_function::InstanceId, msg: &[u8]) {
        let str_message = core::str::from_utf8(msg).unwrap_or("");
        
        if str_message == "capture" {
            log::info!("Camera Source: Capturing frame, sending to AI Engine");
            
            let prompt = "A cyberpunk city street at night";
            let payload = format!("{{\"prompt\": \"{}\", \"image\": \"SIMULATED_RAW_FRAME_DATA\"}}", prompt);
            
            // Send the frame to the ai_engine via the output channel
            edgeless_function::cast("raw_image_channel", payload.as_bytes());
            
            // Schedule the next frame capture (loop back to ourselves)
            edgeless_function::delayed_cast(1000, "self", b"capture");
        }
    }

    fn handle_call(
        _src: edgeless_function::InstanceId,
        _msg: &[u8],
    ) -> edgeless_function::CallRet {
        edgeless_function::CallRet::NoReply
    }
    
    fn handle_stop() {}
}

edgeless_function::export!(CameraSource);
