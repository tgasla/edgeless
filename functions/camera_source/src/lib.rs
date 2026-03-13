use edgeless_function::*;

struct CameraSource;

impl EdgeFunction for CameraSource {
    fn handle_init(payload: Option<&[u8]>, init_metadata: Option<&[u8]>) {
        log::info!("Camera Source: Initialized");
        // Start streaming frames. In a real scenario, this would interact with
        // an ingress resource attached to host hardware. For this demo,
        // we can cast periodic requests or simulate streaming.
        edgeless_function::delayed_cast(
            1000,
            "raw_image_channel",
            b"SIMULATED_FRAME_START",
        );
    }

    fn handle_cast(src: edgeless_function::InstanceId, msg: &[u8]) {
        // If it's our own simulation loop, cast and schedule the next
        log::info!("Camera Source: capturing sending frame to AI Engine");
        
        let prompt = "A cyberpunk city street at night";
        let payload = format!("{{\"prompt\": \"{}\", \"image\": \"...\"}}", prompt);
        
        edgeless_function::cast("raw_image_channel", payload.as_bytes());
        
        // Loop simulation
        edgeless_function::delayed_cast(
            1000,
            "raw_image_channel",
            b"SIMULATED_FRAME_START",
        );
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
