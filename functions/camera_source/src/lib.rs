use edgeless_function::*;

struct CameraSource;

impl EdgeFunction for CameraSource {
    fn handle_init(&mut self, payload: Option<&[u8]>, init_metadata: Option<edgeless_function::InitMetadata>) {
        log::info!("Camera Source: Initialized");
        // Start streaming frames. In a real scenario, this would interact with
        // an ingress resource attached to host hardware. For this demo,
        // we can cast periodic requests or simulate streaming.
        edgeless_function::delayed_cast(
            edgeless_function::InstanceId::none(),
            "raw_image_channel",
            b"SIMULATED_FRAME_START",
            1000,
        );
    }

    fn handle_cast(&mut self, src: edgeless_function::InstanceId, msg: &[u8]) {
        // If it's our own simulation loop, cast and schedule the next
        log::info!("Camera Source: capturing sending frame to AI Engine");
        
        let prompt = "A cyberpunk city street at night";
        let payload = format!("{{\"prompt\": \"{}\", \"image\": \"...\"}}", prompt);
        
        edgeless_function::cast("raw_image_channel", payload.as_bytes());
        
        // Loop simulation
        edgeless_function::delayed_cast(
            edgeless_function::InstanceId::none(),
            "raw_image_channel", // Here, we could cast to self if we had an output mapped to ourselves, or handle it via external timer.
            b"SIMULATED_FRAME_START",
            1000,
        );
    }

    fn handle_call(
        &mut self,
        _src: edgeless_function::InstanceId,
        _msg: &[u8],
    ) -> edgeless_function::CallRet {
        edgeless_function::CallRet::NoReply
    }
}

edgeless_function::export!(CameraSource);
