use edgeless_function::*;

struct Visualizer;

impl EdgeFunction for Visualizer {
    fn handle_init(payload: Option<&[u8]>, init_metadata: Option<&[u8]>) {
        log::info!("Visualizer: Initialized");
    }

    fn handle_cast(src: edgeless_function::InstanceId, msg: &[u8]) {
        // Here we would typically route this to an Edgeless Data Egress linked to an external dashboard or file sink.
        log::info!("Visualizer: Successfully received transformed AI frame from GB10!");
        if let Ok(data) = std::str::from_utf8(msg) {
            log::info!("Frame content preview: {}", data);
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

edgeless_function::export!(Visualizer);
