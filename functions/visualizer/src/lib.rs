use edgeless_function::*;
use edgeless_http::*;
use std::sync::atomic::{AtomicU32, Ordering};

struct Visualizer;

/// Must match the struct in ai_engine
#[derive(serde::Deserialize)]
struct ImagePayload {
    frame_id: u32,
    png_data: Vec<u8>,
}

impl EdgeFunction for Visualizer {
    fn handle_init(_payload: Option<&[u8]>, _init_metadata: Option<&[u8]>) {
        edgeless_function::init_logger();
        log::info!("Visualizer: Initialized - will save frames via HTTP POST");
        // Wait for output_mapping to be patched before trying to call http_out
        delayed_cast(5000, "self", b"ready");
    }

    fn handle_cast(_src: InstanceId, msg: &[u8]) {
        let msg_str = core::str::from_utf8(msg).unwrap_or("");

        // Skip the initial "ready" delayed cast - it just waits for output_mapping to be set
        if msg_str == "ready" {
            return;
        }

        // Parse the ImagePayload JSON from ai_engine
        let payload: ImagePayload = match serde_json::from_str(msg_str) {
            Ok(p) => p,
            Err(e) => {
                log::info!("Visualizer: Failed to parse image payload: {}", e);
                return;
            }
        };

        log::info!(
            "Visualizer: Received frame {} ({} PNG bytes), uploading via HTTP",
            payload.frame_id,
            payload.png_data.len()
        );

        // POST the PNG bytes to a local file-saving HTTP server
        let http_req = edgeless_http::EdgelessHTTPRequest {
            protocol: edgeless_http::EdgelessHTTPProtocol::HTTP,
            host: "127.0.0.1:8081".to_string(),
            path: format!("/save/frame_{:04}.png", payload.frame_id),
            method: edgeless_http::EdgelessHTTPMethod::Post,
            headers: std::collections::HashMap::from([("Content-Type".to_string(), "image/png".to_string())]),
            body: Some(payload.png_data),
        };

        let response = call("http_out", edgeless_http::request_to_string(&http_req).as_bytes());

        match response {
            CallRet::Reply(_) => {
                log::info!("Visualizer: Frame {} saved successfully", payload.frame_id);
            }
            CallRet::NoReply => {
                log::info!("Visualizer: No reply when saving frame {}", payload.frame_id);
            }
            CallRet::Err => {
                log::info!("Visualizer: Error saving frame {}", payload.frame_id);
            }
        }
    }

    fn handle_call(_src: InstanceId, _msg: &[u8]) -> CallRet {
        CallRet::NoReply
    }

    fn handle_stop() {}
}

edgeless_function::export!(Visualizer);
