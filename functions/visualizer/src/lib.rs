use edgeless_function::*;
use edgeless_http::*;
use std::sync::atomic::{AtomicU32, Ordering};

struct Visualizer;

/// Stored frame waiting for http_out to be ready
static PENDING_FRAMES: std::sync::Mutex<Vec<(u32, Vec<u8>)>> = std::sync::Mutex::new(Vec::new());
static READY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Must match the struct in ai_engine
#[derive(serde::Deserialize)]
struct ImagePayload {
    frame_id: u32,
    png_data: Vec<u8>,
}

impl Visualizer {
    fn save_frame(frame_id: u32, png_data: Vec<u8>) {
        log::info!("Visualizer: Saving frame {} ({} PNG bytes)", frame_id, png_data.len());

        let http_req = edgeless_http::EdgelessHTTPRequest {
            protocol: edgeless_http::EdgelessHTTPProtocol::HTTP,
            host: "127.0.0.1:8081".to_string(),
            path: format!("/save/frame_{:04}.png", frame_id),
            method: edgeless_http::EdgelessHTTPMethod::Post,
            headers: std::collections::HashMap::from([("Content-Type".to_string(), "image/png".to_string())]),
            body: Some(png_data),
        };

        let response = call("http_out", edgeless_http::request_to_string(&http_req).as_bytes());

        match response {
            CallRet::Reply(_) => {
                log::info!("Visualizer: Frame {} saved successfully", frame_id);
            }
            CallRet::NoReply => {
                log::info!("Visualizer: No reply when saving frame {}", frame_id);
            }
            CallRet::Err => {
                log::info!("Visualizer: Error saving frame {}", frame_id);
            }
        }
    }
}

impl EdgeFunction for Visualizer {
    fn handle_init(_payload: Option<&[u8]>, _init_metadata: Option<&[u8]>) {
        edgeless_function::init_logger();
        READY.store(false, std::sync::atomic::Ordering::SeqCst);
        log::info!("Visualizer: Initialized - will save frames via HTTP POST");
        // Wait for output_mapping to be patched before trying to call http_out
        delayed_cast(5000, "self", b"ready");
    }

    fn handle_cast(_src: InstanceId, msg: &[u8]) {
        let msg_str = core::str::from_utf8(msg).unwrap_or("");

        // Handle the "ready" message - output_mapping should now be set
        if msg_str == "ready" {
            log::info!("Visualizer: Ready signal received, processing queued frames");
            READY.store(true, std::sync::atomic::Ordering::SeqCst);
            // Process any queued frames
            let frames = PENDING_FRAMES.lock().unwrap().clone();
            for (frame_id, png_data) in frames {
                Self::save_frame(frame_id, png_data);
            }
            return;
        }

        // If not ready yet, queue the frame for later
        if !READY.load(std::sync::atomic::Ordering::SeqCst) {
            // Parse and queue
            let payload: ImagePayload = match serde_json::from_str(msg_str) {
                Ok(p) => p,
                Err(e) => {
                    log::info!("Visualizer: Failed to parse image payload: {}", e);
                    return;
                }
            };
            log::info!("Visualizer: Queuing frame {} (waiting for http_out)", payload.frame_id);
            PENDING_FRAMES.lock().unwrap().push((payload.frame_id, payload.png_data));
            return;
        }

        // Parse and save immediately since we're ready
        let payload: ImagePayload = match serde_json::from_str(msg_str) {
            Ok(p) => p,
            Err(e) => {
                log::info!("Visualizer: Failed to parse image payload: {}", e);
                return;
            }
        };

        Self::save_frame(payload.frame_id, payload.png_data);
    }

    fn handle_call(_src: InstanceId, _msg: &[u8]) -> CallRet {
        CallRet::NoReply
    }

    fn handle_stop() {}
}

edgeless_function::export!(Visualizer);
