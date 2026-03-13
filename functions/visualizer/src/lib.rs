use edgeless_function::*;
use edgeless_http::*;
use std::sync::atomic::{AtomicU32, Ordering};

struct Visualizer;

static FRAME_COUNTER: AtomicU32 = AtomicU32::new(0);

impl EdgeFunction for Visualizer {
    fn handle_init(_payload: Option<&[u8]>, _init_metadata: Option<&[u8]>) {
        edgeless_function::init_logger();
        log::info!("Visualizer: Initialized - will save frames via HTTP POST");
    }

    fn handle_cast(_src: InstanceId, msg: &[u8]) {
        let frame_num = FRAME_COUNTER.fetch_add(1, Ordering::SeqCst);
        log::info!("Visualizer: Received frame {} ({} bytes), uploading via HTTP", frame_num, msg.len());

        // POST the PNG bytes to a local file-saving HTTP server
        let http_req = edgeless_http::EdgelessHTTPRequest {
            protocol: edgeless_http::EdgelessHTTPProtocol::HTTP,
            host: "127.0.0.1:8081".to_string(),
            path: format!("/save/frame_{:04}.png", frame_num),
            method: edgeless_http::EdgelessHTTPMethod::Post,
            headers: std::collections::HashMap::from([
                ("Content-Type".to_string(), "image/png".to_string()),
            ]),
            body: Some(msg.to_vec()),
        };

        let response = call("http_out", edgeless_http::request_to_string(&http_req).as_bytes());

        match response {
            CallRet::Reply(resp_bytes) => {
                log::info!("Visualizer: Frame {} saved successfully", frame_num);
            }
            CallRet::NoReply => {
                log::info!("Visualizer: No reply when saving frame {}", frame_num);
            }
            CallRet::Err => {
                log::info!("Visualizer: Error saving frame {}", frame_num);
            }
        }
    }

    fn handle_call(_src: InstanceId, _msg: &[u8]) -> CallRet {
        CallRet::NoReply
    }

    fn handle_stop() {}
}

edgeless_function::export!(Visualizer);
