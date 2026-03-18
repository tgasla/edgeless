use edgeless_function::*;
use edgeless_http::*;
use std::sync::atomic::{AtomicBool, Ordering};

static READY: AtomicBool = AtomicBool::new(false);

struct SdxlWebSender;
impl EdgeFunction for SdxlWebSender {
    fn handle_init(_p: Option<&[u8]>, _m: Option<&[u8]>) {
        init_logger();
        log::info!("WebSender: Waiting for output mapping to be ready...");
        // Wait for output_mapping to be patched before trying to call http_out
        delayed_cast(5000, "self", b"ready");
    }

    fn handle_cast(src: InstanceId, msg: &[u8]) {
        let msg_str = core::str::from_utf8(msg).unwrap_or("");

        // Handle the "ready" message - output_mapping should now be set
        if msg_str == "ready" {
            READY.store(true, Ordering::SeqCst);
            log::info!("WebSender: Ready to send HTTP requests");
            return;
        }

        // Only send if ready
        if !READY.load(Ordering::SeqCst) {
            log::warn!("WebSender: Received message but not ready yet, dropping");
            return;
        }

        // Send the JSON result payload back to the Python Webhook
        let req = EdgelessHTTPRequest {
            protocol: EdgelessHTTPProtocol::HTTP,
            host: "127.0.0.1:8080".to_string(),
            path: "/webhook".to_string(),
            method: EdgelessHTTPMethod::Post,
            headers: std::collections::HashMap::new(),
            body: Some(msg.to_vec()),
        };

        match call("http_out", edgeless_http::request_to_string(&req).as_bytes()) {
            CallRet::Reply(_) => log::info!("WebSender: Response sent to webhook"),
            CallRet::Err => log::error!("WebSender: Failed to reach webhook"),
            CallRet::NoReply => {}
        }
    }
    fn handle_call(_src: InstanceId, _msg: &[u8]) -> CallRet {
        CallRet::NoReply
    }
    fn handle_stop() {}
}
edgeless_function::export!(SdxlWebSender);
