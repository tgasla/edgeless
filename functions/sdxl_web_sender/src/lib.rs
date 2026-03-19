use edgeless_function::*;
use edgeless_http::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

static SESSION_ID: OnceLock<String> = OnceLock::new();

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

        // Also save to database for history
        // Parse the message to extract session_id, source_image, prompt, and generated_image
        if let Ok(data) = serde_json::from_str::<serde_json::Value>(msg_str) {
            let session_id = data.get("id").and_then(|v| v.as_str()).unwrap_or("default").to_string();
            let source_image = data.get("source_image_b64").and_then(|v| v.as_str()).unwrap_or("");
            let prompt = data.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
            let generated_image = data.get("image_base64").and_then(|v| v.as_str()).unwrap_or("");
            let timestep = data.get("timestep").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

            let save_request = serde_json::json!({
                "session_id": session_id,
                "source_image_b64": source_image,
                "prompt": prompt,
                "generated_image_b64": generated_image,
                "timestep": timestep
            });

            let save_msg = format!("SAVE:{}", serde_json::to_string(&save_request).unwrap_or_default());
            cast("db_writer", save_msg.as_bytes());
            log::info!("WebSender: Sent data to db_writer for history");
        }
    }
    fn handle_call(_src: InstanceId, _msg: &[u8]) -> CallRet {
        CallRet::NoReply
    }
    fn handle_stop() {}
}
edgeless_function::export!(SdxlWebSender);
