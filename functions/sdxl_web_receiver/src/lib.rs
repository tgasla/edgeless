use edgeless_function::owned_data::OwnedByteBuff;
use edgeless_function::*;
use edgeless_http::*;

struct SdxlWebReceiver;
impl EdgeFunction for SdxlWebReceiver {
    fn handle_init(_p: Option<&[u8]>, _m: Option<&[u8]>) {
        init_logger();
    }

    fn handle_call(_src: InstanceId, msg: &[u8]) -> CallRet {
        // We receive an EdgelessHTTPRequest from the http-ingress.
        // Extract the body and forward it to the AI Engine.
        let msg_str = core::str::from_utf8(msg).unwrap_or("");

        // Parse the EdgelessHTTPRequest
        let req: EdgelessHTTPRequest = match serde_json::from_str(msg_str) {
            Ok(r) => r,
            Err(e) => {
                log::error!("Failed to parse HTTP request: {}", e);
                let err_response = EdgelessHTTPResponse {
                    status: 400,
                    body: Some(b"Error parsing request".to_vec()),
                    headers: std::collections::HashMap::new(),
                };
                return CallRet::Reply(OwnedByteBuff::new_from_slice(response_to_string(&err_response).as_bytes()));
            }
        };

        // Extract the body and forward it
        if let Some(body) = req.body {
            cast("ai_channel", &body);
        }

        // Reply with proper EdgelessHTTPResponse to release the HTTP connection to Python
        let ok_response = EdgelessHTTPResponse {
            status: 202,
            body: Some(b"Accepted - processing started".to_vec()),
            headers: std::collections::HashMap::new(),
        };
        CallRet::Reply(OwnedByteBuff::new_from_slice(response_to_string(&ok_response).as_bytes()))
    }
    fn handle_cast(_src: InstanceId, _msg: &[u8]) {}
    fn handle_stop() {}
}
edgeless_function::export!(SdxlWebReceiver);
