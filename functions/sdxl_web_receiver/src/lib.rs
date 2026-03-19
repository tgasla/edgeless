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

        // Check if this is a history request (path /history)
        if req.path == "/history" {
            log::info!("WebReceiver: Received history request, calling db_reader");
            // Use call (synchronous) to get history from db_reader
            match call("db_reader", b"GET_HISTORY") {
                CallRet::Reply(data) => {
                    log::info!("WebReceiver: Got history data from db_reader");
                    let history_response = EdgelessHTTPResponse {
                        status: 200,
                        body: Some(data.to_vec()),
                        headers: std::collections::HashMap::new(),
                    };
                    return CallRet::Reply(OwnedByteBuff::new_from_slice(response_to_string(&history_response).as_bytes()));
                }
                CallRet::NoReply => {
                    log::warn!("WebReceiver: db_reader returned no reply");
                    let err_response = EdgelessHTTPResponse {
                        status: 500,
                        body: Some(b"Database error".to_vec()),
                        headers: std::collections::HashMap::new(),
                    };
                    return CallRet::Reply(OwnedByteBuff::new_from_slice(response_to_string(&err_response).as_bytes()));
                }
                CallRet::Err => {
                    log::error!("WebReceiver: db_reader returned error");
                    let err_response = EdgelessHTTPResponse {
                        status: 500,
                        body: Some(b"Database error".to_vec()),
                        headers: std::collections::HashMap::new(),
                    };
                    return CallRet::Reply(OwnedByteBuff::new_from_slice(response_to_string(&err_response).as_bytes()));
                }
            }
        }

        // Check if this is a generate request
        if req.method == EdgelessHTTPMethod::Post {
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
            return CallRet::Reply(OwnedByteBuff::new_from_slice(response_to_string(&ok_response).as_bytes()));
        }

        // For other methods
        let err_response = EdgelessHTTPResponse {
            status: 405,
            body: Some(b"Method not allowed".to_vec()),
            headers: std::collections::HashMap::new(),
        };
        CallRet::Reply(OwnedByteBuff::new_from_slice(response_to_string(&err_response).as_bytes()))
    }
    fn handle_cast(_src: InstanceId, _msg: &[u8]) {}
    fn handle_stop() {}
}
edgeless_function::export!(SdxlWebReceiver);
