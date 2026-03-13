use edgeless_function::*;
use edgeless_http::*;

struct CameraSource;

static FRAME_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

impl EdgeFunction for CameraSource {
    fn handle_init(_payload: Option<&[u8]>, _init_metadata: Option<&[u8]>) {
        edgeless_function::init_logger();
        log::info!("Camera Source: Initialized - will fetch frames from local HTTP server");
        // Kick off the frame capture loop
        delayed_cast(2000, "self", b"capture");
    }

    fn handle_cast(_src: InstanceId, msg: &[u8]) {
        let str_message = core::str::from_utf8(msg).unwrap_or("");

        if str_message == "capture" {
            let frame_num = FRAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            log::info!("Camera Source: Fetching frame {} from HTTP server", frame_num);

            // Use http-egress to fetch an image from a local HTTP file server
            // Run on Macbook: python3 -m http.server 8080 --directory /path/to/images
            let http_req = edgeless_http::EdgelessHTTPRequest {
                protocol: edgeless_http::EdgelessHTTPProtocol::HTTP,
                host: "127.0.0.1:8080".to_string(),
                path: "/frame.png".to_string(),
                method: edgeless_http::EdgelessHTTPMethod::Get,
                headers: std::collections::HashMap::new(),
                body: None,
            };

            let response = call("http_e", edgeless_http::request_to_string(&http_req).as_bytes());

            match response {
                CallRet::Reply(resp_bytes) => {
                    let parsed = edgeless_http::response_from_string(
                        core::str::from_utf8(&resp_bytes).unwrap_or("")
                    );
                    match parsed {
                        Ok(http_resp) => {
                            if http_resp.status == 200 {
                                if let Some(body) = http_resp.body {
                                    log::info!("Camera Source: Got image frame {} ({} bytes), sending to AI Engine", frame_num, body.len());
                                    cast("raw_image_channel", &body);
                                } else {
                                    log::info!("Camera Source: Empty body in HTTP response");
                                }
                            } else {
                                log::info!("Camera Source: HTTP error status {}", http_resp.status);
                            }
                        }
                        Err(e) => {
                            log::info!("Camera Source: Failed to parse HTTP response");
                        }
                    }
                }
                CallRet::NoReply => {
                    log::info!("Camera Source: No reply from HTTP egress");
                }
                CallRet::Err => {
                    log::info!("Camera Source: Error fetching image from HTTP server");
                }
            }

            // Schedule next frame capture (every 2 seconds)
            delayed_cast(2000, "self", b"capture");
        }
    }

    fn handle_call(_src: InstanceId, _msg: &[u8]) -> CallRet {
        CallRet::NoReply
    }

    fn handle_stop() {}
}

edgeless_function::export!(CameraSource);
