use edgeless_function::*;

struct SdxlWebReceiver;
impl EdgeFunction for SdxlWebReceiver {
    fn handle_init(_p: Option<&[u8]>, _m: Option<&[u8]>) {
        init_logger();
    }

    fn handle_call(_src: InstanceId, msg: &[u8]) -> CallRet {
        // We receive the JSON payload from the http-ingress.
        // Forward it immediately to the AI Engine on the MSI node.
        cast("ai_channel", msg);

        // Reply instantly to release the HTTP connection to Python
        CallRet::Reply(b"Forwarded to AI Engine")
    }
    fn handle_cast(_src: InstanceId, _msg: &[u8]) {}
    fn handle_stop() {}
}
edgeless_function::export!(SdxlWebReceiver);
