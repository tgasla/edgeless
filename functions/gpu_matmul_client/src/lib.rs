// SPDX-FileCopyrightText: © 2024 Edgeless Project
// SPDX-License-Identifier: MIT
use edgeless_function::*;

struct GpuClientFun;

impl EdgeFunction for GpuClientFun {
    fn handle_cast(_src: InstanceId, encoded_message: &[u8]) {
        let str_message = core::str::from_utf8(encoded_message).unwrap_or("");
        log::info!("GpuClient: 'Cast' from {}, MSG: {}", _src, str_message);
        
        if str_message == "start" {
            log::info!("GpuClient: Triggering worker...");
            let res = call("worker", b"run_matmul");
            match res {
                CallRet::Reply(msg) => {
                    if let Ok(msg_str) = core::str::from_utf8(&msg) {
                        log::info!("GpuClient: Worker replied: {}", msg_str);
                    }
                }
                CallRet::NoReply => {
                    log::info!("GpuClient: Worker returned NoReply");
                }
                CallRet::Err => {
                    log::info!("GpuClient: Worker returned Err");
                }
            }
            
            // Re-trigger every 10 seconds to keep the demo running interactively
            delayed_cast(10000, "self", b"start");
        }
    }

    fn handle_call(_src: InstanceId, encoded_message: &[u8]) -> CallRet {
        let str_message = core::str::from_utf8(encoded_message).unwrap_or("");
        log::info!("GpuClient: 'Call' called, MSG: {}", str_message);
        CallRet::NoReply
    }

    fn handle_init(_payload: Option<&[u8]>, _serialized_state: Option<&[u8]>) {
        edgeless_function::init_logger();
        log::info!("GpuClient: 'Init' called. Starting computation loop.");
        
        // Trigger the computation loop soon after init
        delayed_cast(2000, "self", b"start");
    }

    fn handle_stop() {
        log::info!("GpuClient: 'Stop' called");
    }
}

edgeless_function::export!(GpuClientFun);
