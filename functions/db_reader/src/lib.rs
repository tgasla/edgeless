// db_reader - reads image history from SQLx database

use edgeless_function::*;
use edgeless_function::owned_data::OwnedByteBuff;

fn call_wrapper(msg: &str) -> Option<OwnedByteBuff> {
    match call("database", msg.as_bytes()) {
        CallRet::Reply(data) => Some(data),
        CallRet::NoReply => {
            log::warn!("db_reader: received empty reply from database");
            None
        }
        CallRet::Err => {
            log::error!("db_reader: error calling database");
            None
        }
    }
}

struct DbReader;

impl EdgeFunction for DbReader {
    fn handle_init(_init_message: Option<&[u8]>, _serialized_state: Option<&[u8]>) {
        edgeless_function::init_logger();
        log::info!("db_reader: initialized");
    }

    fn handle_cast(_src: InstanceId, message: &[u8]) {
        let msg_str = core::str::from_utf8(message).unwrap_or("");
        log::info!("db_reader: received cast: {}", msg_str);
    }

    fn handle_call(_src: InstanceId, message: &[u8]) -> CallRet {
        let msg_str = core::str::from_utf8(message).unwrap_or("");

        if msg_str.starts_with("GET_HISTORY") {
            log::info!("db_reader: received GET_HISTORY request");

            // Query all records from image_history
            let query = "SELECT id, session_id, source_image_b64, prompt, generated_image_b64, timestep, created_at FROM image_history ORDER BY id DESC LIMIT 20";

            if let Some(result) = call_wrapper(query) {
                log::info!("db_reader: got history data from database");
                return CallRet::Reply(result);
            } else {
                log::warn!("db_reader: query returned no result");
                return CallRet::Reply(OwnedByteBuff::new_from_slice(b"[]"));
            }
        }

        log::warn!("db_reader: unknown handle_call request: {}", msg_str);
        CallRet::NoReply
    }

    fn handle_stop() {
        log::info!("db_reader: stopping");
    }
}

edgeless_function::export!(DbReader);
