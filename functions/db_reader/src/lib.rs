// db_reader - reads image history from SQLx database

use edgeless_function::*;
use edgeless_function::owned_data::OwnedByteBuff;

#[derive(serde::Serialize, serde::Deserialize)]
struct HistoryEntry {
    id: Option<i64>,
    session_id: String,
    source_image_b64: String,
    prompt: String,
    generated_image_b64: String,
    timestep: u32,
    created_at: String,
}

#[derive(serde::Serialize)]
struct QueryResponse {
    entries: Vec<HistoryEntry>,
}

fn call_wrapper(msg: &str) -> Option<String> {
    match call("database", msg.as_bytes()) {
        CallRet::Reply(data) => {
            let reply = std::str::from_utf8(&data).map(|s| s.to_string()).unwrap_or_default();
            Some(reply)
        }
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

            let query = "SELECT id, session_id, source_image_b64, prompt, generated_image_b64, timestep, created_at FROM image_history ORDER BY id DESC LIMIT 20";

            if let Some(result) = call_wrapper(query) {
                log::info!("db_reader: got history data from database");

                let mut entries = Vec::new();
                for line in result.lines().skip(1) {
                    let parts: Vec<&str> = line.split('|').collect();
                    if parts.len() >= 7 {
                        entries.push(HistoryEntry {
                            id: parts[0].trim().parse().ok(),
                            session_id: parts[1].trim().to_string(),
                            source_image_b64: parts[2].trim().to_string(),
                            prompt: parts[3].trim().to_string(),
                            generated_image_b64: parts[4].trim().to_string(),
                            timestep: parts[5].trim().parse().unwrap_or(0),
                            created_at: parts[6].trim().to_string(),
                        });
                    }
                }

                let response = QueryResponse { entries };
                if let Ok(json) = serde_json::to_string(&response) {
                    return CallRet::Reply(OwnedByteBuff::new_from_slice(json.as_bytes()));
                }
            }

            log::warn!("db_reader: query returned no result");
            return CallRet::Reply(OwnedByteBuff::new_from_slice(b"[]"));
        }

        log::warn!("db_reader: unknown handle_call request: {}", msg_str);
        CallRet::NoReply
    }

    fn handle_stop() {
        log::info!("db_reader: stopping");
    }
}

edgeless_function::export!(DbReader);
