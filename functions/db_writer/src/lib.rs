// db_writer - receives image generation data and saves to SQLx database

use edgeless_function::*;

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct SaveRequest {
    session_id: String,
    source_image_b64: String,
    prompt: String,
    generated_image_b64: String,
    timestep: u32,
}

fn call_wrapper(msg: &str) -> Option<String> {
    match call("database", msg.as_bytes()) {
        CallRet::Reply(msg) => {
            let reply = std::str::from_utf8(&msg).unwrap_or("not UTF8");
            log::info!("db_writer got reply: {}", reply);
            Some(reply.to_string())
        }
        CallRet::NoReply => {
            log::warn!("db_writer: received empty reply from database");
            None
        }
        CallRet::Err => {
            log::error!("db_writer: error calling database");
            None
        }
    }
}

struct DbWriter;

impl EdgeFunction for DbWriter {
    fn handle_init(_init_message: Option<&[u8]>, _serialized_state: Option<&[u8]>) {
        edgeless_function::init_logger();
        log::info!("db_writer: initializing, creating table if not exists");

        // Create the images table if it doesn't exist
        call_wrapper(
            "CREATE TABLE IF NOT EXISTS image_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                source_image_b64 TEXT,
                prompt TEXT,
                generated_image_b64 TEXT,
                timestep INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )"
        );
        log::info!("db_writer: init complete");
    }

    fn handle_cast(_src: InstanceId, message: &[u8]) {
        let msg_str = core::str::from_utf8(message).unwrap_or("");

        // Handle SAVE: prefix for save requests
        if let Some(json_str) = msg_str.strip_prefix("SAVE:") {
            log::info!("db_writer: received save request");
            if let Ok(save_req) = serde_json::from_str::<SaveRequest>(json_str) {
                let sql = format!(
                    "INSERT INTO image_history (session_id, source_image_b64, prompt, generated_image_b64, timestep) VALUES ('{}', '{}', '{}', '{}', {})",
                    escape_sql(&save_req.session_id),
                    escape_sql(&save_req.source_image_b64),
                    escape_sql(&save_req.prompt),
                    escape_sql(&save_req.generated_image_b64),
                    save_req.timestep
                );
                if let Some(result) = call_wrapper(&sql) {
                    log::info!("db_writer: saved successfully, result: {}", result);
                }
            } else {
                log::error!("db_writer: failed to parse save request: {}", json_str);
            }
        } else {
            log::info!("db_writer: received cast: {}", msg_str);
        }
    }

    fn handle_call(_src: InstanceId, _message: &[u8]) -> CallRet {
        log::warn!("db_writer: handle_call not supported");
        CallRet::NoReply
    }

    fn handle_stop() {
        log::info!("db_writer: stopping");
    }
}

// Simple SQL injection prevention - escape single quotes
fn escape_sql(s: &str) -> String {
    s.replace('\'', "''")
}

edgeless_function::export!(DbWriter);