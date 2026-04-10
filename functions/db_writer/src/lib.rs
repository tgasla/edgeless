// db_writer - receives image generation data and saves to SQLx database

use edgeless_function::*;
use edgeless_function::owned_data::OwnedByteBuff;

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct SaveRequest {
    session_id: String,
    source_image_b64: String,
    prompt: String,
    generated_image_b64: String,
    creativity: u32,
}

fn call_wrapper(msg: &str) -> Option<OwnedByteBuff> {
    match call("database", msg.as_bytes()) {
        CallRet::Reply(data) => {
            log::info!("db_writer got reply from database");
            Some(data)
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
        log::info!("db_writer: initializing");
        // Note: table creation deferred to handle_cast to ensure database resource is ready
    }

    fn handle_cast(_src: InstanceId, message: &[u8]) {
        // Create table on first cast (database resource should be ready by then)
        static INIT_DONE: std::sync::Once = std::sync::Once::new();
        INIT_DONE.call_once(|| {
            log::info!("db_writer: creating table if not exists");
            call_wrapper(
                "CREATE TABLE IF NOT EXISTS image_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    source_image_b64 TEXT,
                    prompt TEXT,
                    generated_image_b64 TEXT,
                    creativity INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )"
            );
        });

        let msg_str = core::str::from_utf8(message).unwrap_or("");

        // Handle SAVE: prefix for save requests
        if let Some(json_str) = msg_str.strip_prefix("SAVE:") {
            log::info!("db_writer: received save request");
            if let Ok(save_req) = serde_json::from_str::<SaveRequest>(json_str) {
                let sql = format!(
                    "INSERT INTO image_history (session_id, source_image_b64, prompt, generated_image_b64, creativity) VALUES ('{}', '{}', '{}', '{}', {})",
                    escape_sql(&save_req.session_id),
                    escape_sql(&save_req.source_image_b64),
                    escape_sql(&save_req.prompt),
                    escape_sql(&save_req.generated_image_b64),
                    save_req.creativity
                );
                if let Some(_result) = call_wrapper(&sql) {
                    log::info!("db_writer: saved successfully");
                    // Update Redis cache with fresh history data
                    if let Some(history_data) = call_wrapper("SELECT id, session_id, source_image_b64, prompt, generated_image_b64, creativity, created_at FROM image_history ORDER BY id DESC LIMIT 20") {
                        if let Ok(history_json) = serde_json::from_slice::<Vec<serde_json::Value>>(&history_data) {
                            if let Ok(json_str) = serde_json::to_string(&history_json) {
                                cast("redis", json_str.as_bytes());
                                log::info!("db_writer: updated redis cache with latest history");
                            }
                        }
                    }
                }
            } else {
                log::error!("db_writer: failed to parse save request: {}", json_str);
            }
        } else {
            log::info!("db_writer: received cast: {}", msg_str);
        }
    }

    fn handle_call(_src: InstanceId, message: &[u8]) -> CallRet {
        let msg_str = core::str::from_utf8(message).unwrap_or("");

        // Handle GET_HISTORY - query database and return results
        if msg_str.starts_with("GET_HISTORY") {
            log::info!("db_writer: received GET_HISTORY request via call");

            // Ensure table exists
            static INIT_DONE: std::sync::Once = std::sync::Once::new();
            INIT_DONE.call_once(|| {
                call_wrapper(
                    "CREATE TABLE IF NOT EXISTS image_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        source_image_b64 TEXT,
                        prompt TEXT,
                        generated_image_b64 TEXT,
                        creativity INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )"
                );
            });

            // Query history - limit to 20 most recent
            let query = "SELECT id, session_id, source_image_b64, prompt, generated_image_b64, creativity, created_at FROM image_history ORDER BY id DESC LIMIT 20";

            if let Some(result) = call_wrapper(query) {
                log::info!("db_writer: history query returned data");
                return CallRet::Reply(result);
            } else {
                log::warn!("db_writer: history query returned no result");
                return CallRet::Reply(OwnedByteBuff::new_from_slice(b"[]"));
            }
        }

        log::warn!("db_writer: handle_call not supported for: {}", msg_str);
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