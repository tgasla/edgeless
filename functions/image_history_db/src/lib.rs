// SPDX-FileCopyrightText: © 2024 Edgeless contributors
// SPDX-License-Identifier: MIT

use edgeless_function::*;
use edgeless_function::owned_data::OwnedByteBuff;

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct HistoryEntry {
    id: Option<i64>,
    session_id: String,
    source_image_b64: String,
    prompt: String,
    generated_image_b64: String,
    timestep: u32,
    created_at: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SaveRequest {
    session_id: String,
    source_image_b64: String,
    prompt: String,
    generated_image_b64: String,
    timestep: u32,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct QueryRequest {
    session_id: Option<String>,
    limit: Option<u32>,
    offset: Option<u32>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct QueryResponse {
    entries: Vec<HistoryEntry>,
    total: u32,
}

struct ImageHistoryDB;

fn db_call(sql: &str) -> Option<String> {
    match call("database", sql.as_bytes()) {
        CallRet::Reply(msg) => {
            let reply = std::str::from_utf8(&msg).unwrap_or("not UTF8");
            Some(reply.to_string())
        }
        CallRet::NoReply => {
            log::warn!("Received empty reply from database");
            None
        }
        CallRet::Err => {
            log::error!("Error when calling database: {}", sql);
            None
        }
    }
}

fn ensure_table_exists() {
    db_call(
        "CREATE TABLE IF NOT EXISTS image_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            source_image_b64 TEXT NOT NULL,
            prompt TEXT NOT NULL,
            generated_image_b64 TEXT NOT NULL,
            timestep INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )"
    );
}

fn save_entry(req: &SaveRequest) -> Option<i64> {
    // Insert and get the last inserted id
    let sql = format!(
        "INSERT INTO image_history (session_id, source_image_b64, prompt, generated_image_b64, timestep) VALUES ('{}', '{}', '{}', '{}', {})",
        req.session_id,
        req.source_image_b64,
        req.prompt.replace("'", "''"),
        req.generated_image_b64,
        req.timestep
    );

    if db_call(&sql).is_some() {
        // Get the last inserted id
        if let Some(result) = db_call("SELECT last_insert_rowid()") {
            if let Ok(id) = result.parse::<i64>() {
                return Some(id);
            }
        }
    }
    None
}

fn query_entries(req: &QueryRequest) -> QueryResponse {
    let limit = req.limit.unwrap_or(50);
    let offset = req.offset.unwrap_or(0);

    let where_clause = if let Some(ref session_id) = req.session_id {
        format!("WHERE session_id = '{}'", session_id)
    } else {
        String::new()
    };

    // Get total count
    let count_sql = format!("SELECT COUNT(*) FROM image_history {}", where_clause);
    let total: u32 = db_call(&count_sql)
        .and_then(|s| s.lines().last().map(|l| l.trim().to_string()))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Get entries with pagination
    let select_sql = format!(
        "SELECT id, session_id, source_image_b64, prompt, generated_image_b64, timestep, created_at
         FROM image_history {} ORDER BY id DESC LIMIT {} OFFSET {}",
        where_clause, limit, offset
    );

    let mut entries = Vec::new();

    if let Some(result) = db_call(&select_sql) {
        for line in result.lines().skip(1) { // Skip header
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
    }

    QueryResponse { entries, total }
}

impl EdgeFunction for ImageHistoryDB {
    fn handle_init(_payload: Option<&[u8]>, _serialized_state: Option<&[u8]>) {
        edgeless_function::init_logger();
        log::info!("Image History DB function init");
        ensure_table_exists();
    }

    fn handle_cast(_src: InstanceId, message: &[u8]) {
        let msg_str = core::str::from_utf8(message).unwrap_or_default();

        if msg_str.starts_with("SAVE:") {
            // Parse save request
            let json_str = msg_str.strip_prefix("SAVE:").unwrap_or("");
            if let Ok(req) = serde_json::from_str::<SaveRequest>(json_str) {
                if let Some(id) = save_entry(&req) {
                    log::info!("Saved history entry with id: {}", id);
                }
            }
        }
    }

    fn handle_call(_src: InstanceId, message: &[u8]) -> CallRet {
        let msg_str = core::str::from_utf8(message).unwrap_or_default();

        if msg_str.starts_with("QUERY:") {
            let json_str = msg_str.strip_prefix("QUERY:").unwrap_or("");
            if let Ok(req) = serde_json::from_str::<QueryRequest>(json_str) {
                let response = query_entries(&req);
                if let Ok(json) = serde_json::to_string(&response) {
                    return CallRet::Reply(OwnedByteBuff::new_from_slice(json.as_bytes()));
                }
            }
        }

        CallRet::NoReply
    }

    fn handle_stop() {
        log::info!("Image History DB function stopped");
    }
}

edgeless_function::export!(ImageHistoryDB);