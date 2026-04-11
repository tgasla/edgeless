// db_reader_with_cache - reads image history from SQLx database with Redis cache-aside pattern

use edgeless_function::*;
use edgeless_function::owned_data::OwnedByteBuff;

const CACHE_KEY: &str = "image_history";
const CACHE_TTL_SECONDS: u64 = 120;

#[derive(serde::Serialize, serde::Deserialize)]
struct HistoryEntry {
    id: Option<i64>,
    session_id: String,
    source_image_b64: String,
    prompt: String,
    generated_image_b64: String,
    creativity: u32,
    created_at: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CachedData {
    entries: Vec<HistoryEntry>,
    cached_at: u64,
}

fn call_database(msg: &str) -> Option<String> {
    match call("database", msg.as_bytes()) {
        CallRet::Reply(data) => {
            let reply = std::str::from_utf8(&data).map(|s| s.to_string()).unwrap_or_default();
            Some(reply)
        }
        CallRet::NoReply => {
            log::warn!("db_reader_with_cache: received empty reply from database");
            None
        }
        CallRet::Err => {
            log::error!("db_reader_with_cache: error calling database");
            None
        }
    }
}

fn call_redis_get(key: &str) -> Option<String> {
    // Try to GET from Redis, catch any panics from the Redis provider (e.g., TypeError on nil)
    let result = std::panic::catch_unwind(|| {
        call("redis", key.as_bytes())
    });

    match result {
        Ok(CallRet::Reply(data)) => {
            let reply = std::str::from_utf8(&data).map(|s| s.to_string()).unwrap_or_default();
            if reply.is_empty() {
                log::info!("db_reader_with_cache: redis returned empty value for key {}", key);
                None
            } else {
                log::info!("db_reader_with_cache: redis cache hit for key {}", key);
                Some(reply)
            }
        }
        Ok(CallRet::NoReply) => {
            log::info!("db_reader_with_cache: redis cache miss (no reply) for key {}", key);
            None
        }
        Ok(CallRet::Err) => {
            log::warn!("db_reader_with_cache: redis get returned error for key {}, falling back to DB", key);
            None
        }
        Err(_) => {
            log::warn!("db_reader_with_cache: redis call panicked for key {}, falling back to DB", key);
            None
        }
    }
}

fn call_redis_set(value: &str) {
    cast("redis", value.as_bytes());
    log::info!("db_reader_with_cache: cache set via cast");
}

fn query_db_and_build_json() -> String {
    // Fetch metadata + images for first 3 entries (most recent), metadata only for rest
    // This keeps the response small enough while still showing images in history
    let query_meta = "SELECT id, session_id, prompt, creativity, created_at FROM image_history ORDER BY id DESC LIMIT 20";
    let query_images = "SELECT id, session_id, source_image_b64, generated_image_b64 FROM image_history ORDER BY id DESC LIMIT 3";

    log::info!("db_reader_with_cache: querying database...");
    let result = call_database(query_meta);
    log::info!("db_reader_with_cache: call_database returned: {:?}", result.is_some());

    if let Some(result) = result {
        log::info!("db_reader_with_cache: got history data from database, result len: {}", result.len());

        let mut entries = Vec::new();
        for line in result.lines().skip(1) {
            let parts: Vec<&str> = line.split('|').collect();
            if parts.len() >= 5 {
                let entry = HistoryEntry {
                    id: parts[0].trim().parse().ok(),
                    session_id: parts[1].trim().to_string(),
                    source_image_b64: String::new(),
                    prompt: parts[2].trim().to_string(),
                    generated_image_b64: String::new(),
                    creativity: parts[3].trim().parse().unwrap_or(0),
                    created_at: parts[4].trim().to_string(),
                };
                entries.push(entry);
            }
        }

        // Now fetch images for the 3 most recent entries
        if let Some(img_result) = call_database(query_images) {
            log::info!("db_reader_with_cache: got image data, result len: {}", img_result.len());
            let mut img_count = 0;
            for line in img_result.lines().skip(1) {
                let parts: Vec<&str> = line.split('|').collect();
                if parts.len() >= 4 && img_count < entries.len() {
                    // Merge image data into the corresponding entry
                    let entry_session = parts[1].trim().to_string();
                    if entries[img_count].session_id == entry_session {
                        entries[img_count].source_image_b64 = parts[2].trim().to_string();
                        entries[img_count].generated_image_b64 = parts[3].trim().to_string();
                        img_count += 1;
                    }
                }
            }
        }

        serde_json::to_string(&entries).unwrap_or_else(|_| "[]".to_string())
    } else {
        log::warn!("db_reader_with_cache: query returned no result");
        "[]".to_string()
    }
}

fn is_cache_valid(cached_data: &CachedData, now: u64) -> bool {
    now.saturating_sub(cached_data.cached_at) < CACHE_TTL_SECONDS
}

struct DbReaderWithCache;

impl EdgeFunction for DbReaderWithCache {
    fn handle_init(_init_message: Option<&[u8]>, _serialized_state: Option<&[u8]>) {
        edgeless_function::init_logger();
        log::info!("db_reader_with_cache: initialized");
    }

    fn handle_cast(_src: InstanceId, message: &[u8]) {
        let msg_str = core::str::from_utf8(message).unwrap_or("");
        log::info!("db_reader_with_cache: received cast: {}", msg_str);
    }

    fn handle_call(_src: InstanceId, message: &[u8]) -> CallRet {
        let msg_str = core::str::from_utf8(message).unwrap_or("");

        if msg_str.starts_with("GET_HISTORY") {
            log::info!("db_reader_with_cache: received GET_HISTORY request");

            // Step 1: Check Redis cache first
            if let Some(cached) = call_redis_get(CACHE_KEY) {
                if let Ok(cached_data) = serde_json::from_str::<CachedData>(&cached) {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);

                    if is_cache_valid(&cached_data, now) {
                        let json = serde_json::to_string(&cached_data.entries).unwrap_or_else(|_| "[]".to_string());
                        log::info!("db_reader_with_cache: cache hit, returning {} bytes", json.len());
                        return CallRet::Reply(OwnedByteBuff::new_from_slice(json.as_bytes()));
                    } else {
                        log::info!("db_reader_with_cache: cache expired");
                    }
                } else {
                    log::warn!("db_reader_with_cache: failed to parse cached data");
                }
            } else {
                log::info!("db_reader_with_cache: cache miss");
            }

            // Step 2: Cache miss or expired - query database
            let json = query_db_and_build_json();
            log::info!("db_reader_with_cache: DB returned {} bytes of history data", json.len());

            // Step 3: Update Redis cache with result
            let cache_value = CachedData {
                entries: serde_json::from_str(&json).unwrap_or_default(),
                cached_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            };

            if let Ok(cache_json) = serde_json::to_string(&cache_value) {
                call_redis_set(&cache_json);
            }

            return CallRet::Reply(OwnedByteBuff::new_from_slice(json.as_bytes()));
        }

        log::warn!("db_reader_with_cache: unknown handle_call request: {}", msg_str);
        CallRet::NoReply
    }

    fn handle_stop() {
        log::info!("db_reader_with_cache: stopping");
    }
}

edgeless_function::export!(DbReaderWithCache);
