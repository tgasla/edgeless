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
    match call("redis", key.as_bytes()) {
        CallRet::Reply(data) => {
            let reply = std::str::from_utf8(&data).map(|s| s.to_string()).unwrap_or_default();
            Some(reply)
        }
        CallRet::NoReply => {
            log::warn!("db_reader_with_cache: received empty reply from redis");
            None
        }
        CallRet::Err => {
            log::error!("db_reader_with_cache: error calling redis");
            None
        }
    }
}

fn call_redis_set(key: &str, value: &str) -> bool {
    let payload = format!("SETEX {} {} {}", key, CACHE_TTL_SECONDS, value);
    match call("redis", payload.as_bytes()) {
        CallRet::Reply(_) => {
            log::info!("db_reader_with_cache: cache updated successfully");
            true
        }
        CallRet::NoReply => {
            log::warn!("db_reader_with_cache: redis set returned no reply");
            false
        }
        CallRet::Err => {
            log::error!("db_reader_with_cache: error setting redis cache");
            false
        }
    }
}

fn query_db_and_build_json() -> String {
    let query = "SELECT id, session_id, source_image_b64, prompt, generated_image_b64, creativity, created_at FROM image_history ORDER BY id DESC LIMIT 20";

    if let Some(result) = call_database(query) {
        log::info!("db_reader_with_cache: got history data from database");

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
                    creativity: parts[5].trim().parse().unwrap_or(0),
                    created_at: parts[6].trim().to_string(),
                });
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
                        log::info!("db_reader_with_cache: cache hit, returning cached data");
                        let json = serde_json::to_string(&cached_data.entries).unwrap_or_else(|_| "[]".to_string());
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

            // Step 3: Update Redis cache with result
            let cache_value = CachedData {
                entries: serde_json::from_str(&json).unwrap_or_default(),
                cached_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            };

            if let Ok(cache_json) = serde_json::to_string(&cache_value) {
                call_redis_set(CACHE_KEY, &cache_json);
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
