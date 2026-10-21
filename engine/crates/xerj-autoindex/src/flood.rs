//! Fail fast when the node's data volume is already at the flood-stage
//! watermark (#2077).
//!
//! The watermark lives on the **node**: `limits.disk_flood_stage_percent`
//! (default 95) measured against the filesystem that holds the node's data
//! directory. Autoindex and the node are not necessarily the same machine, so
//! this probe reads `GET /_cat/allocation?format=json` — the endpoint that
//! already reports that volume's used percent and free bytes — plus the
//! effective threshold and the latched write block the node adds to that
//! JSON. It does not `stat` the client's disk.
//!
//! A node that does not send `disk.flood_stage` (an older XERJ, Elasticsearch,
//! or the HTTP doubles in this crate's tests) is not guessed at. Those runs
//! behave as they did before: the write itself still reports the block if one
//! engages later.

use serde_json::Value;

use crate::esclient::Es;

const ALLOCATION_PATH: &str = "/_cat/allocation?format=json";

/// One row of `GET /_cat/allocation?format=json`.
///
/// The budget gate reads free space and the flood threshold from this same
/// response. It does not issue a second probe, and it does not fill in a
/// threshold the node left out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Volume {
    pub total: Option<u64>,
    pub avail: Option<u64>,
    pub used_percent: Option<u64>,
    pub flood_stage: Option<u8>,
    pub threshold_enabled: bool,
    pub flood_blocked: bool,
}

/// The one allocation read: the parsed volume, and the refusal when the
/// node says writes will fail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Probe {
    pub volume: Option<Volume>,
    pub refusal: Option<String>,
}

/// `None` fields when the node cannot be asked. A stats blip must not abort
/// a run whose disk is fine, and the write path still refuses if the block
/// is actually engaged.
pub(crate) fn probe(es: &Es) -> Probe {
    let Ok((status, body)) = es.request_json("GET", ALLOCATION_PATH, None) else {
        return Probe {
            volume: None,
            refusal: None,
        };
    };
    if !(200..300).contains(&status) {
        return Probe {
            volume: None,
            refusal: None,
        };
    }
    Probe {
        volume: volume_from_allocation(&body),
        refusal: refusal_from_allocation(&body),
    }
}

/// Parsed disk columns from one allocation body. `None` when the body is
/// not an allocation row (the HTTP doubles answer `{"acknowledged": true}`).
pub(crate) fn volume_from_allocation(body: &Value) -> Option<Volume> {
    let row = body.as_array()?.first()?;
    let flood_stage = row
        .get("disk.flood_stage")
        .and_then(parse_u64)
        .and_then(|n| u8::try_from(n.min(100)).ok());
    Some(Volume {
        total: row.get("disk.total").and_then(parse_bytes),
        avail: row.get("disk.avail").and_then(parse_bytes),
        used_percent: row.get("disk.percent").and_then(parse_u64),
        flood_stage,
        threshold_enabled: row
            .get("disk.threshold_enabled")
            .and_then(parse_bool)
            .unwrap_or(true),
        flood_blocked: row
            .get("disk.flood_blocked")
            .and_then(parse_bool)
            .unwrap_or(false),
    })
}

/// Decide from one `/_cat/allocation?format=json` body.
///
/// `Some` is the error text for a volume already at or over the watermark,
/// or for a write block the node has not released yet (the 1% hysteresis
/// below the threshold still refuses writes). `None` means index.
pub(crate) fn refusal_from_allocation(body: &Value) -> Option<String> {
    let row = body.as_array()?.first()?;
    // Absent threshold: this response is not the XERJ flood-stage report.
    // Do not assume 95. The node's config may be 99 (the #2077 workaround)
    // or 0 (watermark off), and a hard-coded default would refuse both.
    let flood_raw = parse_u64(row.get("disk.flood_stage")?)?;
    let flood = u8::try_from(flood_raw.min(100)).ok()?;
    let enabled = row
        .get("disk.threshold_enabled")
        .and_then(parse_bool)
        .unwrap_or(true);
    let blocked = row
        .get("disk.flood_blocked")
        .and_then(parse_bool)
        .unwrap_or(false);
    let used = row.get("disk.percent").and_then(parse_u64);
    let free = row.get("disk.avail").and_then(parse_bytes);

    let over = enabled && flood > 0 && used.is_some_and(|pct| pct >= u64::from(flood));
    if !blocked && !over {
        return None;
    }
    Some(refusal_message(free, used, flood, blocked && !over))
}

fn refusal_message(free: Option<u64>, used: Option<u64>, flood: u8, hysteresis: bool) -> String {
    let free = match free {
        Some(bytes) => format_bytes(bytes),
        None => "unknown".to_string(),
    };
    let used = match used {
        Some(pct) => format!("{pct}% used"),
        None => "used percent unknown".to_string(),
    };
    let why = if hysteresis {
        "the disk flood-stage write block is still engaged"
    } else {
        "its data volume is already at or over the disk flood-stage watermark"
    };
    format!(
        "the node will refuse every write: {why} (free {free}, {used}, threshold {flood}%). \
         Set limits.disk_flood_stage_percent above the used percent in the node config and \
         restart (0 disables the watermark), or raise it live with \
         PUT /_cluster/settings {{\"persistent\":{{\"cluster.routing.allocation.disk.watermark.flood_stage\":\"<percent>\"}}}}. \
         No files were walked and nothing was written."
    )
}

/// Same scale as the engine's flood-stage log line (`2.0gb`, `512b`).
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    if bytes >= GB {
        format!("{:.1}gb", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}mb", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1}kb", bytes as f64 / KB as f64)
    } else {
        format!("{bytes}b")
    }
}

fn parse_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.trim().trim_end_matches('%').parse().ok(),
        _ => None,
    }
}

fn parse_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        Value::String(s) => match s.trim() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// `/_cat/allocation` emits free space as a raw byte count with a `b` suffix
/// (`"2147483648b"`). Anything else is left unknown rather than guessed.
fn parse_bytes(value: &Value) -> Option<u64> {
    match value {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => {
            let s = s.trim();
            let s = s
                .strip_suffix('b')
                .or_else(|| s.strip_suffix('B'))
                .unwrap_or(s);
            if !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()) {
                s.parse().ok()
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn row(percent: &str, avail: &str, flood: &str, enabled: &str, blocked: &str) -> Value {
        json!([{
            "disk.percent": percent,
            "disk.avail": avail,
            "disk.flood_stage": flood,
            "disk.threshold_enabled": enabled,
            "disk.flood_blocked": blocked,
        }])
    }

    #[test]
    fn at_or_over_the_watermark_refuses_with_free_space_threshold_and_override() {
        let message = refusal_from_allocation(&row("98", "2147483648b", "95", "true", "false"))
            .expect("98% against a 95% watermark must refuse");
        assert!(message.contains("free 2.0gb"), "{message}");
        assert!(message.contains("98% used"), "{message}");
        assert!(message.contains("threshold 95%"), "{message}");
        assert!(
            message.contains("limits.disk_flood_stage_percent"),
            "{message}"
        );
        assert!(message.contains("No files were walked"), "{message}");
        assert!(
            message.contains("at or over the disk flood-stage watermark"),
            "{message}"
        );

        // The watermark is inclusive: used == threshold is already blocked
        // (`disk_used_pct >= flood_pct` in the governor).
        assert!(refusal_from_allocation(&row("95", "1b", "95", "true", "false")).is_some());
    }

    #[test]
    fn a_healthy_volume_and_a_raised_or_disabled_watermark_do_not_refuse() {
        assert!(refusal_from_allocation(&row("40", "1000b", "95", "true", "false")).is_none());
        // One percent under the threshold, block not latched.
        assert!(refusal_from_allocation(&row("94", "1000b", "95", "true", "false")).is_none());
        // The field-report workaround: config raised to 99 while the volume
        // sits at 98. Writes are admitted; autoindex must not refuse.
        assert!(refusal_from_allocation(&row("98", "1000b", "99", "true", "false")).is_none());
        // `limits.disk_flood_stage_percent = 0` turns the watermark off.
        assert!(refusal_from_allocation(&row("98", "1000b", "0", "true", "false")).is_none());
        // `cluster.routing.allocation.disk.threshold_enabled: false`.
        assert!(refusal_from_allocation(&row("98", "1000b", "95", "false", "false")).is_none());
    }

    #[test]
    fn hysteresis_still_refuses_and_a_silent_node_does_not() {
        let message = refusal_from_allocation(&row("94", "4096b", "95", "true", "true"))
            .expect("a latched block under the threshold still refuses writes");
        assert!(
            message.contains("write block is still engaged"),
            "{message}"
        );
        assert!(message.contains("free 4.0kb"), "{message}");
        assert!(message.contains("threshold 95%"), "{message}");

        // Older XERJ / Elasticsearch / the in-crate HTTP doubles: allocation
        // JSON without the threshold is not a reason to stop.
        assert!(refusal_from_allocation(&json!([{
            "disk.percent": "99",
            "disk.avail": "1b"
        }]))
        .is_none());
        assert!(refusal_from_allocation(&json!({"acknowledged": true})).is_none());
        assert!(refusal_from_allocation(&json!([])).is_none());
    }

    #[test]
    fn the_same_row_carries_free_and_total_bytes_for_the_budget() {
        let body = json!([{
            "disk.percent": "40",
            "disk.avail": "1000b",
            "disk.total": "10000b",
            "disk.flood_stage": "95",
            "disk.threshold_enabled": "true",
            "disk.flood_blocked": "false",
        }]);
        let volume = volume_from_allocation(&body).expect("an allocation row");
        assert_eq!(volume.avail, Some(1000));
        assert_eq!(volume.total, Some(10000));
        assert_eq!(volume.used_percent, Some(40));
        assert_eq!(volume.flood_stage, Some(95));
        assert!(volume_from_allocation(&json!({"acknowledged": true})).is_none());
    }

    /// The probe runs before the walk: a flooded node refuses with the
    /// override text and never opens the folder, and a healthy node is not
    /// refused (the walk then reports that the root is a file).
    #[test]
    fn autoindex_refuses_before_walk_only_when_the_node_reports_flood_stage() {
        let flooded = Node::start(true);
        let healthy = Node::start(false);
        let root_dir = tempfile::tempdir().unwrap();
        let root = root_dir.path().join("not-a-folder");
        std::fs::write(&root, "bytes the walk must not have to read\n").unwrap();

        let flooded_state = tempfile::tempdir().unwrap();
        let error = crate::run_index_report(index_cfg(&root, flooded_state.path(), &flooded.url))
            .expect_err("a flooded data volume must fail the run");
        let message = format!("{error:#}");
        assert!(
            message.contains("limits.disk_flood_stage_percent"),
            "{message}"
        );
        assert!(message.contains("free 2.0gb"), "{message}");
        assert!(message.contains("threshold 95%"), "{message}");
        assert!(
            !message.contains("is not a directory"),
            "the walk must not have started: {message}"
        );
        assert!(
            !flooded_state.path().join(".autoindex.lock").exists(),
            "refusing before the walk must not open the state directory"
        );
        assert_eq!(
            flooded.requests(),
            vec![
                ("GET".to_string(), "/".to_string()),
                (
                    "GET".to_string(),
                    "/_cat/allocation?format=json".to_string(),
                ),
            ]
        );

        let healthy_state = tempfile::tempdir().unwrap();
        let error = crate::run_index_report(index_cfg(&root, healthy_state.path(), &healthy.url))
            .expect_err("a file root still cannot be walked");
        let message = format!("{error:#}");
        assert!(
            message.contains("is not a directory"),
            "a healthy volume must reach the walk: {message}"
        );
        assert!(
            !message.contains("limits.disk_flood_stage_percent"),
            "{message}"
        );

        // A dry run writes nothing: a flooded node does not refuse it.
        let dry_state = tempfile::tempdir().unwrap();
        let mut dry = index_cfg(&root, dry_state.path(), &flooded.url);
        dry.dry_run = true;
        if let Err(error) = crate::run_index_report(dry) {
            let message = format!("{error:#}");
            assert!(
                !message.contains("limits.disk_flood_stage_percent"),
                "a dry run must not be refused for disk: {message}"
            );
        }
    }

    fn index_cfg(
        root: &std::path::Path,
        state: &std::path::Path,
        url: &str,
    ) -> crate::cli::IndexCfg {
        let args = vec![
            root.to_string_lossy().to_string(),
            "--url".into(),
            url.into(),
            "--state-dir".into(),
            state.to_string_lossy().to_string(),
            "--prefix".into(),
            "flood2077".into(),
            "--no-graph".into(),
            "--quiet".into(),
        ];
        match crate::cli::parse(args).expect("argv parses") {
            crate::cli::Cmd::Index(cfg) => *cfg,
            other => panic!("expected index, got {other:?}"),
        }
    }

    struct Node {
        url: String,
        requests: std::sync::Arc<std::sync::Mutex<Vec<(String, String)>>>,
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
        join: Option<std::thread::JoinHandle<()>>,
    }

    impl Node {
        fn start(flooded: bool) -> Self {
            use std::io::{BufRead, BufReader, Read, Write};
            use std::net::TcpListener;
            use std::sync::atomic::{AtomicBool, Ordering};
            use std::sync::{Arc, Mutex};
            use std::time::Duration;

            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let requests = Arc::new(Mutex::new(Vec::new()));
            let stop = Arc::new(AtomicBool::new(false));
            let requests_bg = Arc::clone(&requests);
            let stop_bg = Arc::clone(&stop);
            let join = std::thread::spawn(move || {
                while !stop_bg.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            stream.set_nonblocking(false).unwrap();
                            stream
                                .set_read_timeout(Some(Duration::from_secs(5)))
                                .unwrap();
                            let mut reader = BufReader::new(stream.try_clone().unwrap());
                            let mut request_line = String::new();
                            if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
                                continue;
                            }
                            let mut content_length = 0usize;
                            loop {
                                let mut line = String::new();
                                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                                    break;
                                }
                                if line == "\r\n" || line.is_empty() {
                                    break;
                                }
                                if let Some(value) =
                                    line.to_ascii_lowercase().strip_prefix("content-length:")
                                {
                                    content_length = value.trim().parse().unwrap_or(0);
                                }
                            }
                            if content_length > 0 {
                                let mut body = vec![0; content_length];
                                let _ = reader.read_exact(&mut body);
                            }
                            let mut parts = request_line.split_whitespace();
                            let method = parts.next().unwrap_or("").to_string();
                            let path = parts.next().unwrap_or("").to_string();
                            requests_bg
                                .lock()
                                .unwrap()
                                .push((method.clone(), path.clone()));
                            let (status, body) = if method == "GET" && path == "/" {
                                (200, r#"{"ok":true}"#)
                            } else if method == "GET" && path.starts_with("/_cat/allocation") {
                                if flooded {
                                    (
                                        200,
                                        r#"[{"disk.percent":"98","disk.avail":"2147483648b","disk.flood_stage":"95","disk.threshold_enabled":"true","disk.flood_blocked":"false"}]"#,
                                    )
                                } else {
                                    (
                                        200,
                                        r#"[{"disk.percent":"10","disk.avail":"1073741824b","disk.flood_stage":"95","disk.threshold_enabled":"true","disk.flood_blocked":"false"}]"#,
                                    )
                                }
                            } else {
                                (404, r#"{"error":"absent"}"#)
                            };
                            let reason = if status == 200 { "OK" } else { "Not Found" };
                            let _ = write!(
                                stream,
                                "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                                body.len()
                            );
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("accept failed: {error}"),
                    }
                }
            });
            Self {
                url,
                requests,
                stop,
                join: Some(join),
            }
        }

        fn requests(&self) -> Vec<(String, String)> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl Drop for Node {
        fn drop(&mut self) {
            self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
            if let Some(join) = self.join.take() {
                let _ = join.join();
            }
        }
    }
}
