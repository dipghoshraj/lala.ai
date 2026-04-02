use std::fs;
use std::path::Path;

use crate::agent::memory::LlmMemoryExtractor;
use crate::agent::model::ApiClient;
use rag::RagStore;

use super::display;

/// Default ingest directory relative to the working directory.
const DEFAULT_INGEST_DIR: &str = "./ingest";

/// Resolve the ingest directory path from `LALA_INGEST_DIR` env or the default.
fn ingest_dir() -> String {
    std::env::var("LALA_INGEST_DIR").unwrap_or_else(|_| DEFAULT_INGEST_DIR.to_string())
}

/// Scan the ingest directory and return sorted list of file paths.
fn scan_ingest_dir(dir: &str) -> anyhow::Result<Vec<String>> {
    let path = Path::new(dir);
    if !path.exists() {
        fs::create_dir_all(path)?;
        display::info(&format!("Created ingest directory: {dir}"));
        return Ok(Vec::new());
    }
    if !path.is_dir() {
        anyhow::bail!("{dir} exists but is not a directory");
    }

    let mut files = Vec::new();
    collect_files(path, &mut files)?;
    files.sort();
    Ok(files)
}

/// Recursively collect all files under `dir`.
fn collect_files(dir: &Path, out: &mut Vec<String>) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        if ft.is_dir() {
            collect_files(&entry.path(), out)?;
        } else if ft.is_file() {
            if let Some(p) = entry.path().to_str() {
                out.push(p.to_string());
            }
        }
    }
    Ok(())
}

/// `/ingest [path]` — batch-ingest all files in the given directory (or the default ingest directory).
pub fn ingest_all(store: &RagStore, client: &ApiClient, args: &str) {
    let dir = if args.is_empty() { ingest_dir() } else { args.to_string() };

    let files = match scan_ingest_dir(&dir) {
        Ok(f) => f,
        Err(e) => {
            display::error(&format!("Failed to scan ingest directory: {e}"));
            return;
        }
    };

    if files.is_empty() {
        display::warn(&format!("No files found in {dir}/"));
        display::info("Place files in the ingest directory and run /ingest again.");
        return;
    }

    let total = files.len();
    println!();
    display::info(&format!("Found {total} file(s) in {dir}/"));
    println!();

    let mut ingested = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;
    let mut total_chunks = 0usize;

    for (i, file_path) in files.iter().enumerate() {
        let filename = Path::new(file_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| file_path.clone());

        display::progress(i + 1, total, &filename);

        match ingest_single_file(store, client, file_path) {
            IngestResult::Ok(count) => {
                display::success(&format!("{filename} → {count} chunks"));
                ingested += 1;
                total_chunks += count;
            }
            IngestResult::Skipped(reason) => {
                display::warn(&format!("{filename}: {reason}"));
                skipped += 1;
            }
            IngestResult::Err(e) => {
                display::error(&format!("{filename}: {e}"));
                failed += 1;
            }
        }
    }

    // ── Summary ───────────────────────────────────────────────────────────
    println!();
    let sep = "─".repeat(display::SECTION_WIDTH);
    println!("{}{}{}", display::DIM, sep, display::RESET);
    println!(
        "  Ingested: {}{}{}  Skipped: {}{}{}  Failed: {}{}{}  Chunks: {}",
        display::BOLD_GREEN,
        ingested,
        display::RESET,
        display::YELLOW,
        skipped,
        display::RESET,
        if failed > 0 { display::BOLD_RED } else { display::DIM },
        failed,
        display::RESET,
        total_chunks,
    );
    println!("{}{}{}", display::DIM, sep, display::RESET);
    println!();
}

/// `/ingest-file <path>` — ingest a single file by explicit path.
pub fn ingest_file(store: &RagStore, client: &ApiClient, path: &str) {
    if path.is_empty() {
        println!("Usage: /ingest-file <path>\n");
        return;
    }

    let filename = Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string());

    match ingest_single_file(store, client, path) {
        IngestResult::Ok(count) => display::success(&format!("{filename} → {count} chunks")),
        IngestResult::Skipped(reason) => display::warn(&format!("{filename}: {reason}")),
        IngestResult::Err(e) => display::error(&format!("{filename}: {e}")),
    }
    println!();
}

// ── Internal ──────────────────────────────────────────────────────────────────

enum IngestResult {
    Ok(usize),
    Skipped(String),
    Err(String),
}

fn ingest_single_file(store: &RagStore, client: &ApiClient, path: &str) -> IngestResult {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return IngestResult::Err(format!("cannot read file: {e}")),
    };

    if content.is_empty() {
        return IngestResult::Skipped("file is empty".to_string());
    }

    let title = Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string());

    let extractor = LlmMemoryExtractor::new(client);

    match store.ingest(&title, path, &content, Some(&extractor)) {
        Ok(count) => IngestResult::Ok(count),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("Already ingested") {
                IngestResult::Skipped(msg)
            } else {
                IngestResult::Err(msg)
            }
        }
    }
}

