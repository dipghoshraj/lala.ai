use std::path::Path;

use crate::agent::model::ApiClient;
use documents::{self, FileIngestStatus};
use news;
use rag::RagStore;

use super::display;

/// `/ingest [path]` — batch-ingest all files in the given directory (or the default ingest directory).
pub fn ingest_all(store: &RagStore, _client: &ApiClient, args: &str) {
    let dir = if args.is_empty() {
        documents::ingest_dir_from_env_or_default()
    } else {
        args.to_string()
    };

    println!();
    display::info(&format!("Ingesting files from: {dir}/"));
    println!();

    match documents::ingest_directory(store, &dir, |current, total, filename, status| {
        display::progress(current, total, filename);
        match status {
            FileIngestStatus::New(count) => {
                display::success(&format!("{filename} → {count} chunks"));
            }
            FileIngestStatus::Updated(count) => {
                display::success(&format!("{filename} → {count} chunks (updated)"));
            }
            FileIngestStatus::Skipped(reason) => {
                display::warn(&format!("{filename}: {reason}"));
            }
            FileIngestStatus::Failed(reason) => {
                display::error(&format!("{filename}: {reason}"));
            }
        }
    }) {
        Ok(summary) => {
            // ── Summary ───────────────────────────────────────────────────
            println!();
            let sep = "─".repeat(display::SECTION_WIDTH);
            println!("{}{}{}", display::DIM, sep, display::RESET);
            println!(
                "  Ingested: {}{}{}  Updated: {}{}{}  Skipped: {}{}{}  Failed: {}{}{}  Chunks: {}",
                display::BOLD_GREEN,
                summary.ingested,
                display::RESET,
                display::BOLD_GREEN,
                summary.updated,
                display::RESET,
                display::YELLOW,
                summary.skipped,
                display::RESET,
                if summary.failed > 0 { display::BOLD_RED } else { display::DIM },
                summary.failed,
                display::RESET,
                summary.chunks,
            );
            println!("{}{}{}", display::DIM, sep, display::RESET);
            println!();
        }
        Err(e) => {
            display::error(&format!("Failed to ingest directory: {e}"));
            println!();
        }
    }
}

/// `/ingest-file <path>` — ingest a single file by explicit path.
pub fn ingest_file(store: &RagStore, _client: &ApiClient, path: &str) {
    if path.is_empty() {
        println!("Usage: /ingest-file <path>\n");
        return;
    }

    let filename = Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string());

    match documents::ingest_file(store, path) {
        FileIngestStatus::New(count) => display::success(&format!("{filename} → {count} chunks")),
        FileIngestStatus::Updated(count) => {
            display::success(&format!("{filename} → {count} chunks (updated)"))
        }
        FileIngestStatus::Skipped(reason) => display::warn(&format!("{filename}: {reason}")),
        FileIngestStatus::Failed(reason) => display::error(&format!("{filename}: {reason}")),
    }
    println!();
}

/// `/ingest-news <rss_url>` — fetch RSS feed and ingest all articles into RAG.
pub fn ingest_news(store: &RagStore, url: &str) {
    if url.is_empty() {
        println!("Usage: /ingest-news <rss_url>\n");
        println!("Example: /ingest-news https://feeds.bbci.co.uk/news/rss.xml");
        println!();
        return;
    }

    display::info(&format!("Ingesting news from: {}", url));
    println!();

    match news::ingest_news_feed_with_progress(store, url, 1000, |_title, status| {
        // Per-article progress can be rendered here when the display layer supports it.
        let _ = status;
    }) {
        Ok((ingested, skipped, failed)) => {
            println!();
            let sep = "─".repeat(display::SECTION_WIDTH);
            println!("{}{}{}", display::DIM, sep, display::RESET);
            println!(
                "  Ingested: {}{}{}  Skipped: {}{}{}  Failed: {}{}{}",
                display::BOLD_GREEN,
                ingested,
                display::RESET,
                display::YELLOW,
                skipped,
                display::RESET,
                if failed > 0 { display::BOLD_RED } else { display::DIM },
                failed,
                display::RESET,
            );
            println!("{}{}{}", display::DIM, sep, display::RESET);
            println!();
        }
        Err(e) => {
            display::error(&format!("News ingestion failed: {e}"));
            println!();
        }
    }
}
