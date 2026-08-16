use rag::RagStore;

use crate::parser::parse_document;
use crate::types::{FileIngestStatus, IngestSummary};

const DEFAULT_INGEST_DIR: &str = "./ingest";

pub fn ingest_file(store: &RagStore, path: &str) -> FileIngestStatus {
    let parsed = match parse_document(path) {
        Ok(p) => p,
        Err(e) => {
            let reason = e.to_string();
            if reason.contains("file is empty") {
                return FileIngestStatus::Skipped("file is empty".to_string());
            }
            return FileIngestStatus::Failed(format!("cannot read file: {e}"));
        }
    };

    let existed = rag::model::document::Document::exist(path).unwrap_or(false);

    match store.ingest(&parsed.title, &parsed.source, &parsed.content) {
        Ok(count) => {
            if existed {
                FileIngestStatus::Updated(count)
            } else {
                FileIngestStatus::New(count)
            }
        }
        Err(e) => FileIngestStatus::Failed(e.to_string()),
    }
}

pub fn ingest_directory<F>(store: &RagStore, dir: &str, mut on_file: F) -> anyhow::Result<IngestSummary>
where
    F: FnMut(usize, usize, &str, &FileIngestStatus),
{
    let files = crate::discovery::scan_directory(dir)?;
    let total = files.len();
    let mut summary = IngestSummary::default();

    for (i, file) in files.iter().enumerate() {
        let filename = file_name(file);
        let existed = rag::model::document::Document::exist(file).unwrap_or(false);
        let status = ingest_file(store, file);
        on_file(i + 1, total, &filename, &status);

        match &status {
            FileIngestStatus::New(count) => {
                if existed {
                    summary.updated += 1;
                } else {
                    summary.ingested += 1;
                }
                summary.chunks += count;
            }
            FileIngestStatus::Updated(count) => {
                summary.updated += 1;
                summary.chunks += count;
            }
            FileIngestStatus::Skipped(_) => summary.skipped += 1,
            FileIngestStatus::Failed(_) => summary.failed += 1,
        }
    }

    Ok(summary)
}

pub fn ingest_dir_from_env_or_default() -> String {
    std::env::var("LALA_INGEST_DIR").unwrap_or_else(|_| DEFAULT_INGEST_DIR.to_string())
}

fn file_name(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}
