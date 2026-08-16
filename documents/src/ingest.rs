use rag::RagStore;

use crate::parser::parse_document;

#[derive(Debug, Clone)]
pub struct IngestSummary {
    pub ingested: usize,
    pub updated: usize,
    pub skipped: usize,
    pub failed: usize,
    pub chunks: usize,
}

pub fn ingest_file(store: &RagStore, path: &str) -> anyhow::Result<usize> {
    let parsed = parse_document(path)?;
    let existed = rag::model::document::Document::exist(path).unwrap_or(false);

    let count = store.ingest(&parsed.title, &parsed.source, &parsed.content)?;
    Ok(if existed { count } else { count })
}

pub fn ingest_directory(store: &RagStore, dir: &str) -> anyhow::Result<IngestSummary> {
    let files = crate::discovery::scan_directory(dir)?;
    let mut summary = IngestSummary {
        ingested: 0,
        updated: 0,
        skipped: 0,
        failed: 0,
        chunks: 0,
    };

    for file in files {
        match ingest_file(store, &file) {
            Ok(count) => {
                summary.ingested += 1;
                summary.chunks += count;
            }
            Err(err) => {
                let reason = err.to_string();
                if reason.contains("file is empty") {
                    summary.skipped += 1;
                } else {
                    summary.failed += 1;
                }
            }
        }
    }

    Ok(summary)
}
