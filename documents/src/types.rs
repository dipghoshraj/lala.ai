#[derive(Debug, Clone)]
pub struct ParsedDocument {
    pub title: String,
    pub source: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct DocumentParseResult {
    pub title: String,
    pub source: String,
    pub content: String,
    pub bytes_read: usize,
}

#[derive(Debug, Clone)]
pub struct DocumentInput {
    pub title: String,
    pub source: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub enum FileIngestStatus {
    New(usize),
    Updated(usize),
    Skipped(String),
    Failed(String),
}

#[derive(Debug, Clone, Default)]
pub struct IngestSummary {
    pub ingested: usize,
    pub updated: usize,
    pub skipped: usize,
    pub failed: usize,
    pub chunks: usize,
}
