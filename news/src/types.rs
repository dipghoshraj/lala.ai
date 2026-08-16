#[derive(Debug, Clone)]
pub enum ArticleIngestStatus {
    Ingested,
    Skipped(&'static str),
    Failed(String),
}
