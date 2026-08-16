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
