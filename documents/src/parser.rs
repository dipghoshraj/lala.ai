use std::fs;
use std::path::Path;

use crate::types::{DocumentParseResult, ParsedDocument};

pub fn parse_text(path: &str, content: &str) -> ParsedDocument {
    let title = Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string());

    ParsedDocument {
        title,
        source: path.to_string(),
        content: content.trim().to_string(),
    }
}

pub fn parse_document(path: &str) -> anyhow::Result<DocumentParseResult> {
    let content = fs::read_to_string(path)?;
    let parsed = parse_text(path, &content);

    Ok(DocumentParseResult {
        title: parsed.title,
        source: parsed.source,
        content: parsed.content,
        bytes_read: content.len(),
    })
}
