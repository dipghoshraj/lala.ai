use std::fs;
use std::path::Path;

use crate::types::{DocumentParseResult, ParsedDocument};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentFormat {
    Pdf,
    Text,
}

pub fn format_from_path(path: &str) -> DocumentFormat {
    match Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("pdf") => DocumentFormat::Pdf,
        _ => DocumentFormat::Text,
    }
}

pub fn parse_text(path: &str, content: &str) -> ParsedDocument {
    let title = Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string());

    ParsedDocument {
        title,
        source: path.to_string(),
        content: content.to_string(),
    }
}

pub fn parse_pdf(path: &str) -> anyhow::Result<DocumentParseResult> {
    let markdown = anydoc::to_markdown(path).map_err(|e: anydoc::ConvertError| match e {
        anydoc::ConvertError::Malformed { .. } => anyhow::anyhow!("no extractable text: {e}"),
        _ => anyhow::Error::from(e),
    })?;
    if markdown.trim().is_empty() {
        anyhow::bail!("no extractable text");
    }

    let parsed = parse_text(path, &markdown);
    let bytes_read = fs::metadata(path)?.len() as usize;

    Ok(DocumentParseResult {
        title: parsed.title,
        source: parsed.source,
        content: parsed.content,
        bytes_read,
    })
}

pub fn parse_document(path: &str) -> anyhow::Result<DocumentParseResult> {
    match format_from_path(path) {
        DocumentFormat::Pdf => parse_pdf(path),
        DocumentFormat::Text => {
            let content = fs::read_to_string(path)?;
            if content.trim().is_empty() {
                anyhow::bail!("file is empty");
            }
            let parsed = parse_text(path, &content);

            Ok(DocumentParseResult {
                title: parsed.title,
                source: parsed.source,
                content: parsed.content,
                bytes_read: content.len(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_from_path_detects_pdf_and_text_formats() {
        assert_eq!(format_from_path("guide.pdf"), DocumentFormat::Pdf);
        assert_eq!(format_from_path("guide.PDF"), DocumentFormat::Pdf);
        assert_eq!(format_from_path("notes.md"), DocumentFormat::Text);
        assert_eq!(format_from_path("notes.txt"), DocumentFormat::Text);
        assert_eq!(format_from_path("README"), DocumentFormat::Text);
    }

    #[test]
    fn parse_pdf_returns_markdown_for_fixture() {
        let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/sample.pdf");
        let parsed = parse_pdf(fixture).expect("fixture PDF should parse");

        assert!(parsed.bytes_read > 0);
        assert!(!parsed.content.trim().is_empty());
        assert!(parsed.content.contains("Sample PDF"));
        assert!(
            parsed
                .content
                .contains("This paragraph came from a PDF fixture.")
        );
    }
}
