pub mod discovery;
pub mod ingest;
pub mod parser;
pub mod types;

pub use ingest::{ingest_dir_from_env_or_default, ingest_directory, ingest_file};
pub use parser::{DocumentFormat, format_from_path, parse_document, parse_pdf, parse_text};
pub use types::IngestSummary;
pub use types::{DocumentInput, DocumentParseResult, FileIngestStatus, ParsedDocument};
