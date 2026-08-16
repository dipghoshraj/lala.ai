pub mod discovery;
pub mod ingest;
pub mod parser;
pub mod types;

pub use ingest::{ingest_directory, ingest_dir_from_env_or_default, ingest_file};
pub use types::IngestSummary;
pub use parser::{parse_document, parse_text};
pub use types::{DocumentInput, DocumentParseResult, FileIngestStatus, ParsedDocument};
