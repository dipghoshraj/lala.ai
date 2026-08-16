pub mod discovery;
pub mod ingest;
pub mod parser;
pub mod types;

pub use ingest::{ingest_directory, ingest_file, IngestSummary};
pub use parser::{parse_document, parse_text};
pub use types::{DocumentInput, DocumentParseResult, ParsedDocument};
