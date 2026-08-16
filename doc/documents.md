# documents Crate

> **Location:** `lala.ai/documents/`
> **Role:** Document ingestion library — file discovery, parsing, normalization, and orchestration of storage via `rag::RagStore`.

## Overview

The `documents` crate owns everything between the filesystem and the RAG store for file-based ingestion. It is intentionally separate from the CLI and from the storage layer so that parsing rules, directory scanning, and ingestion policies can evolve independently.

Responsibilities:
- Discover files recursively in an ingest directory
- Read files, detect format by extension, and parse them into normalized documents
- Normalize metadata (title, source path, content)
- Decide new vs. updated documents based on existing `rag::model::document::Document` records
- Report per-file status and aggregate summaries back to the caller
- Skip files that contain no extractable text (empty or whitespace-only text files, scanned/image-only PDFs)

Non-responsibilities:
- Direct database access beyond what `rag::RagStore` provides
- UI rendering or command parsing (owned by `lala`)
- Chunking, FTS indexing, or embedding storage (owned by `rag`)

## Module Layout

```
documents/src/
  lib.rs          # Public exports
  discovery.rs    # scan_directory(), recursive file collection
  parser.rs       # parse_document(), parse_text(), parse_pdf()
  ingest.rs       # ingest_file(), ingest_directory(), IngestSummary
  types.rs        # ParsedDocument, DocumentParseResult, FileIngestStatus
```

## Public API

```rust
use documents::{ingest_file, ingest_directory, FileIngestStatus, IngestSummary};
use rag::RagStore;

// Single file (text, Markdown, PDF, etc.)
let status = ingest_file(&store, "./ingest/notes.txt");
let status = ingest_file(&store, "./ingest/report.pdf");

// Directory with progress callback
let summary = ingest_directory(&store, "./ingest", |current, total, filename, status| {
    println!("[{current}/{total}] {filename}: {status:?}");
}).expect("ingest failed");
```

## Parsing Behavior

`parse_document()` selects a parser based on file extension:

| Extension | Parser | Notes |
|-----------|--------|-------|
| `.pdf` (case-insensitive) | `anydoc::to_markdown()` | Converts PDF content to Markdown. |
| everything else | Plain text | Read directly as UTF-8. |

Skipped-content handling:

- Text/Markdown files that are empty or contain only whitespace are reported as `Skipped("file is empty")`.
- PDFs that produce no Markdown output or raise `anydoc::ConvertError::Malformed` are reported as `Skipped("no extractable text")`. This covers scanned/image-only PDFs and other documents with no meaningful text.
- Other conversion errors (I/O, encrypted, unsupported, etc.) remain `Failed`.

The caller sees a `FileIngestStatus`:

```rust
pub enum FileIngestStatus {
    New(usize),
    Updated(usize),
    Skipped(String),
    Failed(String),
}
```

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `LALA_INGEST_DIR` | `./ingest` | Directory scanned by `ingest_dir_from_env_or_default()` |

## Dependencies

| Crate | Purpose |
|-------|---------|
| `anyhow` | Error propagation |
| `anydoc` | Document-to-Markdown conversion (PDF, DOCX, ODT, etc.) |
| `rag` (path) | RAG store for chunking and ingestion |

## Future Work

- Extension-based parser registry (DOCX, HTML, RTF, EPUB)
- File-hash or mtime-based change detection
- Asynchronous/streaming ingestion for large directories
