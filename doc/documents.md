# documents Crate

> **Location:** `lala.ai/documents/`
> **Role:** Document ingestion library — file discovery, parsing, normalization, and orchestration of storage via `rag::RagStore`.

## Overview

The `documents` crate owns everything between the filesystem and the RAG store for file-based ingestion. It is intentionally separate from the CLI and from the storage layer so that parsing rules, directory scanning, and ingestion policies can evolve independently.

Responsibilities:
- Discover files recursively in an ingest directory
- Read files and normalize metadata (title, source path, content)
- Decide new vs. updated documents based on existing `rag::model::document::Document` records
- Report per-file status and aggregate summaries back to the caller

Non-responsibilities:
- Direct database access beyond what `rag::RagStore` provides
- UI rendering or command parsing (owned by `lala`)
- Chunking, FTS indexing, or embedding storage (owned by `rag`)

## Module Layout

```
documents/src/
  lib.rs          # Public exports
  discovery.rs    # scan_directory(), recursive file collection
  parser.rs       # parse_document(), parse_text()
  ingest.rs       # ingest_file(), ingest_directory(), IngestSummary
  types.rs        # ParsedDocument, DocumentParseResult, FileIngestStatus
```

## Public API

```rust
use documents::{ingest_file, ingest_directory, FileIngestStatus, IngestSummary};
use rag::RagStore;

// Single file
let status = ingest_file(&store, "./ingest/notes.txt");

// Directory with progress callback
let summary = ingest_directory(&store, "./ingest", |current, total, filename, status| {
    println!("[{current}/{total}] {filename}: {status:?}");
}).expect("ingest failed");
```

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `LALA_INGEST_DIR` | `./ingest` | Directory scanned by `ingest_dir_from_env_or_default()` |

## Future Work

- Extension-based parser registry (Markdown, PDF, DOCX, HTML)
- File-hash or mtime-based change detection
- Asynchronous/streaming ingestion for large directories
