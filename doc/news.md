# news Crate

> **Location:** `lala.ai/news/`
> **Role:** RSS/news ingestion library — feed fetching, article download, HTML-to-text extraction, and orchestration of storage via `rag::RagStore`.

## Overview

The `news` crate owns RSS-based ingestion. It fetches RSS feeds, downloads linked articles, extracts clean text from HTML, and stores the result through `rag::RagStore`.

Responsibilities:
- Fetch RSS feeds via HTTP
- Download each article with a polite delay between requests
- Extract readable text from HTML (strip scripts, styles, tags, normalize whitespace)
- Fall back to a CORS proxy on HTTP 403 responses
- Ingest articles into `rag::RagStore`
- Report per-article status and aggregate counts

Non-responsibilities:
- UI rendering or command parsing (owned by `lala`)
- Chunking, FTS indexing, or database storage (owned by `rag`)

## Module Layout

```
news/src/
  lib.rs      # Public exports
  ingest.rs   # ingest_news_feed(), ingest_news_feed_with_progress()
  types.rs    # ArticleIngestStatus
```

## Public API

```rust
use news::{ingest_news_feed, ingest_news_feed_with_progress, ArticleIngestStatus};
use rag::RagStore;

// Simple API returning aggregate counts
let (ingested, skipped, failed) = ingest_news_feed(
    &store,
    "https://feeds.bbci.co.uk/news/rss.xml",
    1000,
).expect("news ingest failed");

// Progress-aware API
let (ingested, skipped, failed) = ingest_news_feed_with_progress(
    &store,
    "https://feeds.bbci.co.uk/news/rss.xml",
    1000,
    |title, status| {
        println!("{title}: {status:?}");
    },
).expect("news ingest failed");
```

## Polite Scraping

The crate sleeps `delay_ms` milliseconds between article fetches to avoid hammering origin servers. The default used by the CLI is 1000 ms.

## Future Work

- Feed-level metadata (lastBuildDate, TTL)
- Article deduplication beyond URL/source
- Content-type-aware extraction (JSON feeds, Atom)
- Configurable user-agent and proxy settings
