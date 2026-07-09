# lala.ai

`lala.ai` is a local Agentic RAG system for running a Rust CLI and Python inference service together. It is designed to help you start quickly, ingest text content, and use a local LLM with retrieval-augmented reasoning.

## What is lala.ai?

`lala.ai` combines three main components:

- `lala` — a Rust CLI client with a terminal REPL, project support, document ingestion, search, and RAG-aware question routing.
- `LLML` — a Python FastAPI inference server that loads local GGUF models and exposes OpenAI-compatible APIs.
- PostgreSQL + pgvector — the retrieval backend for document chunks, embeddings, and memory search.

## Why use it?

- Local-first: run the model inference server on your machine with GGUF models.
- Agentic workflow: smart query routing chooses direct answers or multi-step reasoning.
- Project-aware ingestion: ingest files, RSS feeds, and perform keyword/memory search per project.
- Extensible architecture: Rust CLI, Python API, and PostgreSQL backend are all separable.

## Get Started

Follow the first-run path below.

### 00 Install Docker Desktop

`lala serve` depends on Docker being available locally. For Windows and macOS users, install Docker Desktop and verify Docker is running before starting `lala`.

> Install Docker Desktop and verify Docker is running.

### 01 Download the Windows binary

Use GitHub Releases as the distribution channel and download the release binary directly.

- Download `lala-v1.x.x-windows-amd64.exe` from the latest GitHub Release.
- Do not rely on browsing the repository source as the primary install path.

### 02 Bring up the local runtime

Run the bundled runtime using the downloaded binary.

```sh
lala serve
```

This starts the local inference layer and PostgreSQL, then prints or persists the connection info.

### 03 Start lala and create a project

The first-run UX should immediately teach the project workflow.

```sh
lala
/project create --name my-notes
/ingest ./docs
```

On first run, `lala serve` is the canonical path. The binary path should be the primary onboarding path, not `cargo run`.

## Quick Commands

Use these commands inside the `lala` REPL:

- `/project create --name <name>` — create and select a project
- `/project select <name-or-id>` — select an existing project
- `/ingest [dir]` — ingest all files from `./ingest/`
- `/ingest-file <path>` — ingest one file
- `/ingest-news <url>` — ingest RSS feed articles
- `/search <query>` — search ingested document chunks
- `/memory-search <query>` — search structured memory blocks
- `Plan: <query>` — generate a project-specific plan
- `/clear` — reset conversation history
- `/help` — show available commands
- `/exit` or `/quit` — quit the REPL

## Recommended Run Workflow

1. Start LLML.
2. Start `lala`.
3. Create or select a project.
4. Ingest documents or RSS feeds.
5. Ask questions and use `/search` or `/memory-search` as needed.

## Configuration

- `ai-config.yaml` is read by the LLML server only.
- `LLML_API_URL` points the CLI to the inference server.
- `DATABASE_URL` points the CLI to PostgreSQL.

## Documentation

For full technical reference, API details, and architecture, see:

- `doc/product-details.md` — full product reference moved from the root README
- `doc/architecture.md` — system architecture and flow
- `doc/lala.md` — CLI client internals and commands
- `doc/LLML.md` — inference server internals and config
- `doc/RAG.md` — retrieval store and database design

---

## Need help?

- Start the LLML API docs at `http://localhost:3000/docs`
- Use `/help` inside the `lala` REPL
- Review `doc/product-details.md` for full configuration, API, and design details
