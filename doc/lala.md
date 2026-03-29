# lala — CLI Client

> **Location:** `lala.ai/lala/`  
> **Role:** User-facing layer — interactive terminal REPL that sends conversation turns to the LLML API and displays responses with a live spinner.

---

## Overview

`lala` is the front-end of the system. It owns the user experience: readline input, multi-turn conversation history, a spinner animation during inference, and clean error recovery. It has no direct knowledge of the model — all LLM communication goes through HTTP to the LLML server.

```
User (terminal)
      │
  rustyline REPL
      │
  conversation history (in-memory)
      │
  POST /v1/chat/completions  ──►  LLML API server
      │
  spinner thread  (while waiting)
      │
  print response
```

---

## Source Layout

```
lala/src/
  main.rs          # Entry point — resolves API URL, starts CLI
  cli.rs           # REPL loop, spinner, conversation history management
  agent/
    mod.rs
    model.rs       # ApiClient — HTTP wrapper for the LLML chat endpoint
  db/
    connection.rs  # PostgreSQL helpers (document_chunks, memory) — future use
```

---

## Running

```sh
cd lala

# Default — connects to http://localhost:3000
cargo run

# Custom server URL via argument
cargo run -- http://192.168.1.10:3000

# Custom server URL via environment variable
LLML_API_URL=http://192.168.1.10:3000 cargo run
```

URL resolution priority: **CLI argument → `LLML_API_URL` env var → `http://localhost:3000`**

---

## CLI Commands

| Input     | Action |
|-----------|--------|
| Any text  | Send as a user message to the LLM |
| `/clear`  | Reset conversation history (keeps system prompt) |
| `/exit`   | Quit |
| Ctrl-C / Ctrl-D | Quit |

Arrow-key history navigation (up/down) is provided by `rustyline`.

---

## Conversation History

The full message history is maintained in memory for the duration of the session. Each turn appends a `user` and `assistant` message:

```
[ system prompt, user1, assistant1, user2, assistant2, ... ]
```

The entire history is sent with every request so the model maintains context across turns. `/clear` trims the list back to just the system prompt, starting a fresh conversation without restarting the process.

If the API returns an error, the pending `user` message is removed from history so the context stays consistent.

---

## Spinner

While waiting for the API response, a braille spinner runs on a background thread:

```
  ⠼ thinking...
```

The spinner is stopped and the line is erased before the response is printed, so the output is clean:

```
>> What is Rust?
  ⠼ thinking...          ← live, while waiting
Rust is a systems programming language...   ← replaces the spinner line
```

Implementation: `Arc<AtomicBool>` stop flag shared between the main thread and the spinner thread. When the API call returns (success or error), the flag is set to `false`, the spinner thread exits, and the cursor line is cleared with spaces before any output is printed.

---

## HTTP Client — `agent/model.rs`

### `ApiClient`

```rust
ApiClient::new(base_url: &str) -> ApiClient
```

Wraps a `reqwest::blocking::Client` configured with **no timeout** — CPU inference can take tens of seconds and must not be aborted prematurely.

```rust
client.chat(messages: &[ChatMessage], max_tokens: Option<usize>) -> anyhow::Result<String>
```

Sends a `POST /v1/chat/completions` request with the full conversation history and returns the assistant's reply text. Handles HTTP error status codes and malformed responses as `anyhow::Error`.

### `ChatMessage`

```rust
pub struct ChatMessage {
    pub role: String,    // "system" | "user" | "assistant"
    pub content: String,
}
```

Shared between the CLI history and the HTTP request body — same type used everywhere.

---

## System Prompt

The system prompt is hardcoded in `cli.rs` and always occupies index 0 of the history:

```
You are a friendly AI assistant named lala.
Explain things clearly and naturally.
Respond in full sentences.
```

This can be edited in `cli.rs` under `const SYSTEM_PROMPT`.

---

## Dependencies

| Crate       | Purpose |
|-------------|---------|
| `reqwest`   | Blocking HTTP client for LLML API calls |
| `rustyline` | Readline-style input with history and arrow-key navigation |
| `serde` / `serde_json` | HTTP request/response serialization |
| `anyhow`    | Error propagation |
| `sqlx`      | PostgreSQL client (wired for future memory/RAG integration) |

---

## System Architecture

`lala` and `LLML` are separate processes that communicate over HTTP. They can run on the same machine or on different machines on the same network.

```
┌─────────────┐          HTTP POST            ┌──────────────────┐
│   lala CLI  │  ────────────────────────►   │   LLML server    │
│  (lala/)    │  /v1/chat/completions         │   (LLML/)        │
│             │  ◄────────────────────────   │                  │
│  User REPL  │  JSON response               │  llama_cpp model │
└─────────────┘                              └──────────────────┘
```

See [LLML.md](LLML.md) for the server-side documentation.
