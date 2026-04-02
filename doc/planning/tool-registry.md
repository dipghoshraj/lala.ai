# Tool Registry — Agent Tool Integration

> **Status:** Planned
> **Depends on:** [phase0-rag.md](phase0-rag.md) — RAG retrieval layer
> **Goal:** Introduce a standalone `tools` Rust workspace crate providing a `Tool` trait, `ToolRegistry`, and built-in tools. Wire a hybrid tool-selection step (keyword pre-filter → LLM confirmation) into the `lala` agent on both direct and reasoning paths.

---

## 1. Background & Motivation

The current agent flow is:

```
Chat::handle(input)
  → classify() → RouteDecision::Direct | RouteDecision::Reasoning
  → run_direct()    — single model call (decision model)
  → run_reasoning() — RAG retrieval → reasoning model → decision model
```

There is no mechanism for the agent to call external tools or APIs at runtime. Adding a dynamic tool registry enables the agent to:

- Fetch live data such as current time, file contents, and later web/API results
- Route to the right tool automatically without hard-coding tool names in the agent
- Keep the `tools` crate reusable across projects with no dependency on `lala`, `rag`, or LLML

---

## 2. Architecture Overview

```
tools/                        ← NEW standalone workspace crate
  Cargo.toml                  deps: anyhow, serde+derive, serde_json, chrono
  src/
    lib.rs                    Tool trait, ToolDescription, re-exports
    registry.rs               ToolRegistry (HashMap<String, Box<dyn Tool + Send + Sync>>)
    builtin/
      mod.rs
      time.rs                 GetCurrentTimeTool — chrono::Utc::now(), ISO8601
      file.rs                 FileReaderTool — std::fs, absolute paths only

lala/src/tools/               ← NEW project-specific tools (depends on rag)
  mod.rs
  rag.rs                      RagSearchTool — opens fresh rusqlite connection per call

lala/src/agent/model.rs       ← EXTEND — add ToolSelection + ApiClient::select_tool()
lala/src/agent/planner.rs     ← EXTEND — Agent gains tool_registry, run_tool_calls()
lala/src/cli/chat.rs          ← EXTEND — Chat owns ToolRegistry, wires both paths
```

---

## 3. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tool selection strategy | Hybrid: keyword pre-filter → LLM confirmation | Keyword filter avoids LLM round-trip cost on irrelevant queries; LLM extracts structured input parameters |
| Invocation point | Both `run_direct()` and `run_reasoning()` paths | Tools can be needed on both simple and complex queries |
| Tool I/O format | `serde_json::Value` in / `String` out | Flexible for multi-parameter tools and avoids per-tool parsing in agent code |
| LLM tool-select response format | `TOOL_NAME: <name>\nTOOL_INPUT: <json>` or `NO_TOOL` | Plain text, easy to parse without structured output support |
| `RagSearchTool` location | `lala/src/tools/` not `tools/` | Depends on `rusqlite` and project db path; keeps `tools` crate dependency-free |
| `rusqlite::Connection` + `Sync` | Open fresh read-only connection per call | `Connection` is `!Sync`; a fresh connection is safe for read-only queries with no shared state |

---

## 4. Implementation Plan

### Phase 1 — `tools` crate

**Step 1.** Add `"tools"` to workspace `members` in root `Cargo.toml`.

**Step 2.** Create `tools/Cargo.toml` with:

- `anyhow = "1.0"`
- `serde = { version = "1", features = ["derive"] }`
- `serde_json = "1"`
- `chrono = { version = "0.4", features = ["serde"] }`

**Step 3.** Create `tools/src/lib.rs` with:

- `ToolDescription { name: String, description: String, keywords: Vec<String> }`
- `Tool` trait with `name(&self) -> &str`, `description(&self) -> &str`, `keywords(&self) -> &[&str]`, `execute(&self, input: serde_json::Value) -> anyhow::Result<String>`
- re-export of `ToolRegistry`

**Step 4.** Create `tools/src/registry.rs` with:

- `tools: HashMap<String, Box<dyn Tool + Send + Sync>>`
- `register(tool: Box<dyn Tool + Send + Sync>)`
- `get(name: &str) -> Option<&(dyn Tool + Send + Sync)>`
- `all_descriptions() -> Vec<ToolDescription>`
- `keyword_candidates(query: &str) -> Vec<String>` based on lowercased keyword substring matches

**Step 5.** Create `tools/src/builtin/time.rs` for `GetCurrentTimeTool`:

- keywords: `time`, `date`, `clock`, `now`, `today`
- returns `chrono::Utc::now().to_rfc3339()`

**Step 6.** Create `tools/src/builtin/file.rs` for `FileReaderTool`:

- keywords: `read file`, `file`, `open file`, `file contents`
- expects `input["path"]`
- validates absolute path, rejects `..`, confirms file exists, returns UTF-8 contents

---

### Phase 2 — Project-Specific Tools In `lala`

**Step 7.** Add `tools = { path = "../tools" }` to `lala/Cargo.toml`.

**Step 8.** Create `lala/src/tools/mod.rs` and re-export `RagSearchTool`.

**Step 9.** Create `lala/src/tools/rag.rs` with `RagSearchTool { db_path: String }`:

- keywords: `search`, `find`, `retrieve`, `lookup`, `documents`, `knowledge`
- opens `RagStore::open(&self.db_path)`
- reads `input["query"]`
- calls `store.retrieve(query, 5)` and formats the chunks as text

---

### Phase 3 — `ApiClient` Extension

**Step 10.** Add `ToolSelection { tool_name: String, input: serde_json::Value }` to `lala/src/agent/model.rs`.

**Step 11.** Add `ApiClient::select_tool(query, candidate_descriptions, history) -> anyhow::Result<Option<ToolSelection>>`:

- builds a tool-selection prompt listing only candidate tools
- asks the reasoning model to reply with `TOOL_NAME` and `TOOL_INPUT` or `NO_TOOL`
- parses the response and deserializes the JSON input payload

---

### Phase 4 — `Agent` Integration

**Step 12.** Add `tool_registry: Option<&'a tools::ToolRegistry>` to `Agent<'a>` and update `Agent::new()`.

**Step 13.** Add `Agent::run_tool_calls(&self, query: &str, history: &[ChatMessage]) -> anyhow::Result<Option<String>>`:

- if no registry is present, return `Ok(None)`
- run `keyword_candidates(query)` first; if empty, skip tool selection entirely
- build descriptions for the candidate tools only
- call `client.select_tool(...)`
- execute the selected tool and return its output

**Step 14.** Update `run_reasoning()` and `run_direct()` to accept optional tool output and inject it into the model context when present.

---

### Phase 5 — `Chat` Wiring

**Step 15.** Add an owned `ToolRegistry` to `Chat` and register:

- `GetCurrentTimeTool`
- `FileReaderTool`
- `RagSearchTool { db_path }`

**Step 16.** Pass `Some(&self.tool_registry)` into `Agent::new()`.

**Step 17.** In `run_direct()`:

- call `agent.run_tool_calls()` before the decision model call
- if a tool result exists, pass it as context to the decision model

**Step 18.** In `run_reasoning()`:

- call `agent.run_tool_calls()` after RAG retrieval
- merge tool output with retrieved chunk context before running reasoning and decision

---

## 5. Data Flow After Integration

```
Chat::handle(input)
  → classify() → RouteDecision
  → run_tool_calls(input)          ← NEW: keyword filter → LLM select → execute
      → keyword_candidates()       ← no LLM cost if no keyword matches
      → client.select_tool()       ← only called if candidates exist
      → tool.execute(input)
  → run_direct(tool_result)        or
  → run_reasoning(rag_ctx + tool_result) → run_decision(...)
```

---

## 6. File Inventory

| File | Action |
|------|--------|
| `Cargo.toml` | Add `"tools"` to workspace members |
| `tools/Cargo.toml` | Create |
| `tools/src/lib.rs` | Create |
| `tools/src/registry.rs` | Create |
| `tools/src/builtin/mod.rs` | Create |
| `tools/src/builtin/time.rs` | Create |
| `tools/src/builtin/file.rs` | Create |
| `lala/Cargo.toml` | Add `tools` path dependency |
| `lala/src/tools/mod.rs` | Create |
| `lala/src/tools/rag.rs` | Create |
| `lala/src/agent/model.rs` | Extend with `ToolSelection` and `select_tool()` |
| `lala/src/agent/planner.rs` | Extend with `tool_registry` and `run_tool_calls()` |
| `lala/src/cli/chat.rs` | Extend to own the registry and wire both paths |

---

## 7. Verification Checklist

- [ ] `cargo build -p tools` succeeds
- [ ] `cargo build -p lala` succeeds
- [ ] Input `what time is it` triggers `GetCurrentTimeTool`
- [ ] Input requesting file contents triggers `FileReaderTool`
- [ ] Input requesting document search triggers `RagSearchTool`
- [ ] Input `hello` triggers no tool call and keeps the current direct path unchanged
- [ ] Queries with no keyword hits still behave normally through direct or reasoning flow