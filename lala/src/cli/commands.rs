use rag::RagStore;

use super::display;
use super::ingest;

/// Dispatch a `/`-prefixed command. Returns `true` if it was handled.
///
/// Returns `false` for unknown commands so the REPL can treat them as chat input.
pub fn dispatch(input: &str, store: &RagStore) -> CommandResult {
    let (cmd, args) = split_command(input);

    match cmd {
        "/exit" | "/quit" => CommandResult::Exit,
        "/clear" => CommandResult::Clear,
        "/help" => {
            print_help();
            CommandResult::Handled
        }
        "/status" => {
            print_status(store);
            CommandResult::Handled
        }
        "/ingest" => {
            ingest::ingest_all(store, args);
            CommandResult::Handled
        }
        "/ingest-file" => {
            ingest::ingest_file(store, args);
            CommandResult::Handled
        }
        "/search" => {
            search(store, args);
            CommandResult::Handled
        }
        _ => CommandResult::NotACommand,
    }
}

/// Result of command dispatch.
pub enum CommandResult {
    /// Command ran successfully, continue REPL.
    Handled,
    /// User wants to exit.
    Exit,
    /// User wants to clear conversation.
    Clear,
    /// Input was not a recognised command — treat as chat.
    NotACommand,
}

/// Split `/command args...` into the command token and the rest.
fn split_command(input: &str) -> (&str, &str) {
    match input.find(' ') {
        Some(pos) => (&input[..pos], input[pos..].trim()),
        None => (input, ""),
    }
}

// ── /help ─────────────────────────────────────────────────────────────────────

fn print_help() {
    let sep = "─".repeat(display::SECTION_WIDTH);
    println!();
    println!("{}{}{}", display::DIM, sep, display::RESET);
    println!(
        "  {}Commands{}",
        display::BOLD,
        display::RESET,
    );
    println!("{}{}{}", display::DIM, sep, display::RESET);
    println!(
        "  {}/ingest [dir]{}      Batch-ingest files (recursive); defaults to ./ingest/",
        display::BOLD_GREEN,
        display::RESET,
    );
    println!(
        "  {}/ingest-file <p>{}  Ingest a single file by path",
        display::BOLD_GREEN,
        display::RESET,
    );
    println!(
        "  {}/search <query>{}   Search ingested documents (BM25)",
        display::BOLD_CYAN,
        display::RESET,
    );
    println!(
        "  {}/status{}            Show database statistics",
        display::CYAN,
        display::RESET,
    );
    println!(
        "  {}/clear{}             Reset conversation history",
        display::YELLOW,
        display::RESET,
    );
    println!(
        "  {}/help{}              Show this help",
        display::DIM,
        display::RESET,
    );
    println!(
        "  {}/exit{}              Quit",
        display::DIM,
        display::RESET,
    );
    println!("{}{}{}", display::DIM, sep, display::RESET);
    println!();
}

// ── /search ───────────────────────────────────────────────────────────────────

fn search(store: &RagStore, query: &str) {
    if query.is_empty() {
        println!("Usage: /search <query>\n");
        return;
    }

    match store.retrieve(query, 5) {
        Ok(chunks) if chunks.is_empty() => {
            display::warn(&format!("No results found for: {query}"));
            println!();
        }
        Ok(chunks) => {
            println!();
            let sep = "─".repeat(display::SECTION_WIDTH);
            println!("{}{}{}", display::DIM, sep, display::RESET);
            for (i, c) in chunks.iter().enumerate() {
                let preview: String = c.chunk_text.chars().take(100).collect();
                println!(
                    "  {}[{}]{} score: {}{:.4}{}  chunk #{}",
                    display::BOLD,
                    i + 1,
                    display::RESET,
                    display::CYAN,
                    c.score,
                    display::RESET,
                    c.chunk_index,
                );
                println!(
                    "      {}{}…{}",
                    display::DIM,
                    preview,
                    display::RESET,
                );
                println!();
            }
            println!("{}{}{}", display::DIM, sep, display::RESET);
            println!();
        }
        Err(e) => {
            display::error(&format!("Search error: {e}"));
            println!();
        }
    }
}

// ── /status ───────────────────────────────────────────────────────────────────

fn print_status(store: &RagStore) {
    let docs = match store.document_count() {
        Ok(count) => count.to_string(),
        Err(e) => {
            display::error(&format!("Failed to read document count: {e}"));
            "N/A".to_string()
        }
    };
    let chunks = match store.chunk_count() {
        Ok(count) => count.to_string(),
        Err(e) => {
            display::error(&format!("Failed to read chunk count: {e}"));
            "N/A".to_string()
        }
    };
    let ingest_dir = std::env::var("LALA_INGEST_DIR").unwrap_or_else(|_| "./ingest".to_string());

    println!();
    let sep = "─".repeat(display::SECTION_WIDTH);
    println!("{}{}{}", display::DIM, sep, display::RESET);
    println!(
        "  {}Documents:{} {}    {}Chunks:{} {}",
        display::BOLD, display::RESET, docs,
        display::BOLD, display::RESET, chunks,
    );
    println!(
        "  {}Ingest dir:{} {}",
        display::BOLD, display::RESET, ingest_dir,
    );
    println!("{}{}{}", display::DIM, sep, display::RESET);
    println!();
}
