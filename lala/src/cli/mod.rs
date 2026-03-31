mod chat;
mod commands;
mod display;
mod ingest;

use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use crate::agent::model::ApiClient;
use rag::RagStore;

use chat::Chat;
use commands::CommandResult;

pub fn run(api_url: &str, smart_router: bool, store: RagStore) -> anyhow::Result<()> {
    let client = ApiClient::new(api_url);
    let mut chat = Chat::new(&client, smart_router);
    let mut rl = DefaultEditor::new()?;

    print_banner(api_url, smart_router);

    loop {
        let line = match rl.readline(">> ") {
            Ok(l) => l,
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
            Err(e) => return Err(e.into()),
        };

        let input = line.trim().to_string();
        if input.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(&input);

        if input.starts_with('/') {
            match commands::dispatch(&input, &store) {
                CommandResult::Exit => break,
                CommandResult::Clear => {
                    chat.clear();
                    continue;
                }
                CommandResult::Handled => continue,
                CommandResult::NotACommand => {
                    display::warn(&format!("Unknown command: {input}"));
                    display::info("Type /help for available commands.");
                    println!();
                    continue;
                }
            }
        }

        chat.handle(&input);
    }

    println!("Bye!");
    Ok(())
}

fn print_banner(api_url: &str, smart_router: bool) {
    println!();
    let sep = "─".repeat(display::SECTION_WIDTH);
    println!("{}{}{}", display::DIM, sep, display::RESET);
    println!(
        "  {}lala{}  —  connected to {}{}{}",
        display::BOLD, display::RESET,
        display::CYAN, api_url, display::RESET,
    );
    if smart_router {
        println!(
            "  {}Router:{} LLM classifier",
            display::DIM, display::RESET,
        );
    }
    println!(
        "  Type {}/help{} for commands",
        display::DIM, display::RESET,
    );
    println!("{}{}{}", display::DIM, sep, display::RESET);
    println!();
}
