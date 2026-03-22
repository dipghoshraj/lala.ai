use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use crate::agent::model::{ApiClient, ChatMessage};

// Braille spinner — visible in any modern terminal (Windows Terminal, VS Code, etc.)
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const SPINNER_LABEL: &str = "thinking or you can say simulating pattern recognition";

const SYSTEM_PROMPT: &str =
    "You are a friendly AI assistant named lala. \
     Explain things clearly and naturally. \
     Respond in full sentences.";

pub fn run(api_url: &str) -> anyhow::Result<()> {
    let client = ApiClient::new(api_url);
    let mut rl = DefaultEditor::new()?;

    // Conversation history — system prompt is permanently at index 0.
    let mut history: Vec<ChatMessage> = vec![ChatMessage {
        role: "system".to_string(),
        content: SYSTEM_PROMPT.to_string(),
    }];

    println!("lala  —  connected to {api_url}");
    println!("Commands: /clear  reset conversation | /exit  quit\n");

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

        match input.as_str() {
            "/exit" | "/quit" => break,
            "/clear" => {
                history.truncate(1); // keep system prompt
                println!("Conversation cleared.\n");
                continue;
            }
            _ => {}
        }

        history.push(ChatMessage {
            role: "user".to_string(),
            content: input.clone(),
        });

        // ── Spinner ────────────────────────────────────────────────────────
        let running = Arc::new(AtomicBool::new(true));
        let spin_flag = Arc::clone(&running);
        let spinner_handle = thread::spawn(move || {
            let mut i = 0usize;
            loop {
                if !spin_flag.load(Ordering::Relaxed) {
                    break;
                }
                print!("\r  {} {}...", SPINNER[i % SPINNER.len()], SPINNER_LABEL);
                io::stdout().flush().ok();
                i += 1;
                thread::sleep(Duration::from_millis(80));
            }
            // Erase the spinner line cleanly.
            print!("\r{}\r", " ".repeat(SPINNER_LABEL.len() + 16));
            io::stdout().flush().ok();
        });

        let result = client.chat(&history, None, None);

        // Stop the spinner before printing anything.
        running.store(false, Ordering::Relaxed);
        spinner_handle.join().ok();
        // ──────────────────────────────────────────────────────────────────

        match result {
            Ok(reply) => {
                println!("{}\n", reply);
                history.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: reply,
                });
            }
            Err(e) => {
                eprintln!("Error: {e}\n");
                // Remove the user turn so history stays consistent.
                history.pop();
            }
        }
    }

    println!("Bye!");
    Ok(())
}

