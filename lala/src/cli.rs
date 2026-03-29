use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use crate::agent::model::{ApiClient, ChatMessage, RouteDecision};
use crate::agent::planner::{Agent, needs_reasoning};

// Braille spinner — visible in any modern terminal (Windows Terminal, VS Code, etc.)
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// ── ANSI colour codes ─────────────────────────────────────────────────────────
const RESET: &str = "\x1b[0m";
const BOLD_YELLOW: &str = "\x1b[1;33m"; // reasoning — header & border
const DIM_YELLOW: &str = "\x1b[2;33m";  // reasoning — body text
const BOLD_CYAN: &str = "\x1b[1;36m";   // answer — header & border
const CYAN: &str = "\x1b[36m";          // answer — body text
const SECTION_WIDTH: usize = 60;

const SYSTEM_PROMPT: &str =
    "You are a friendly AI assistant named lala. \
     Explain things clearly and naturally. \
     Respond in full sentences.";

/// Run `f` while displaying a braille spinner labelled `label`.
/// Stops and erases the spinner line before returning.
fn with_spinner<F, T>(label: &str, f: F) -> T
where
    F: FnOnce() -> T,
{
    let running = Arc::new(AtomicBool::new(true));
    let spin_flag = Arc::clone(&running);
    let label_owned = label.to_string();
    let clear_len = label.len() + 8;
    let spinner_handle = thread::spawn(move || {
        let mut i = 0usize;
        loop {
            if !spin_flag.load(Ordering::Relaxed) {
                break;
            }
            print!("\r  {} {}...", SPINNER[i % SPINNER.len()], label_owned);
            io::stdout().flush().ok();
            i += 1;
            thread::sleep(Duration::from_millis(80));
        }
        print!("\r{}\r", " ".repeat(clear_len));
        io::stdout().flush().ok();
    });
    let result = f();
    running.store(false, Ordering::Relaxed);
    spinner_handle.join().ok();
    result
}

/// Print a titled section with a coloured header and separator.
///
/// ```
/// ▷ Reasoning
/// ────────────────────────────────────────────────────────────
/// [body text]
/// ────────────────────────────────────────────────────────────
/// ```
fn print_section(header: &str, header_color: &str, text_color: &str, content: &str) {
    let sep = "─".repeat(SECTION_WIDTH);
    println!("\n{}{} {}{}", header_color, "▷", header, RESET);
    println!("{}{}{}", header_color, sep, RESET);
    println!("{}{}{}", text_color, content.trim(), RESET);
    println!("{}{}{}", header_color, sep, RESET);
    println!();
}

pub fn run(api_url: &str, smart_router: bool) -> anyhow::Result<()> {
    let client = ApiClient::new(api_url);
    let agent = Agent::new(&client);
    let mut rl = DefaultEditor::new()?;

    // Conversation history — system prompt is permanently at index 0.
    let mut history: Vec<ChatMessage> = vec![ChatMessage {
        role: "system".to_string(),
        content: SYSTEM_PROMPT.to_string(),
    }];

    println!("lala  —  connected to {api_url}");
    if smart_router {
        println!("Router: LLM classifier (LALA_SMART_ROUTER=1)");
    }
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

        // ── Router: classify query to skip reasoning when not needed ──────
        let route = if smart_router {
            agent.classify_query(&input, &history)
        } else if needs_reasoning(&input) {
            RouteDecision::Reasoning
        } else {
            RouteDecision::Direct
        };

        if route == RouteDecision::Direct {
            let result = with_spinner("thinking", || agent.run_direct(&history));
            match result {
                Ok(reply) => {
                    print_section("Answer", BOLD_CYAN, CYAN, &reply);
                    history.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: reply,
                    });
                }
                Err(e) => {
                    eprintln!("{}Error: {}{}", BOLD_CYAN, e, RESET);
                    history.pop();
                }
            }
            continue;
        }

        // ── Step 1: Reasoning ─────────────────────────────────────────────
        let reasoning_result = with_spinner("reasoning", || agent.run_reasoning(&history));

        match reasoning_result {
            Err(e) => {
                eprintln!("{}Error during reasoning: {}{}\n", BOLD_YELLOW, e, RESET);
                history.pop();
                continue;
            }
            Ok(analysis) => {
                print_section("Reasoning", BOLD_YELLOW, DIM_YELLOW, &analysis);

                // ── Step 2: Decision ───────────────────────────────────────
                let decision_result =
                    with_spinner("deciding", || agent.run_decision(&history, &analysis));

                match decision_result {
                    Ok(reply) => {
                        print_section("Answer", BOLD_CYAN, CYAN, &reply);
                        history.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: reply,
                        });
                    }
                    Err(e) => {
                        eprintln!("{}Error: {}{}\n", BOLD_CYAN, e, RESET);
                        history.pop();
                    }
                }
            }
        }
    }

    println!("Bye!");
    Ok(())
}

