mod agent;
mod cli;
mod config;

use crate::config::LalaConfig;
use rag::RagStore;

use figlet_rs::FIGfont;
use colored::*;


fn print_banner() {
    let standard_font = FIGfont::standard().unwrap();
    let figure = standard_font.convert("lala.ai");

    if let Some(fig) = figure {
        println!("{}", fig.to_string().bright_green());
    }

    println!("{}", "Intelligent • Developer • Productivity".green());
    println!("{}", "⚡ Built with Rust".green());
}

fn main() -> anyhow::Result<()> {
    let config = LalaConfig::load(None)?;
    print_banner();

    // API URL from CLI arg, then env var, then default.
    let api_url = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("LLML_API_URL").ok())
        .unwrap_or_else(|| "http://localhost:3000".to_string());

    // Set LALA_SMART_ROUTER=1 to enable LLM-based query classification.
    // Unset or any other value keeps the local heuristic.
    let smart_router = std::env::var("LALA_SMART_ROUTER")
        .map(|v| v.trim() == "1")
        .unwrap_or(false);

    // DB path from env var, then default.
    let db_path = std::env::var("LALA_DB_PATH").unwrap_or_else(|_| "./lala.db".to_string());
    let store = RagStore::open(&db_path)?;

    cli::run(&api_url, smart_router, store, config)
}
