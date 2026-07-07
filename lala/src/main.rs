mod agent;
mod cli;
mod config;
mod llml;

use crate::config::LalaConfig;
use rag;

use figlet_rs::FIGfont;
use colored::*;

fn print_banner() {
    let standard_font = FIGfont::standard().unwrap();
    let figure = standard_font.convert("lala.ai");

    if let Some(fig) = figure {
        println!("{}", fig.to_string().bright_green());
    }

    println!("{}", "Intelligent • Local first • Reasoning".green());
    println!("{}", "⚡ Built with Local First Principles".green());
}

fn main() -> anyhow::Result<()> {
    let config = LalaConfig::load(None)?;
    print_banner();

    // API URL from CLI arg, then env var, then default.
    let mut args = std::env::args().skip(1);
    let first_arg = args.next();

    if let Some(cmd) = first_arg.as_deref() {
        if cmd == "serve" {
            let ports = llml::allocate_available_ports(2)?;
            let llml_port = ports[0];
            let psql_port = ports[1];
            let api_url = format!("http://127.0.0.1:{llml_port}");
            let database_url = format!(
                "postgres://{user}:{password}@127.0.0.1:{port}/{db}",
                user = config.database.user,
                password = config.database.password,
                port = psql_port,
                db = config.database.name,
            );

            let model_dir = config.model_dir.clone();
            println!("Using model directory: {}", model_dir);
            llml::start_llml_docker(llml_port, &model_dir)?;
            llml::start_postgres_docker(
                psql_port,
                &config.database.user,
                &config.database.password,
                &config.database.name,
            )?;

            let serve_env = llml::ServeEnv {
                api_url: api_url.clone(),
                database_url: database_url.clone(),
            };
            llml::write_serve_env(&serve_env)?;

            llml::print_serve_instructions(&api_url, &database_url);
            return Ok(());
        }
    }

    let env = llml::read_serve_env().unwrap_or(None);
    let api_url = first_arg
        .or_else(|| std::env::var("LLML_API_URL").ok())
        .or_else(|| env.as_ref().map(|e| e.api_url.clone()))
        .unwrap_or_else(|| "http://localhost:3000".to_string());

    // Smart routing is enabled by default.
    // Set LALA_SMART_ROUTER=0 to disable LLM-based query classification.
    let smart_router = std::env::var("LALA_SMART_ROUTER")
        .map(|v| v.trim() != "0")
        .unwrap_or(true);

    // Database URL from env var, then temp file fallback, then ai-config defaults.
    let database_url = std::env::var("DATABASE_URL")
        .ok()
        .or_else(|| env.as_ref().map(|e| e.database_url.clone()))
        .unwrap_or_else(|| {
            format!(
                "postgres://{user}:{password}@localhost:5432/{db}",
                user = config.database.user,
                password = config.database.password,
                db = config.database.name,
            )
        });


    // let store = RagStore::open(&database_url)?;
    let _: () = rag::model::init_db(&database_url)?;

    {
        let mut client = rag::model::db().client();
        rag::migrate::run_migrations(&mut client)?;
    }

    let store = rag::RagStore::new();

    cli::run(&api_url, smart_router, store, config)
}
