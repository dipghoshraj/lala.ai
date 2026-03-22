mod agent;
mod cli;

fn main() -> anyhow::Result<()> {
    // API URL from CLI arg, then env var, then default.
    let api_url = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("LLML_API_URL").ok())
        .unwrap_or_else(|| "http://localhost:3000".to_string());

    cli::run(&api_url)
}
