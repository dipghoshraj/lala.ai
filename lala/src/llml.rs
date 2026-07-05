use serde::{Deserialize, Serialize};
use std::fs;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Serialize, Deserialize)]
pub struct ServeEnv {
    pub api_url: String,
    pub database_url: String,
}

pub fn temp_env_file() -> PathBuf {
    std::env::temp_dir().join("lala-serve-env.json")
}

pub fn write_serve_env(env: &ServeEnv) -> anyhow::Result<()> {
    let content = serde_json::to_string_pretty(env)?;
    fs::write(temp_env_file(), content)?;
    Ok(())
}

pub fn read_serve_env() -> anyhow::Result<Option<ServeEnv>> {
    let path = temp_env_file();
    if !path.exists() {
        return Ok(None);
    }
    let data = fs::read_to_string(path)?;
    let env = serde_json::from_str(&data)?;
    Ok(Some(env))
}

pub fn allocate_available_ports(count: usize) -> anyhow::Result<Vec<u16>> {
    let mut listeners = Vec::with_capacity(count);
    for _ in 0..count {
        listeners.push(TcpListener::bind("127.0.0.1:0")?);
    }
    let ports = listeners
        .iter()
        .map(|listener| listener.local_addr().unwrap().port())
        .collect();
    drop(listeners);
    Ok(ports)
}

pub fn start_llml_docker(port: u16) -> anyhow::Result<()> {
    let port_mapping = format!("{port}:3000");
    let output = Command::new("docker")
        .arg("run")
        .arg("-d")
        .arg("--rm")
        .arg("-p")
        .arg(&port_mapping)
        .arg("dipghoshraj/llml:latest")
        .output()?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        Err(anyhow::anyhow!(
            "Failed to start LLML Docker container. docker output:\n{}{}",
            stdout,
            stderr
        ))
    }
}

pub fn start_postgres_docker(port: u16, user: &str, password: &str, db_name: &str) -> anyhow::Result<()> {
    let port_mapping = format!("{port}:5432");
    let output = Command::new("docker")
        .arg("run")
        .arg("-d")
        .arg("--rm")
        .arg("-p")
        .arg(&port_mapping)
        .arg("-v")
        .arg("lala-postgres-data:/var/lib/postgresql/data")
        .arg("-e")
        .arg(format!("POSTGRES_USER={user}"))
        .arg("-e")
        .arg(format!("POSTGRES_PASSWORD={password}"))
        .arg("-e")
        .arg(format!("POSTGRES_DB={db_name}"))
        .arg("pgvector/pgvector:pg17")
        .output()?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        Err(anyhow::anyhow!(
            "Failed to start PostgreSQL Docker container. docker output:\n{}{}",
            stdout,
            stderr
        ))
    }
}

pub fn print_serve_instructions(api_url: &str, database_url: &str) {
    println!("LLML and PostgreSQL Docker containers started successfully.");
    println!("LLML API URL: {api_url}");
    println!("DATABASE_URL: {database_url}");
    println!();
    println!("Service URLs were written to: {}", temp_env_file().display());
    println!();
    println!("Set the environment variables before running lala:");
    println!("  PowerShell:");
    println!("    $env:LLML_API_URL = '{api_url}'");
    println!("    $env:DATABASE_URL = '{database_url}'");
    println!("  CMD:");
    println!("    set LLML_API_URL={api_url}");
    println!("    set DATABASE_URL={database_url}");
    println!("Then run: cd lala && cargo run");
}
