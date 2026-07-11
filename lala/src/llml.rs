use serde::{Deserialize, Serialize};
use std::fs;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Command;

const LLML_CONTAINER_NAME: &str = "lala-llml-server";
const POSTGRES_CONTAINER_NAME: &str = "lala-postgres-server";

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

fn ensure_docker_available() -> anyhow::Result<()> {
    let output = Command::new("docker").arg("-v").output();
    match output {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            Err(anyhow::anyhow!(
                "Docker is not available or failed to respond to 'docker -v'.\n{}{}\nPlease make sure Docker is installed and the Docker daemon is running.",
                stdout,
                stderr
            ))
        }
        Err(err) => Err(anyhow::anyhow!(
            "Unable to execute 'docker -v': {err}. Please install Docker and make sure the Docker CLI is available in your PATH, then start Docker.",
        )),
    }
}

pub fn start_llml_docker(port: u16, models_dir: &str) -> anyhow::Result<()> {
    ensure_docker_available()?;
    let port_mapping = format!("{port}:3000");

    let config_path_env = std::env::var("CONFIG_PATH").ok();
    let config_path = config_path_env
        .clone()
        .unwrap_or_else(|| "ai-config.yaml".to_string());

    if models_dir.trim().is_empty() {
        return Err(anyhow::anyhow!(
            "ai-config.yaml must define 'model_dir' and it cannot be empty."
        ));
    }

    println!("model dir: {}", models_dir);

    let models_path = std::path::PathBuf::from(models_dir);
    let models_path = if models_path.is_absolute() {
        models_path
    } else {
        std::env::current_dir()?.join(models_path)
    };

    println!("Resolved model_dir path: {}", models_path.display());

    if !models_path.is_dir() {
        return Err(anyhow::anyhow!(
            "The model_dir path '{}' does not exist or is not a directory.",
            models_path.display()
        ));
    }

    let config_path = std::path::PathBuf::from(&config_path);
    let config_path = if config_path.is_absolute() {
        config_path
    } else {
        std::env::current_dir()?.join(config_path)
    };

    if !config_path.is_file() {
        if config_path_env.is_some() {
            return Err(anyhow::anyhow!(
                "CONFIG_PATH was set to '{}', but that file does not exist. Please point CONFIG_PATH to a valid ai-config.yaml file.",
                config_path.display()
            ));
        }

        return Err(anyhow::anyhow!(
            "Cannot start LLML Docker because ai-config.yaml was not found at '{}'.\nIf you are running from the lala crate directory, make sure ai-config.yaml exists or set CONFIG_PATH to the correct path.",
            config_path.display()
        ));
    }

    fn docker_path(path: &std::path::Path) -> String {
        let mut s = path.to_string_lossy().to_string();
        if let Some(stripped) = s.strip_prefix(r"\\?\") {
            s = stripped.to_string();
        }
        if let Some(stripped) = s.strip_prefix("//?/") {
            s = stripped.to_string();
        }
        s.replace('\\', "/")
    }

    let models_mapping = format!("{}:/models:ro", docker_path(&models_path));
    println!("Mounting models directory: {}", models_path.to_str().unwrap_or("<invalid path>"));
    let config_mapping = format!("{}:/app/ai-config.yaml:ro", docker_path(&config_path));

    let output = Command::new("docker")
        .arg("run")
        .arg("-d")
        .arg("--rm")
        .arg("--name")
        .arg(LLML_CONTAINER_NAME)
        .arg("-p")
        .arg(&port_mapping)
        .arg("-v")
        .arg("lala-llml-data:/app/data")
        .arg("-v")
        .arg(&models_mapping)
        .arg("-v")
        .arg(&config_mapping)
        .arg("dipghoshraj/llml:0.0.2")
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
    ensure_docker_available()?;
    let port_mapping = format!("{port}:5432");
    let output = Command::new("docker")
        .arg("run")
        .arg("-d")
        .arg("--rm")        .arg("--name")
        .arg(POSTGRES_CONTAINER_NAME)        .arg("-p")
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

pub fn stop_docker_containers() -> anyhow::Result<()> {
    ensure_docker_available()?;
    let mut stopped_any = false;
    let mut errors = Vec::new();

    for container in [LLML_CONTAINER_NAME, POSTGRES_CONTAINER_NAME] {
        let output = Command::new("docker").arg("stop").arg(container).output()?;
        if output.status.success() {
            stopped_any = true;
            continue;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}{}", stdout, stderr);

        if combined.contains("No such container")
            || combined.contains("no such container")
            || combined.contains("is not running")
            || combined.contains("Error response from daemon") && combined.contains("No such container")
        {
            continue;
        }

        errors.push(format!("Failed to stop {}: {}", container, combined.trim()));
    }

    if temp_env_file().exists() {
        let _ = fs::remove_file(temp_env_file());
    }

    if !errors.is_empty() {
        return Err(anyhow::anyhow!(errors.join("\n")));
    }

    if stopped_any {
        println!("Stopped local LLML serve containers.");
    }

    Ok(())
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
