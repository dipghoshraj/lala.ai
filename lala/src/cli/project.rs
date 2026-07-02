use crate::cli::display;
use rag::RagStore;
use rag::model::project::Project;

pub fn handle_project_command(store: &RagStore, args: &str) {
    let mut parts = args.splitn(2, ' ');
    let subcmd = parts.next().unwrap_or("").trim();
    let rest = parts.next().unwrap_or("").trim();

    match subcmd {
        "create" => create_project(store, rest),
        "select" => select_project(store, rest),
        "deselect" => deselect_project(store),
        "list" => list_projects(),
        "current" => current_project(store),
        "" => print_project_help(),
        _ => {
            display::error(&format!("Unknown /project command: {subcmd}"));
            println!();
            print_project_help();
        }
    }
}

fn print_project_help() {
    println!("Usage: /project <command>");
    println!();
    println!("  /project create --name <name> [--description <description>]    Create a new project and select it");
    println!("      Example: /project create --name lala.ai --description \"A local only AI assistant with context and built-in RAG\"");
    println!("  /project select <name-or-id>                                 Select an existing project");
    println!("  /project deselect                                            Deselect the current project");
    println!("  /project list                                                List available projects");
    println!("  /project current                                             Show the selected project");
    println!();
}

fn create_project(store: &RagStore, args: &str) {
    match parse_create_args(args) {
        Ok((name, description)) => {
            let project = Project::new(&name, &description);
            match project.insert() {
                Ok(()) => {
                    store.select_project(&project.id);
                    display::success(&format!("Created and selected project: {} ({})", project.name, project.id));
                    println!();
                }
                Err(e) => {
                    display::error(&format!("Failed to create project: {e}"));
                    println!();
                }
            }
        }
        Err(err) => {
            display::error(&format!("{err}"));
            println!("Usage: /project create --name <name> [--description <description>]\n");
        }
    }
}

fn parse_create_args(args: &str) -> Result<(String, String), String> {
    let tokens = tokenize_args(args);
    if tokens.is_empty() {
        return Err("Missing arguments for /project create.".to_string());
    }

    let mut name = None;
    let mut description = None;
    let mut i = 0;

    while i < tokens.len() {
        match tokens[i].as_str() {
            "--name" | "-n" => {
                i += 1;
                if i >= tokens.len() {
                    return Err("Expected a value after --name.".to_string());
                }
                name = Some(tokens[i].clone());
            }
            "--description" | "-d" => {
                i += 1;
                if i >= tokens.len() {
                    return Err("Expected a value after --description.".to_string());
                }
                description = Some(tokens[i].clone());
            }
            token if token.starts_with("--name=") => {
                name = Some(token[7..].to_string());
            }
            token if token.starts_with("--description=") => {
                description = Some(token[14..].to_string());
            }
            token if token.starts_with("-n=") => {
                name = Some(token[3..].to_string());
            }
            token if token.starts_with("-d=") => {
                description = Some(token[3..].to_string());
            }
            other => {
                return Err(format!("Unknown argument: {other}"));
            }
        }
        i += 1;
    }

    let name = name.ok_or_else(|| "Missing required --name argument.".to_string())?;
    let description = description.unwrap_or_default();
    Ok((name, description))
}

fn tokenize_args(args: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut quote_char = '\0';

    for ch in args.chars() {
        match ch {
            ' ' | '\t' if !in_quotes => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            }
            '"' | '\'' => {
                if in_quotes {
                    if ch == quote_char {
                        in_quotes = false;
                        quote_char = '\0';
                    } else {
                        current.push(ch);
                    }
                } else {
                    in_quotes = true;
                    quote_char = ch;
                }
            }
            _ => current.push(ch),
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

fn select_project(store: &RagStore, args: &str) {
    if args.is_empty() {
        display::error("Usage: /project select <name-or-id>");
        println!();
        return;
    }

    match Project::find_by_id_or_name(args) {
        Ok(Some(project)) => {
            store.select_project(&project.id);
            display::success(&format!("Selected project: {} ({})", project.name, project.id));
            println!();
        }
        Ok(None) => {
            display::warn(&format!("No project found for: {args}"));
            println!();
        }
        Err(e) => {
            display::error(&format!("Failed to select project: {e}"));
            println!();
        }
    }
}

fn deselect_project(store: &RagStore) {
    if store.current_project_id().is_some() {
        store.deselect_project();
        display::success("Project deselected. LLM-only mode enabled.");
        println!();
    } else {
        display::warn("No project is currently selected.");
        println!();
    }
}

fn list_projects() {
    match Project::fetch_all() {
        Ok(projects) if projects.is_empty() => {
            display::warn("No projects exist yet. Create one with /project create <name>.");
            println!();
        }
        Ok(projects) => {
            println!();
            let sep = "─".repeat(display::SECTION_WIDTH);
            println!("{}{}{}", display::DIM, sep, display::RESET);
            println!("  {}Projects{}", display::BOLD, display::RESET);
            println!("{}{}{}", display::DIM, sep, display::RESET);
            for project in projects {
                println!("  {}{}{} ({})", display::BOLD_GREEN, project.name, display::RESET, project.id);
                if !project.description.is_empty() {
                    println!("    {}Description:{} {}", display::DIM, display::RESET, project.description);
                }
                println!();
            }
            println!("{}{}{}", display::DIM, sep, display::RESET);
            println!();
        }
        Err(e) => {
            display::error(&format!("Failed to list projects: {e}"));
            println!();
        }
    }
}

fn current_project(store: &RagStore) {
    match store.current_project_id() {
        Some(project_id) => match Project::fetch_by_id(&project_id) {
            Ok(Some(project)) => {
                println!();
                display::success(&format!("Current project: {} ({})", project.name, project.id));
                if !project.description.is_empty() {
                    println!("  Description: {}", project.description);
                }
                println!();
            }
            Ok(None) => {
                display::warn("The selected project ID does not exist in the database.");
                println!();
            }
            Err(e) => {
                display::error(&format!("Failed to load current project: {e}"));
                println!();
            }
        },
        None => {
            display::warn("No project is currently selected.");
            println!();
        }
    }
}
