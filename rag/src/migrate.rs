/// migrate.rs — simple SQL migration runner for PostgreSQL.
///
/// Reads `*.sql` files from `migrations_dir` in lexicographic order and
/// applies each one that hasn't been recorded in `schema_migrations`.
/// Each file is applied atomically inside a transaction.
///
/// Idempotent: files already recorded in `schema_migrations` are skipped.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use postgres::Client;

/// Apply all pending migrations in `migrations_dir` to the connected client.
///
/// Returns the list of migration versions that were applied this run.
pub fn run_migrations(client: &mut Client, migrations_dir: &str) -> Result<Vec<String>> {
    // Ensure the tracking table exists before we query it.
    client.batch_execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (
             version    TEXT        PRIMARY KEY,
             applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
         )",
    )?;

    // Collect already-applied versions.
    let applied: Vec<String> = client
        .query("SELECT version FROM schema_migrations ORDER BY version", &[])
        .context("querying schema_migrations")?
        .into_iter()
        .map(|row| row.get::<_, String>(0))
        .collect();

    // Discover migration files.
    let dir = Path::new(migrations_dir);
    if !dir.exists() {
        return Ok(vec![]);
    }

    let mut files: Vec<_> = fs::read_dir(dir)
        .with_context(|| format!("reading migrations dir: {migrations_dir}"))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()?.to_str()? == "sql" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    // Apply in lexicographic order (001_, 002_, …).
    files.sort();

    let mut applied_this_run = Vec::new();

    for file in &files {
        let version = file
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();

        if applied.contains(&version) {
            continue;
        }

        let sql = fs::read_to_string(file)
            .with_context(|| format!("reading migration file: {}", file.display()))?;

        // Apply inside a transaction for atomicity.
        let mut tx = client.transaction().context("starting migration transaction")?;
        tx.batch_execute(&sql)
            .with_context(|| format!("applying migration {version}"))?;
        tx.execute(
            "INSERT INTO schema_migrations (version) VALUES ($1)
             ON CONFLICT (version) DO NOTHING",
            &[&version],
        )
        .context("recording migration")?;
        tx.commit().context("committing migration")?;

        eprintln!("[migrate] applied {version}");
        applied_this_run.push(version);
    }

    Ok(applied_this_run)
}
