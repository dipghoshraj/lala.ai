/// migrate.rs — simple SQL migration runner for PostgreSQL.
///
/// Reads `*.sql` files from `migrations_dir` in lexicographic order and
/// applies each one that hasn't been recorded in `schema_migrations`.
/// Each file is applied atomically inside a transaction.
///
/// Idempotent: files already recorded in `schema_migrations` are skipped.

use anyhow::{Context, Result};
use postgres::Client;

/// Apply all embedded migrations to the connected client.
///
/// Returns the list of migration versions that were applied this run.
pub fn run_migrations(client: &mut Client) -> Result<Vec<String>> {
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

    let mut applied_this_run = Vec::new();

    for (version, sql) in crate::migrations::MIGRATIONS {
        let version = version.to_string();
        if applied.contains(&version) {
            continue;
        }

        let mut tx = client.transaction().context("starting migration transaction")?;
        tx.batch_execute(sql)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_migrations_are_present() {
        assert!(!crate::migrations::MIGRATIONS.is_empty(), "No embedded migrations found");
        let versions: Vec<_> = crate::migrations::MIGRATIONS
            .iter()
            .map(|(version, _)| *version)
            .collect();
        let mut sorted = versions.clone();
        sorted.sort();
        assert_eq!(versions, sorted, "Embedded migrations are not in lexicographic order");
        assert_eq!(versions.len(), versions.iter().collect::<std::collections::HashSet<_>>().len(), "Duplicate migration versions found");
    }
}
