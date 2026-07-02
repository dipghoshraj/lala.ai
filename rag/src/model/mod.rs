pub mod document;
pub mod chunk;
mod sql;
pub mod memory;
pub mod project;

use std::sync::{Mutex, MutexGuard};
use postgres::{Client, NoTls};

use std::sync::OnceLock;

static DB: OnceLock<RagDB> = OnceLock::new();

pub struct RagDB {
    pub client: Mutex<Client>,
}

impl RagDB {

    pub fn new(url: &str) -> anyhow::Result<Self> {
        Ok(Self {
            client: Mutex::new(Client::connect(url, NoTls)?),
        })
    }

    pub fn client(&self) -> MutexGuard<'_, Client> {
        self.client.lock().unwrap()
    }

    pub fn execute(&self, query: &str, params: &[&(dyn postgres::types::ToSql + Sync)]) -> anyhow::Result<u64> {
        let mut client = self.client();
        let rows_affected = client.execute(query, params)?;
        Ok(rows_affected)
    }
}

pub fn init_db(url: &str) -> anyhow::Result<()> {
    DB.set(RagDB::new(url)?).map_err(|_| anyhow::anyhow!("DB already initialized"))?;
    Ok(())
}

pub fn db() -> &'static RagDB {
    DB.get().expect("DB not initialized")
}

pub fn chrono_now() -> String {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs();
    format!("{secs}")
}


/// Placeholder memory block builder — stores chunk text as fallback.
/// Real extraction is done via LLM in the lala CLI ingest pipeline.
///
/// Returns (facts, capabilities, constraints).
pub fn build_memory_block(chunk: &str) -> (String, String, String) {
    let chunk_text = chunk.to_string();
    (chunk_text.clone(), chunk_text.clone(), chunk_text)
}