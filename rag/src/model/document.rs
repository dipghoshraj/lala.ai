// use postgres::Row;
use uuid::Uuid;

use crate::model::chrono_now;
#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub title: String,
    pub source: String,
    pub created_at: String,
}

impl Document {
    pub fn new(title: &str, source: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            title: title.to_string(),
            source: source.to_string(),
            created_at: chrono_now(),
        }
    }

    pub fn exist(source: &str) -> anyhow::Result<bool> {
        let db = crate::model::db();
        let mut client = db.client();
        let row = client.query_one(
            crate::model::sql::DOCUMENT_EXISTS,
            &[&source],
        )?;
        Ok(row.get::<_, i64>(0) > 0)
    }

    pub fn insert(&self) -> anyhow::Result<()> {
        let db = crate::model::db();
        let mut client = db.client();
        client.execute(
            crate::model::sql::INSERT_DOCUMENT,
            &[&self.id, &self.title, &self.source, &self.created_at],
        )?;
        Ok(())
    }

    
}
