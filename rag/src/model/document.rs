// use postgres::Row;
use uuid::Uuid;

use crate::model::chrono_now;
#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub title: String,
    pub source: String,
    pub created_at: String,
    pub project_id: String,
}

impl Document {
    pub fn new(title: &str, source: &str, project_id: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            title: title.to_string(),
            source: source.to_string(),
            created_at: chrono_now(),
            project_id,
        }
    }

    pub fn get_by_source(source: &str) -> anyhow::Result<Option<Self>> {
        let db = crate::model::db();
        let mut client = db.client();
        let row = client.query_opt(
            crate::model::sql::SELECT_DOCUMENT_BY_SOURCE,
            &[&source],
        )?;
        if let Some(row) = row {
            Ok(Some(Self {
                id: row.get(0),
                title: row.get(1),
                source: row.get(2),
                created_at: row.get(3),
                project_id: row.get(4),
            }))
        } else {
            Ok(None)
        }
    }

    pub fn exist(source: &str) -> anyhow::Result<bool> {
        Ok(Self::get_by_source(source)?.is_some())
    }

    pub fn delete_by_source(source: &str) -> anyhow::Result<()> {
        let db = crate::model::db();
        let mut client = db.client();
        client.execute(
            crate::model::sql::DELETE_DOCUMENT_BY_SOURCE,
            &[&source],
        )?;
        Ok(())
    }

    pub fn insert(&self) -> anyhow::Result<()> {
        let db = crate::model::db();
        let mut client = db.client();
        client.execute(
            crate::model::sql::INSERT_DOCUMENT,
            &[&self.id, &self.title, &self.source, &self.created_at, &self.project_id],
        )?;
        Ok(())
    }

    pub fn count() -> anyhow::Result<usize> {
        let db = crate::model::db();
        let mut client = db.client();
        let row = client.query_one("SELECT COUNT(*) FROM documents", &[])?;
        Ok(row.get::<_, i64>(0) as usize)
    }

    pub fn count_by_project(project_id: &str) -> anyhow::Result<usize> {
        let db = crate::model::db();
        let mut client = db.client();
        let row = client.query_one(crate::model::sql::SELECT_DOCUMENT_COUNT_BY_PROJECT, &[&project_id])?;
        Ok(row.get::<_, i64>(0) as usize)
    }

    pub fn fetch_by_project(project_id: &str) -> anyhow::Result<Vec<Self>> {
        let db = crate::model::db();
        let mut client = db.client();
        let rows = client.query(crate::model::sql::SELECT_DOCUMENTS_BY_PROJECT, &[&project_id])?;
        Ok(rows
            .into_iter()
            .map(|row| Self {
                id: row.get(0),
                title: row.get(1),
                source: row.get(2),
                created_at: row.get(3),
                project_id: row.get(4),
            })
            .collect())
    }
}
