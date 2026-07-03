use uuid::Uuid;
use crate::model::chrono_now;

#[derive(Debug, Clone)]
pub struct Project {
    pub id: String,
    pub name: String,
    pub description: String,
    pub created_at: String,
}




impl Project {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            description: description.to_string(),
            created_at: chrono_now(),
        }
    }

    pub fn insert(&self) -> anyhow::Result<()> {
        let db = crate::model::db();
        let mut client = db.client();
        client.execute(
            crate::model::sql::INSERT_PROJECT,
            &[&self.id, &self.name, &self.description],
        )?;
        Ok(())
    }

    pub fn fetch_all() -> anyhow::Result<Vec<Self>> {
        let db = crate::model::db();
        let mut client = db.client();
        let rows = client.query(crate::model::sql::SELECT_PROJECTS, &[])?;
        Ok(rows
            .into_iter()
            .map(|row| Self {
                id: row.get(0),
                name: row.get(1),
                description: row.get(2),
                created_at: row.get(3),
            })
            .collect())
    }

    pub fn fetch_by_id(id: &str) -> anyhow::Result<Option<Self>> {
        let db = crate::model::db();
        let mut client = db.client();
        let row = client.query_opt(crate::model::sql::SELECT_PROJECT_BY_ID, &[&id])?;
        Ok(row.map(|row| Self {
            id: row.get(0),
            name: row.get(1),
            description: row.get(2),
            created_at: row.get(3),
        }))
    }

    pub fn fetch_by_name(name: &str) -> anyhow::Result<Vec<Self>> {
        let db = crate::model::db();
        let mut client = db.client();
        let rows = client.query(crate::model::sql::SELECT_PROJECT_BY_NAME, &[&name])?;
        Ok(rows
            .into_iter()
            .map(|row| Self {
                id: row.get(0),
                name: row.get(1),
                description: row.get(2),
                created_at: row.get(3),
            })
            .collect())
    }

    pub fn find_by_id_or_name(value: &str) -> anyhow::Result<Option<Self>> {
        if let Some(project) = Self::fetch_by_id(value)? {
            return Ok(Some(project));
        }
        let mut matches = Self::fetch_by_name(value)?;
        Ok(matches.pop())
    }
}
