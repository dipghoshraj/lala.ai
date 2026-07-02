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
            &[&self.id, &self.name, &self.description, &self.created_at],
        )?;
        Ok(())
    }



}