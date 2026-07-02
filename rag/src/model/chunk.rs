use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct DocumentChunk {
    pub id: String,
    pub document_id: String,
    pub index: i32,
    pub text: String,
    pub char_count: i32,
}

#[derive(Clone)]
pub struct ChunkRow {
    pub id: String,
    pub document_id: String,
    pub chunk_index: usize,
    pub chunk_text: String,
    pub score: f32,
    pub title: String,
    pub source: String,
}

impl DocumentChunk {
    pub fn new(document_id: &str, index: i32, text: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            document_id: document_id.to_string(),
            index,
            char_count: text.chars().count() as i32,
            text,
        }
    }

    pub fn insert(&self) -> anyhow::Result<()> {
        let db = crate::model::db();
        let mut client = db.client();
        client.execute(
            crate::model::sql::INSERT_CHUNK,
            &[ &self.id, &self.document_id, &self.index, &self.text, &self.char_count],
        )?;
        Ok(())
    }

    pub fn fetch_by_documents(query: &str, project_id: &str, limit: i64) -> anyhow::Result<Vec<ChunkRow>> {
        let db = crate::model::db();
        let mut client = db.client();
        let rows = client.query(
            crate::model::sql::SEARCH_CHUNKS,
            &[&query, &limit, &project_id],
        )?;
        let chunks = rows
            .into_iter()
            .map(|row| ChunkRow {
                id: row.get(0),
                document_id: row.get(1),
                chunk_index: row.get::<_, i32>(2) as usize,
                chunk_text: row.get(3),
                score: row.get(4),
                title: row.get(5),
                source: row.get(6),
                
            })
            .collect();
        Ok(chunks)
    }

    pub fn count() -> anyhow::Result<usize> {
        let db = crate::model::db();
        let mut client = db.client();
        let row = client.query_one("SELECT COUNT(*) FROM chunks", &[])?;
        Ok(row.get::<_, i64>(0) as usize)
    }
}