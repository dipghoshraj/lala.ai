use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct DocumentChunk {
    pub id: String,
    pub document_id: String,
    pub index: i32,
    pub text: String,
    pub char_count: i32,
}

impl DocumentChunk {
    pub fn new(document_id: &str, index: usize, text: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            document_id: document_id.to_string(),
            index: index as i32,
            char_count: text.chars().count() as i32,
            text,
        }
    }
}