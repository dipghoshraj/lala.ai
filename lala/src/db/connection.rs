use sqlx::PgPool;

pub async fn init_pool() -> PgPool {
    PgPool::connect(&std::env::var("DATABASE_URL").unwrap())
        .await
        .unwrap()
}

pub async fn insert_chunk(pool: &PgPool, doc_id: &str, content: &str, embedding: Vec<f32>) {
    sqlx::query!(
        "INSERT INTO document_chunks (document_id, content, embedding, embedding_model)
         VALUES ($1, $2, $3, $4)",
        doc_id,
        content,
        &embedding,
        "bge-small"
    )
    .execute(pool)
    .await
    .unwrap();
}

pub async fn retrieve_chunks(pool: &PgPool, query_embedding: Vec<f32>, top_k: i64) -> Vec<String> {
    let rows = sqlx::query!(
        "SELECT content FROM document_chunks
         ORDER BY embedding <=> $1
         LIMIT $2",
        &query_embedding,
        top_k
    )
    .fetch_all(pool)
    .await
    .unwrap();

    rows.into_iter().map(|r| r.content).collect()
}

pub async fn add_memory(pool: &PgPool, content: &str, embedding: Vec<f32>) {
    sqlx::query!(
        "INSERT INTO memory (content, embedding, embedding_model)
         VALUES ($1, $2, $3)",
        content,
        &embedding,
        "bge-small"
    )
    .execute(pool)
    .await
    .unwrap();
}

pub async fn retrieve_memory(pool: &PgPool, query_embedding: Vec<f32>, top_k: i64) -> Vec<String> {
    let rows = sqlx::query!(
        "SELECT content FROM memory
         ORDER BY embedding <=> $1
         LIMIT $2",
        &query_embedding,
        top_k
    )
    .fetch_all(pool)
    .await
    .unwrap();

    rows.into_iter().map(|r| r.content).collect()
}