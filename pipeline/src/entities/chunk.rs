use sea_orm::entity::prelude::*;
use uuid::Uuid;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "document_chunks")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: Uuid,
    pub document_id: Uuid,
    pub chunk_index: i32,
    pub chunk_text: String,
    pub embedding: Option<Vec<f32>>,
    pub token_count: i32,
    pub created_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter)]
pub enum Relation {
    Document,
}

impl RelationTrait for Relation {
    fn def(&self) -> RelationDef {
        match self {
            Self::Document => Entity::belongs_to(super::documents::Entity)
                .from(Column::DocumentId)
                .to(super::documents::Column::Id)
                .into(),
        }
    }
}

impl ActiveModelBehavior for ActiveModel {}