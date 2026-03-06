use sea_orm::entity::prelude::*;
use uuid::Uuid;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "answers")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: Uuid,
    pub query_id: Uuid,
    pub message_id: Option<Uuid>,
    pub answer_text: Option<String>,
    pub created_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter)]
pub enum Relation {
    Query,
    Message,
}

impl RelationTrait for Relation {
    fn def(&self) -> RelationDef {
        match self {
            Self::Query => Entity::belongs_to(super::queries::Entity)
                .from(Column::QueryId)
                .to(super::queries::Column::Id)
                .into(),
            Self::Message => Entity::belongs_to(super::messages::Entity)
                .from(Column::MessageId)
                .to(super::messages::Column::Id)
                .into(),
        }
    }
}

impl ActiveModelBehavior for ActiveModel {}