
create Table if NOT exists projects (
    id          TEXT        PRIMARY KEY,
    name        TEXT        NOT NULL,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

Alter Table documents Add Column project_id TEXT REFERENCES projects (id) ON DELETE CASCADE;