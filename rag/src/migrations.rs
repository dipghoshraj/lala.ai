pub const MIGRATIONS: &[(&str, &str)] = &[
    (
        "001_initial_schema.sql",
        include_str!("../../migrations/001_initial_schema.sql"),
    ),
    (
        "002_pgvector.sql",
        include_str!("../../migrations/002_pgvector.sql"),
    ),
    (
        "003_projects_schema.sql",
        include_str!("../../migrations/003_projects_schema.sql"),
    ),
    (
        "004_project_dir_add.sql",
        include_str!("../../migrations/004_project_dir_add.sql"),
    ),
];
