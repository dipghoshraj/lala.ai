use serde_json::Value;
use anyhow::Result;
use serde::{Serialize, Deserialize};

/// Description of a tool for discovery and filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescription {
    pub name: String,
    pub description: String,
    pub keywords: Vec<String>,
}

/// Trait for executable tools
pub trait Tool: Send + Sync {
    /// Tool name (unique identifier)
    fn name(&self) -> &str;

    /// Human-readable description
    fn description(&self) -> &str;

    /// Keywords for pre-filtering (e.g., "time", "date", "search")
    fn keywords(&self) -> &[&str];

    /// Execute the tool with JSON input
    fn execute(&self, input: Value) -> Result<String>;

    /// Get tool description for discovery
    fn to_description(&self) -> ToolDescription {
        ToolDescription {
            name: self.name().to_string(),
            description: self.description().to_string(),
            keywords: self.keywords().iter().map(|k| k.to_string()).collect(),
        }
    }
}

pub mod registry;
pub mod builtin;

pub use registry::ToolRegistry;
pub use builtin::{GetCurrentTimeTool, FileReaderTool};
