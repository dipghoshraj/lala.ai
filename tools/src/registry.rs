use std::collections::HashMap;
use crate::{Tool, ToolDescription};

/// Registry for managing and discovering tools
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        ToolRegistry {
            tools: HashMap::new(),
        }
    }

    /// Register a tool in the registry
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|b| b.as_ref() as &dyn Tool)
    }

    /// Get all tool descriptions
    pub fn all_descriptions(&self) -> Vec<ToolDescription> {
        self.tools
            .values()
            .map(|tool| tool.to_description())
            .collect()
    }

    /// Find candidate tools based on keyword substring matches (case-insensitive)
    pub fn keyword_candidates(&self, query: &str) -> Vec<String> {
        let query_lower = query.to_lowercase();
        self.tools
            .values()
            .filter_map(|tool| {
                let keywords = tool.keywords();
                if keywords.iter().any(|kw| query_lower.contains(&kw.to_lowercase())) {
                    Some(tool.name().to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get descriptions for candidate tool names
    pub fn descriptions_for(&self, tool_names: &[String]) -> Vec<ToolDescription> {
        tool_names
            .iter()
            .filter_map(|name| self.get(name).map(|tool| tool.to_description()))
            .collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
