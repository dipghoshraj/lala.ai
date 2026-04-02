use crate::Tool;
use anyhow::Result;
use serde_json::Value;

/// Tool for getting the current time and date
pub struct GetCurrentTimeTool;

impl Tool for GetCurrentTimeTool {
    fn name(&self) -> &str {
        "get_current_time"
    }

    fn description(&self) -> &str {
        "Get the current time and date in ISO8601 format"
    }

    fn keywords(&self) -> &[&str] {
        &["time", "date", "clock", "now", "today"]
    }

    fn execute(&self, _input: Value) -> Result<String> {
        let now = chrono::Utc::now();
        Ok(now.to_rfc3339())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_current_time() {
        let tool = GetCurrentTimeTool;
        let result = tool.execute(Value::Null).expect("should get time");
        // Should be a valid RFC3339 timestamp
        assert!(result.len() > 0);
        assert!(result.contains("T"));
        assert!(result.contains("Z"));
    }
}
