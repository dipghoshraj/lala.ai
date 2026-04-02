use crate::Tool;
use anyhow::{Result, anyhow, Context};
use serde_json::Value;
use std::path::Path;

/// Tool for reading file contents
pub struct FileReaderTool;

impl Tool for FileReaderTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read and return the contents of a file (absolute paths only)"
    }

    fn keywords(&self) -> &[&str] {
        &["read file", "file", "open file", "file contents", "read"]
    }

    fn execute(&self, input: Value) -> Result<String> {
        // Extract path from input
        let path = input
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing required field: path (must be a string)"))?;

        // Validate: must be absolute path
        let path_obj = Path::new(path);
        if !path_obj.is_absolute() {
            return Err(anyhow!(
                "Path must be absolute; got: {}",
                path
            ));
        }

        // Validate: reject parent directory traversal
        if path.contains("..") {
            return Err(anyhow!("Path traversal (..) not allowed: {}", path));
        }

        // Check file exists
        if !path_obj.exists() {
            return Err(anyhow!("File does not exist: {}", path));
        }

        // Check it's a file, not a directory
        if !path_obj.is_file() {
            return Err(anyhow!("Path is not a file: {}", path));
        }

        // Read and return as UTF-8
        std::fs::read_to_string(path_obj)
            .with_context(|| format!("Failed to read file: {}", path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn test_read_file_absolute_path() {
        let tool = FileReaderTool;
        
        // Create a temporary test file
        let temp_path = PathBuf::from(std::env::temp_dir()).join("lala_test_read.txt");
        fs::write(&temp_path, "test content").expect("write test file");

        let input = serde_json::json!({
            "path": temp_path.to_str().expect("convert path to string")
        });

        let result = tool.execute(input).expect("should read file");
        assert_eq!(result, "test content");

        // Cleanup
        fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_read_file_relative_path_rejected() {
        let tool = FileReaderTool;
        let input = serde_json::json!({ "path": "relative/path.txt" });
        let result = tool.execute(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be absolute"));
    }

    #[test]
    fn test_read_file_traversal_rejected() {
        let tool = FileReaderTool;
        let input = serde_json::json!({ "path": "/some/path/../../../etc/passwd" });
        let result = tool.execute(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not allowed"));
    }

    #[test]
    fn test_read_file_nonexistent() {
        let tool = FileReaderTool;
        let input = serde_json::json!({ "path": "/nonexistent/path/file.txt" });
        let result = tool.execute(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }
}
