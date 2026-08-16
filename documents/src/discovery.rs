use std::fs;
use std::path::Path;

const SUPPORTED_EXTENSIONS: &[&str] = &[
    "pdf", "md", "txt"
];

// "rs", "py", "js", "ts", "go", "java", "json", "yaml", "yml", "toml", "xml", "html", "css", "sql",

pub fn scan_directory(dir: &str) -> anyhow::Result<Vec<String>> {
    let path = Path::new(dir);
    if !path.exists() {
        fs::create_dir_all(path)?;
        return Ok(Vec::new());
    }
    if !path.is_dir() {
        anyhow::bail!("{dir} exists but is not a directory");
    }

    let mut files = Vec::new();
    collect_files(path, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_files(dir: &Path, out: &mut Vec<String>) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let ft = entry.file_type()?;
        if ft.is_dir() {
            collect_files(&path, out)?;
        } else if ft.is_file() && is_supported_file(&path) {
            if let Some(path) = path.to_str() {
                out.push(path.to_string());
            }
        }
    }
    Ok(())
}

fn is_supported_file(path: &Path) -> bool {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension) => SUPPORTED_EXTENSIONS
            .iter()
            .any(|supported| extension.eq_ignore_ascii_case(supported)),
        None => true,
    }
}
