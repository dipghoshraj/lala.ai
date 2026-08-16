use std::fs;
use std::path::Path;

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
        let ft = entry.file_type()?;
        if ft.is_dir() {
            collect_files(&entry.path(), out)?;
        } else if ft.is_file() {
            if let Some(path) = entry.path().to_str() {
                out.push(path.to_string());
            }
        }
    }
    Ok(())
}
