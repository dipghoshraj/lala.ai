/// Split `text` into overlapping character windows.
///
/// Edge cases:
///   - Empty text → empty Vec
///   - Text shorter than `chunk_size` → single-element Vec containing the full text
///   - `overlap >= chunk_size` → treated as `overlap = 0` (no overlap)
pub fn chunk(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() || chunk_size == 0 {
        return Vec::new();
    }

    let overlap = if overlap >= chunk_size { 0 } else { overlap };
    let step = chunk_size - overlap;
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    if len <= chunk_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < len {
        let end = (start + chunk_size).min(len);
        let chunk_text: String = chars[start..end].iter().collect();
        chunks.push(chunk_text);

        if end == len {
            break;
        }
        start += step;
    }

    chunks
}

