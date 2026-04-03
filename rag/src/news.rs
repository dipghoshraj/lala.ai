/// News ingest module: fetch RSS feeds, parse articles, ingest into RAG.

use anyhow::{Result, bail};
use reqwest::blocking::Client;
use rss::Channel;
use std::time::Duration;
use std::thread;

use crate::RagStore;

/// Fetch an RSS feed, download & parse each article, ingest into RAG store.
///
/// # Arguments
/// * `store` - RagStore instance
/// * `rss_url` - URL to the RSS feed
/// * `delay_ms` - milliseconds to wait between fetches (polite scraping)
///
/// # Returns
/// * `(ingested, skipped, failed)` counts
pub fn ingest_news_feed(
    store: &RagStore,
    rss_url: &str,
    delay_ms: u64,
) -> Result<(usize, usize, usize)> {
    let client = Client::builder()
        .timeout(Duration::from_secs(20))
        .build()?;

    // Fetch RSS
    eprintln!("Fetching RSS feed: {}", rss_url);
    let resp = client.get(rss_url).send()?;
    let feed_content = resp.text()?;
    let channel = Channel::read_from(feed_content.as_bytes())?;

    let mut ingested = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;

    for (idx, item) in channel.items.iter().enumerate() {
        let title = item.title().unwrap_or("(no title)");
        let link = match item.link() {
            Some(l) => l,
            None => {
                eprintln!("[{}] Skipped (no link): {}", idx + 1, title);
                skipped += 1;
                continue;
            }
        };

        eprintln!("[{}] Processing: {}", idx + 1, title);

        match fetch_and_ingest_article(&client, store, title, link) {
            Ok(_) => {
                eprintln!("  ✓ Ingested");
                ingested += 1;
            }
            Err(e) => {
                eprintln!("  ✗ Failed: {}", e);
                failed += 1;
            }
        }

        thread::sleep(Duration::from_millis(delay_ms));
    }

    Ok((ingested, skipped, failed))
}

/// Fetch a single article, extract text + metadata, ingest into RAG.
fn fetch_and_ingest_article(
    client: &Client,
    store: &RagStore,
    title: &str,
    url: &str,
) -> Result<()> {
    let user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36";

    let html = match client.get(url)
        .header("User-Agent", user_agent)
        .send()
    {
        Ok(resp) if resp.status().as_u16() == 403 => {
            // Fallback: try via CORS proxy
            eprintln!("    403 from {}, trying proxy...", url);
            let proxy_url = format!("https://api.allorigins.win/raw?url={}", 
                urlencoding::encode(url));
            let proxy_resp = client.get(&proxy_url).send()?;
            proxy_resp.text()?
        }
        Ok(resp) => resp.text()?,
        Err(e) => bail!("Failed to fetch {}: {}", url, e),
    };

    let text = extract_text_from_html(&html);
    if text.is_empty() {
        bail!("Empty text extracted from {}", url);
    }

    store.ingest(title, url, &text)?;
    Ok(())
}

/// Extract clean text from HTML using a simple approach.
fn extract_text_from_html(html: &str) -> String {
    // Remove script and style tags using regex
    let text = regex::Regex::new(r"(?is)<script[^>]*>.*?</script>")
        .unwrap()
        .replace_all(html, "");
    let text = regex::Regex::new(r"(?is)<style[^>]*>.*?</style>")
        .unwrap()
        .replace_all(&text, "");

    // Remove HTML tags
    let text = regex::Regex::new(r"<[^>]+>")
        .unwrap()
        .replace_all(&text, " ");

    // Normalize whitespace
    let text = regex::Regex::new(r"\s+")
        .unwrap()
        .replace_all(&text, " ");

    text.trim().to_string()
}
