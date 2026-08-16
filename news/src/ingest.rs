use anyhow::{Result, bail};
use reqwest::blocking::Client;
use rss::Channel;
use std::thread;
use std::time::Duration;

use rag::RagStore;

use crate::types::ArticleIngestStatus;

pub fn ingest_news_feed(
    store: &RagStore,
    rss_url: &str,
    delay_ms: u64,
) -> Result<(usize, usize, usize)> {
    ingest_news_feed_with_progress(store, rss_url, delay_ms, |_title, _status| {} )
}

pub fn ingest_news_feed_with_progress<F>(
    store: &RagStore,
    rss_url: &str,
    delay_ms: u64,
    mut on_article: F,
) -> Result<(usize, usize, usize)>
where
    F: FnMut(&str, &ArticleIngestStatus),
{
    let client = Client::builder()
        .timeout(Duration::from_secs(20))
        .build()?;

    let resp = client.get(rss_url).send()?;
    let feed_content = resp.text()?;
    let channel = Channel::read_from(feed_content.as_bytes())?;

    let mut ingested = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;

    for item in channel.items.iter() {
        let title = item.title().unwrap_or("(no title)");
        let link = match item.link() {
            Some(l) => l,
            None => {
                on_article(title, &ArticleIngestStatus::Skipped("no link"));
                skipped += 1;
                continue;
            }
        };

        let status = match fetch_and_ingest_article(&client, store, title, link) {
            Ok(_) => ArticleIngestStatus::Ingested,
            Err(e) => ArticleIngestStatus::Failed(e.to_string()),
        };

        on_article(title, &status);
        match &status {
            ArticleIngestStatus::Ingested => ingested += 1,
            ArticleIngestStatus::Skipped(_) => skipped += 1,
            ArticleIngestStatus::Failed(_) => failed += 1,
        }

        thread::sleep(Duration::from_millis(delay_ms));
    }

    Ok((ingested, skipped, failed))
}

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
            let proxy_url = format!(
                "https://api.allorigins.win/raw?url={}",
                urlencoding::encode(url)
            );
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

fn extract_text_from_html(html: &str) -> String {
    let text = regex::Regex::new(r"(?is)<script[^>]*>.*?</script>")
        .unwrap()
        .replace_all(html, "");
    let text = regex::Regex::new(r"(?is)<style[^>]*>.*?</style>")
        .unwrap()
        .replace_all(&text, "");

    let text = regex::Regex::new(r"<[^>]+>")
        .unwrap()
        .replace_all(&text, " ");

    let text = regex::Regex::new(r"\s+")
        .unwrap()
        .replace_all(&text, " ");

    text.trim().to_string()
}
