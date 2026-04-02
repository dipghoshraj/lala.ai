# news_ingest.py

from webbrowser import Chrome

import requests
import xml.etree.ElementTree as ET
import hashlib
import re
import time
from bs4 import BeautifulSoup

# ----------- Configuration -------------
RSS_URL = "https://feeds.bbci.co.uk/news/rss.xml"
CORS_PROXY = "https://api.allorigins.win/raw?url="
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DELAY_BETWEEN_REQUESTS = 1  # in seconds

# ---------- Functions -----------------
def fetch_rss():
    """Fetch RSS feed and parse metadata."""
    response = requests.get(RSS_URL)
    print(f"Fetched RSS feed with status code: {response.status_code}")
    response.raise_for_status()

    time.sleep(2)  
    root = ET.fromstring(response.content)

    articles = []
    for item in root.findall(".//item"):
        article = {
            "title": item.find("title").text,
            "link": item.find("link").text,
            "pub_date": item.find("pubDate").text,
            "summary": item.find("description").text
        }
        articles.append(article)

    return articles

def _extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()

    raw_text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", raw_text)


def _extract_metadata(html):
    soup = BeautifulSoup(html, "html.parser")

    # authors (meta name=author, meta property=article:author, author tag)
    authors = set()
    for key in ["author", "article:author"]:
        for m in soup.find_all("meta", attrs={"name": re.compile(key, re.I)}):
            if m.get("content"):
                authors.add(m.get("content").strip())
        for m in soup.find_all("meta", attrs={"property": re.compile(key, re.I)}):
            if m.get("content"):
                authors.add(m.get("content").strip())

    author_tags = soup.find_all(attrs={"itemprop": re.compile("author", re.I)})
    for tag in author_tags:
        text = tag.get_text(strip=True)
        if text:
            authors.add(text)

    # publication date (OpenGraph, article:published_time, meta name=date)
    publish_date = None
    for attr in ["article:published_time", "og:published_time", "date"]:
        meta = soup.find("meta", attrs={"property": attr}) or soup.find("meta", attrs={"name": attr})
        if meta and meta.get("content"):
            publish_date = meta.get("content").strip()
            break

    # top image
    top_image = None
    og_image = soup.find("meta", attrs={"property": "og:image"})
    if og_image and og_image.get("content"):
        top_image = og_image.get("content").strip()
    else:
        img = soup.find("img")
        if img and img.get("src"):
            top_image = img.get("src").strip()

    return {
        "authors": list(authors),
        "publish_date": publish_date,
        "top_image": top_image,
    }


def fetch_full_article(url):
    """Download full article content using requests and plain HTML extraction.

    Tries browser-like headers, then falls back to a CORS proxy on 403.
    """
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": RSS_URL,
        "DNT": "1",
        "Connection": "keep-alive",
    }

    def _get_url(target):
        return session.get(target, headers=headers, timeout=20, allow_redirects=True)

    try:
        resp = _get_url(url)
        if resp.status_code == 403:
            print(f"403 received from {url}, trying proxy fallback")
            proxy_url = CORS_PROXY + requests.utils.requote_uri(url)
            resp = _get_url(proxy_url)

        resp.raise_for_status()

        html = resp.text
        text = _extract_text_from_html(html)
        meta = _extract_metadata(html)

        time.sleep(1)  # polite scraping
        return {
            "text": text,
            "authors": meta.get("authors", []),
            "publish_date": meta.get("publish_date"),
            "top_image": meta.get("top_image")
        }
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def generate_id(text):
    """Generate a unique ID from text (hash)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def build_news_dataset():
    """Fetch RSS, get full articles, chunk content, build dataset."""
    rss_articles = fetch_rss()
    enriched_articles = []

    for item in rss_articles:
        print(f"Processing: {item['title']}")
        full_content = fetch_full_article(item["link"])
        time.sleep(DELAY_BETWEEN_REQUESTS)  # polite scraping

        if full_content:
            print(f"Fetched full content for: {item['title']} (length: {full_content['text']})")
            chunks = chunk_text(full_content["text"])

            enriched_articles.append({
                "id": generate_id(item["link"]),
                "title": item["title"],
                "url": item["link"],
                "published": item["pub_date"],
                "summary": item["summary"],
                "content": full_content["text"],
                "authors": full_content["authors"],
                "image": full_content["top_image"],
                "source": "Yahoo News",
                "chunked_text": chunks
            })

    return enriched_articles

# ----------- Main --------------------
if __name__ == "__main__":
    news_data = build_news_dataset()
    print(f"Fetched {len(news_data)} articles.")
    
    # Example: print first article
    if news_data:
        first = news_data[0]
        print("\nTitle:", first["title"])
        print("URL:", first["url"])
        print("Chunks:", len(first["chunked_text"]))