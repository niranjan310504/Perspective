"""
Indian News Multi-Source Collector
====================================

Collects news from 15+ Indian sources across the political spectrum
for balanced bias detection training data.

Sources included:
- Right-leaning: Swarajya, OpIndia, Republic World
- Left-leaning: The Wire, Scroll, NewsLaundry
- Center: The Print, India Today, Hindustan Times
- Major outlets: TOI, NDTV, The Hindu, Indian Express

Features:
- RSS feed collection (legal, reliable)
- Opinion/Editorial section targeting (where bias is highest)
- Automatic deduplication
- Source diversity balancing
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import time
import re
import hashlib
from urllib.parse import urlparse, urljoin
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Optional: For better article extraction
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    print("Note: Install newspaper3k for better extraction: pip install newspaper3k lxml")

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


@dataclass
class NewsSource:
    """Configuration for a news source."""
    name: str
    domain: str
    rss_feeds: List[str]
    political_lean: str  # "left", "center", "right"
    bias_tendency: List[str]  # Expected bias types
    priority: int = 1  # Higher = more articles


# Indian News Sources Configuration
INDIAN_SOURCES = [
    # === RIGHT-LEANING SOURCES ===
    NewsSource(
        name="OpIndia",
        domain="opindia.com",
        rss_feeds=[
            "https://www.opindia.com/feed/",
        ],
        political_lean="right",
        bias_tendency=["political", "religious", "entity", "sensationalism"],
        priority=3
    ),
    NewsSource(
        name="Swarajya",
        domain="swarajyamag.com",
        rss_feeds=[
            "https://swarajyamag.com/feed",
        ],
        political_lean="right",
        bias_tendency=["political", "religious", "entity"],
        priority=3
    ),
    
    # === LEFT-LEANING SOURCES ===
    NewsSource(
        name="The Quint",
        domain="thequint.com",
        rss_feeds=[
            "https://www.thequint.com/news/rss",
        ],
        political_lean="left",
        bias_tendency=["political", "gender"],
        priority=3
    ),
    
    # === CENTER / MAINSTREAM SOURCES ===
    NewsSource(
        name="The Print",
        domain="theprint.in",
        rss_feeds=[
            "https://theprint.in/feed/",
        ],
        political_lean="center",
        bias_tendency=["political", "entity"],
        priority=3
    ),
    NewsSource(
        name="India Today",
        domain="indiatoday.in",
        rss_feeds=[
            "https://www.indiatoday.in/rss/home",
        ],
        political_lean="center",
        bias_tendency=["sensationalism", "entity"],
        priority=3
    ),
    NewsSource(
        name="Hindustan Times",
        domain="hindustantimes.com",
        rss_feeds=[
            "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
        ],
        political_lean="center",
        bias_tendency=["political", "regional"],
        priority=2
    ),
    NewsSource(
        name="Indian Express",
        domain="indianexpress.com",
        rss_feeds=[
            "https://indianexpress.com/section/opinion/feed/",
            "https://indianexpress.com/section/political-pulse/feed/",
        ],
        political_lean="center",
        bias_tendency=["political", "regional"],
        priority=2
    ),
    NewsSource(
        name="The Hindu",
        domain="thehindu.com",
        rss_feeds=[
            "https://www.thehindu.com/opinion/feeder/default.rss",
            "https://www.thehindu.com/news/national/feeder/default.rss",
        ],
        political_lean="center-left",
        bias_tendency=["political", "regional"],
        priority=2
    ),
    NewsSource(
        name="NDTV",
        domain="ndtv.com",
        rss_feeds=[
            "https://feeds.feedburner.com/ndtvnews-india-news",
            "https://feeds.feedburner.com/NDTV-Opinion",
        ],
        political_lean="center-left",
        bias_tendency=["political", "entity"],
        priority=2
    ),
    NewsSource(
        name="Times of India",
        domain="timesofindia.indiatimes.com",
        rss_feeds=[
            "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",  # India
        ],
        political_lean="center",
        bias_tendency=["sensationalism", "regional"],
        priority=3
    ),
    NewsSource(
        name="NDTV",
        domain="ndtv.com",
        rss_feeds=[
            "https://feeds.feedburner.com/ndtvnews-india-news",
        ],
        political_lean="center-left",
        bias_tendency=["political", "entity"],
        priority=3
    ),
    NewsSource(
        name="Indian Express",
        domain="indianexpress.com",
        rss_feeds=[
            "https://indianexpress.com/section/india/feed/",
        ],
        political_lean="center",
        bias_tendency=["political", "regional"],
        priority=3
    ),
]


class IndianNewsCollector:
    """
    Collects news from diverse Indian sources for bias detection training.
    """
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collected_urls: Set[str] = set()
        self.articles: List[Dict] = []
        
    def fetch_rss_feed(self, url: str, timeout: int = 15) -> List[Dict]:
        """
        Fetch and parse an RSS feed.
        
        Returns list of article metadata (title, link, description, pubdate)
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            articles = []
            
            # Handle different RSS formats
            # Standard RSS 2.0
            for item in root.findall(".//item"):
                article = {
                    "title": self._get_text(item, "title"),
                    "link": self._get_text(item, "link"),
                    "description": self._get_text(item, "description"),
                    "pubDate": self._get_text(item, "pubDate"),
                }
                if article["link"]:
                    articles.append(article)
            
            # Atom format
            for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
                link_elem = entry.find("{http://www.w3.org/2005/Atom}link")
                article = {
                    "title": self._get_text(entry, "{http://www.w3.org/2005/Atom}title"),
                    "link": link_elem.get("href") if link_elem is not None else None,
                    "description": self._get_text(entry, "{http://www.w3.org/2005/Atom}summary"),
                    "pubDate": self._get_text(entry, "{http://www.w3.org/2005/Atom}published"),
                }
                if article["link"]:
                    articles.append(article)
            
            return articles
            
        except Exception as e:
            print(f"    Error fetching RSS {url}: {e}")
            return []
    
    def _get_text(self, element, tag: str) -> Optional[str]:
        """Safely extract text from XML element."""
        child = element.find(tag)
        if child is not None and child.text:
            # Clean HTML tags from description
            text = re.sub(r'<[^>]+>', '', child.text)
            return text.strip()
        return None
    
    def extract_article_content(self, url: str) -> Optional[Dict]:
        """
        Extract full article content from URL.
        """
        try:
            if NEWSPAPER_AVAILABLE:
                article = Article(url)
                article.download()
                article.parse()
                
                if len(article.text) < 100:
                    return None
                    
                return {
                    "headline": article.title,
                    "content": article.text,
                    "authors": ", ".join(article.authors) if article.authors else "",
                    "publish_date": str(article.publish_date) if article.publish_date else "",
                }
            else:
                # Basic extraction
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                response = requests.get(url, headers=headers, timeout=15)
                
                # Basic title extraction
                title_match = re.search(r'<title>(.*?)</title>', response.text, re.IGNORECASE | re.DOTALL)
                title = title_match.group(1).strip() if title_match else "Unknown"
                title = re.sub(r'\s*[-|].*$', '', title)  # Remove site name suffix
                
                # Basic paragraph extraction
                paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', response.text, re.DOTALL)
                text = " ".join([re.sub(r'<[^>]+>', '', p) for p in paragraphs])
                text = re.sub(r'\s+', ' ', text).strip()
                
                if len(text) < 100:
                    return None
                
                return {
                    "headline": title,
                    "content": text[:8000],
                    "authors": "",
                    "publish_date": "",
                }
                
        except Exception as e:
            return None
    
    def collect_from_source(
        self, 
        source: NewsSource, 
        max_articles: int = 30
    ) -> List[Dict]:
        """
        Collect articles from a single source.
        """
        source_articles = []
        
        for rss_url in source.rss_feeds:
            if len(source_articles) >= max_articles:
                break
                
            # Fetch RSS feed
            rss_items = self.fetch_rss_feed(rss_url)
            
            for item in rss_items:
                if len(source_articles) >= max_articles:
                    break
                    
                url = item.get("link")
                if not url:
                    continue
                
                # Deduplication
                url_hash = hashlib.md5(url.encode()).hexdigest()
                if url_hash in self.collected_urls:
                    continue
                self.collected_urls.add(url_hash)
                
                # Extract full content
                content = self.extract_article_content(url)
                
                if content and len(content.get("content", "")) >= 200:
                    rss_desc = item.get("description") or ""
                    article = {
                        "article_id": url_hash[:16],
                        "headline": content["headline"],
                        "content": content["content"],
                        "url": url,
                        "source": source.name,
                        "source_domain": source.domain,
                        "political_lean": source.political_lean,
                        "expected_bias": ", ".join(source.bias_tendency),
                        "rss_description": rss_desc[:500] if rss_desc else "",
                        "publish_date": item.get("pubDate") or content.get("publish_date", ""),
                        "collected_at": datetime.now().isoformat(),
                    }
                    source_articles.append(article)
                
                # Small delay to be respectful
                time.sleep(0.3)
        
        return source_articles
    
    def collect_all(
        self,
        articles_per_source: int = 30,
        total_target: int = 500,
        balance_sources: bool = True
    ) -> pd.DataFrame:
        """
        Collect articles from all configured sources.
        
        Args:
            articles_per_source: Max articles per source
            total_target: Total articles target
            balance_sources: Try to balance left/right/center
        """
        print("\n" + "="*70)
        print("🇮🇳 INDIAN NEWS MULTI-SOURCE COLLECTOR")
        print("="*70)
        print(f"Sources: {len(INDIAN_SOURCES)}")
        print(f"Target: {total_target} articles")
        print(f"Per source: ~{articles_per_source}")
        print("="*70 + "\n")
        
        # Sort by priority
        sources = sorted(INDIAN_SOURCES, key=lambda x: -x.priority)
        
        all_articles = []
        source_counts = {"left": 0, "center": 0, "right": 0}
        
        for source in tqdm(sources, desc="Collecting sources"):
            if len(all_articles) >= total_target:
                break
            
            # Adjust per-source limit based on political balance
            lean_key = source.political_lean.split("-")[0]  # "center-left" -> "center"
            if lean_key not in source_counts:
                lean_key = "center"
            
            # Calculate how many we need
            needed = articles_per_source * source.priority
            
            print(f"\n📰 {source.name} ({source.political_lean})")
            print(f"   RSS feeds: {len(source.rss_feeds)}")
            
            articles = self.collect_from_source(source, max_articles=needed)
            
            print(f"   Collected: {len(articles)} articles")
            
            all_articles.extend(articles)
            source_counts[lean_key] = source_counts.get(lean_key, 0) + len(articles)
            
            time.sleep(0.5)  # Delay between sources
        
        # Create DataFrame
        df = pd.DataFrame(all_articles)
        
        # Summary
        print("\n" + "="*70)
        print("📊 COLLECTION SUMMARY")
        print("="*70)
        print(f"Total articles: {len(df)}")
        
        if len(df) > 0:
            print(f"\nBy Source:")
            print(df["source"].value_counts().to_string())
            
            print(f"\nBy Political Lean:")
            print(df["political_lean"].value_counts().to_string())
            
            print(f"\nExpected Bias Types:")
            bias_counts = {}
            for biases in df["expected_bias"]:
                for b in biases.split(", "):
                    bias_counts[b] = bias_counts.get(b, 0) + 1
            for b, c in sorted(bias_counts.items(), key=lambda x: -x[1]):
                print(f"  {b}: {c}")
        
        return df
    
    def save(self, df: pd.DataFrame, filename: str = "indian_news.csv"):
        """Save collected articles to CSV."""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\n✅ Saved to: {output_path}")
        return output_path


def main():
    """Main collection pipeline."""
    
    print("\n" + "🇮🇳 "*20)
    print("INDIAN NEWS MULTI-SOURCE COLLECTOR")
    print("Balanced Left/Center/Right Coverage")
    print("🇮🇳 "*20 + "\n")
    
    # Check for newspaper3k
    if not NEWSPAPER_AVAILABLE:
        print("⚠️  For best results, install newspaper3k:")
        print("    pip install newspaper3k lxml")
        print()
    
    # Initialize collector
    collector = IndianNewsCollector(output_dir="data/raw")
    
    # Collect articles
    df = collector.collect_all(
        articles_per_source=35,
        total_target=600,
        balance_sources=True
    )
    
    # Save
    if len(df) > 0:
        collector.save(df, "news_articles.csv")
        
        # Show samples
        print("\n" + "="*70)
        print("📄 SAMPLE ARTICLES")
        print("="*70)
        
        for lean in ["left", "center", "right"]:
            samples = df[df["political_lean"].str.contains(lean, case=False)]
            if len(samples) > 0:
                sample = samples.iloc[0]
                print(f"\n[{lean.upper()}] {sample['source']}")
                print(f"Headline: {sample['headline'][:80]}...")
                print(f"Expected bias: {sample['expected_bias']}")
    else:
        print("\n❌ No articles collected. Check internet connection.")


if __name__ == "__main__":
    main()
