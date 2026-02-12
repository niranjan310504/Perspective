"""
GDELT News Data Collection - FREE & UNLIMITED
===============================================

GDELT (Global Database of Events, Language, and Tone) is a FREE, open database
that monitors news from 65+ languages and 150+ countries including India.

Advantages over NewsAPI:
✅ Completely FREE - No API key required
✅ UNLIMITED access - No daily limits
✅ Historical data - Access articles from years back
✅ Indian news coverage - Major Indian outlets included
✅ Full article text - Can fetch complete articles

This script uses GDELT's GKG (Global Knowledge Graph) for Indian news.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import time
import re
from urllib.parse import quote
import hashlib

# For article content extraction
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    print("Note: newspaper3k not installed. Will use basic extraction.")
    print("For better results: pip install newspaper3k")

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


class GDELTCollector:
    """
    Collects Indian news articles from GDELT - FREE and UNLIMITED.
    
    GDELT monitors news worldwide and provides free API access.
    No API key required!
    """
    
    # GDELT API endpoints
    DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    # Indian news domains to filter for
    INDIAN_DOMAINS = [
        "timesofindia.indiatimes.com",
        "thehindu.com",
        "hindustantimes.com",
        "indianexpress.com",
        "ndtv.com",
        "news18.com",
        "indiatoday.in",
        "firstpost.com",
        "scroll.in",
        "thewire.in",
        "livemint.com",
        "economictimes.indiatimes.com",
        "deccanherald.com",
        "telegraphindia.com",
        "tribuneindia.com",
        "theprint.in",
        "swarajyamag.com",
        "newslaundry.com",
        "thequint.com",
        "outlookindia.com",
    ]
    
    # Search queries for diverse bias coverage
    SEARCH_QUERIES = [
        # Political (will likely have political bias)
        "BJP Modi government policy",
        "Congress Rahul Gandhi opposition",
        "Parliament bill passed debate",
        "election campaign rally",
        "chief minister state government",
        
        # Religious/Communal (may have religious bias)
        "Hindu Muslim community",
        "temple mosque dispute",
        "religious festival celebration",
        "communal harmony tension",
        
        # Regional (may have regional bias)
        "Maharashtra Mumbai development",
        "Tamil Nadu Chennai politics",
        "Delhi NCR infrastructure",
        "Kashmir conflict peace",
        "Northeast India states",
        
        # Gender (may have gender bias)
        "women empowerment rights",
        "gender equality discrimination",
        "female politician leader",
        "women safety crime",
        
        # General/Sensationalism
        "breaking news India",
        "shocking revelation scandal",
        "exclusive investigation expose",
        "Bollywood celebrity controversy",
        "cricket India match",
    ]
    
    def __init__(self, output_dir: str = "data/raw"):
        """Initialize GDELT collector."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collected_urls = set()  # For deduplication
        
    def search_articles(
        self, 
        query: str,
        max_records: int = 50,
        days_back: int = 30,
        source_country: str = "IN"
    ) -> List[Dict]:
        """
        Search GDELT for articles matching query.
        
        Args:
            query: Search terms
            max_records: Maximum articles to fetch
            days_back: How many days back to search
            source_country: Country code (IN for India)
            
        Returns:
            List of article metadata dictionaries
        """
        # Build GDELT query
        params = {
            "query": f"{query} sourcecountry:{source_country}",
            "mode": "artlist",
            "maxrecords": str(max_records),
            "format": "json",
            "sort": "hybridrel",  # Relevance + recency
            "timespan": f"{days_back}d"
        }
        
        try:
            response = requests.get(self.DOC_API, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get("articles", [])
            
            return articles
            
        except Exception as e:
            print(f"  Error fetching '{query}': {e}")
            return []
    
    def extract_article_content(self, url: str) -> Optional[Dict]:
        """
        Extract full article content from URL.
        
        Args:
            url: Article URL
            
        Returns:
            Dictionary with title, text, etc. or None if failed
        """
        try:
            if NEWSPAPER_AVAILABLE:
                article = Article(url)
                article.download()
                article.parse()
                
                return {
                    "headline": article.title,
                    "content": article.text,
                    "authors": ", ".join(article.authors) if article.authors else "",
                    "publish_date": str(article.publish_date) if article.publish_date else "",
                }
            else:
                # Basic extraction using requests
                response = requests.get(url, timeout=15, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                })
                
                # Very basic title extraction
                title_match = re.search(r"<title>(.*?)</title>", response.text, re.IGNORECASE)
                title = title_match.group(1) if title_match else "Unknown"
                
                # Basic paragraph extraction
                paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", response.text, re.DOTALL)
                text = " ".join([re.sub(r"<[^>]+>", "", p) for p in paragraphs[:20]])
                
                return {
                    "headline": title,
                    "content": text[:5000],  # Limit content length
                    "authors": "",
                    "publish_date": "",
                }
                
        except Exception as e:
            return None
    
    def collect_articles(
        self,
        queries: Optional[List[str]] = None,
        articles_per_query: int = 30,
        total_target: int = 500,
        days_back: int = 60
    ) -> pd.DataFrame:
        """
        Collect articles from GDELT across multiple queries.
        
        Args:
            queries: List of search queries (uses default if None)
            articles_per_query: Articles to fetch per query
            total_target: Stop when reaching this many articles
            days_back: How far back to search
            
        Returns:
            DataFrame with collected articles
        """
        queries = queries or self.SEARCH_QUERIES
        all_articles = []
        
        print("\n" + "="*60)
        print("GDELT News Data Collector - FREE & UNLIMITED")
        print("="*60)
        print(f"Queries: {len(queries)}")
        print(f"Target articles: {total_target}")
        print(f"Time range: Last {days_back} days")
        print("="*60 + "\n")
        
        for query in tqdm(queries, desc="Searching queries"):
            if len(all_articles) >= total_target:
                break
                
            print(f"\nSearching: '{query}'")
            
            # Search GDELT
            results = self.search_articles(
                query=query,
                max_records=articles_per_query,
                days_back=days_back
            )
            
            print(f"  Found {len(results)} results")
            
            # Filter for Indian domains
            indian_results = [
                r for r in results 
                if any(domain in r.get("url", "") for domain in self.INDIAN_DOMAINS)
            ]
            
            print(f"  Indian sources: {len(indian_results)}")
            
            # Extract content
            for result in indian_results:
                url = result.get("url", "")
                
                # Skip duplicates
                url_hash = hashlib.md5(url.encode()).hexdigest()
                if url_hash in self.collected_urls:
                    continue
                self.collected_urls.add(url_hash)
                
                # Extract full content
                content = self.extract_article_content(url)
                
                if content and len(content.get("content", "")) > 200:
                    article_data = {
                        "article_id": url_hash[:16],
                        "headline": content["headline"],
                        "content": content["content"],
                        "url": url,
                        "source": result.get("domain", ""),
                        "source_country": result.get("sourcecountry", "IN"),
                        "language": result.get("language", "English"),
                        "seendate": result.get("seendate", ""),
                        "search_query": query,
                        "collected_at": datetime.now().isoformat()
                    }
                    all_articles.append(article_data)
                    
                if len(all_articles) >= total_target:
                    break
            
            # Small delay to be respectful
            time.sleep(0.5)
        
        # Create DataFrame
        df = pd.DataFrame(all_articles)
        
        print(f"\n{'='*60}")
        print(f"Collection Complete!")
        print(f"Total articles: {len(df)}")
        
        if len(df) > 0:
            print(f"\nBy source:")
            print(df["source"].value_counts().head(10))
            
            print(f"\nBy query category:")
            print(df["search_query"].value_counts())
        
        return df
    
    def save_articles(self, df: pd.DataFrame, filename: str = "gdelt_articles.csv"):
        """Save collected articles to CSV."""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\nSaved to: {output_path}")
        return output_path


def main():
    """Main collection pipeline."""
    
    print("\n" + "🌐 "*20)
    print("GDELT Collector - FREE & UNLIMITED Indian News")
    print("🌐 "*20 + "\n")
    
    # Initialize collector
    collector = GDELTCollector(output_dir="data/raw")
    
    # Collect articles
    df = collector.collect_articles(
        articles_per_query=40,
        total_target=600,  # Collect extra for filtering
        days_back=60
    )
    
    # Save
    if len(df) > 0:
        collector.save_articles(df, "news_articles.csv")
        
        # Print sample
        print("\n" + "="*60)
        print("Sample Articles:")
        print("="*60)
        for _, row in df.head(3).iterrows():
            print(f"\nHeadline: {row['headline'][:80]}...")
            print(f"Source: {row['source']}")
            print(f"Content: {row['content'][:150]}...")
    else:
        print("No articles collected. Check internet connection.")


if __name__ == "__main__":
    main()
