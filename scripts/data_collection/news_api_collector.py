"""
News Data Collection using NewsAPI
====================================

Fetches news articles from Indian news sources using NewsAPI.
Free tier: 100 requests/day, articles up to 1 month old.

Get your API key at: https://newsapi.org/register
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


class NewsAPICollector:
    """
    Collects news articles from Indian sources using NewsAPI.
    """
    
    BASE_URL = "https://newsapi.org/v2"
    
    # Indian news sources available on NewsAPI
    INDIAN_SOURCES = [
        "the-times-of-india",
        "the-hindu",
        "google-news-in",
    ]
    
    # Search queries for diverse topics
    SEARCH_QUERIES = [
        # Political topics
        "BJP government India",
        "Congress party India",
        "Parliament India",
        "Prime Minister Modi",
        "opposition party India",
        
        # Religious/communal topics
        "Hindu Muslim India",
        "temple mosque India",
        "religious festival India",
        
        # Regional topics
        "Maharashtra news",
        "Tamil Nadu news", 
        "Delhi news",
        "Kashmir conflict",
        
        # Gender topics
        "women empowerment India",
        "gender equality India",
        
        # General news
        "India economy",
        "India election",
        "India cricket",
        "Bollywood news",
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the collector.
        
        Args:
            api_key: NewsAPI key (or set NEWS_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "NewsAPI key required. Get one at https://newsapi.org/register\n"
                "Set it as NEWS_API_KEY environment variable or pass to constructor."
            )
        
        self.session = requests.Session()
        self.session.headers.update({"X-Api-Key": self.api_key})
        
        self.collected_articles = []
        self.article_urls = set()  # For deduplication
        
    def fetch_everything(
        self,
        query: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        page_size: int = 100,
        max_pages: int = 1
    ) -> List[Dict]:
        """
        Fetch articles using the /everything endpoint.
        
        Args:
            query: Search query
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            page_size: Articles per page (max 100)
            max_pages: Maximum pages to fetch
            
        Returns:
            List of article dictionaries
        """
        # Default to last 7 days (free tier limitation)
        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")
        
        articles = []
        
        for page in range(1, max_pages + 1):
            params = {
                "q": query,
                "from": from_date,
                "to": to_date,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": page_size,
                "page": page
            }
            
            try:
                response = self.session.get(f"{self.BASE_URL}/everything", params=params)
                response.raise_for_status()
                data = response.json()
                
                if data["status"] != "ok":
                    print(f"API error: {data.get('message', 'Unknown error')}")
                    break
                
                page_articles = data.get("articles", [])
                if not page_articles:
                    break
                    
                articles.extend(page_articles)
                
                # Check if we've got all articles
                if len(articles) >= data.get("totalResults", 0):
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                break
            
            # Rate limiting
            time.sleep(0.5)
        
        return articles
    
    def fetch_top_headlines(
        self,
        country: str = "in",
        category: Optional[str] = None,
        page_size: int = 100
    ) -> List[Dict]:
        """
        Fetch top headlines from India.
        
        Args:
            country: Country code (in = India)
            category: Optional category (business, technology, etc.)
            page_size: Number of articles
            
        Returns:
            List of article dictionaries
        """
        params = {
            "country": country,
            "pageSize": page_size
        }
        
        if category:
            params["category"] = category
        
        try:
            response = self.session.get(f"{self.BASE_URL}/top-headlines", params=params)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] == "ok":
                return data.get("articles", [])
            else:
                print(f"API error: {data.get('message', 'Unknown error')}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return []
    
    def collect_dataset(
        self,
        num_articles: int = 500,
        output_path: str = "data/raw/articles_raw.csv"
    ) -> pd.DataFrame:
        """
        Collect a dataset of articles from various queries.
        
        Args:
            num_articles: Target number of articles
            output_path: Path to save CSV
            
        Returns:
            DataFrame with collected articles
        """
        print(f"Collecting ~{num_articles} articles...")
        print("Note: Free tier limited to 100 requests/day\n")
        
        articles_per_query = max(10, num_articles // len(self.SEARCH_QUERIES))
        
        # Collect from search queries
        for query in tqdm(self.SEARCH_QUERIES, desc="Fetching by query"):
            articles = self.fetch_everything(
                query=query,
                page_size=min(100, articles_per_query)
            )
            self._add_articles(articles, source_query=query)
            
            if len(self.collected_articles) >= num_articles:
                break
            
            time.sleep(1)  # Rate limiting
        
        # Also get top headlines
        print("\nFetching top headlines...")
        for category in ["general", "politics", "entertainment"]:
            headlines = self.fetch_top_headlines(category=category)
            self._add_articles(headlines, source_query=f"headlines_{category}")
            time.sleep(1)
        
        # Convert to DataFrame
        df = self._create_dataframe()
        
        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"\n✓ Collected {len(df)} unique articles")
        print(f"✓ Saved to {output_path}")
        
        return df
    
    def _add_articles(self, articles: List[Dict], source_query: str):
        """Add articles while deduplicating."""
        for article in articles:
            url = article.get("url", "")
            
            # Skip if already collected or no content
            if url in self.article_urls:
                continue
            if not article.get("content") and not article.get("description"):
                continue
            
            self.article_urls.add(url)
            article["_source_query"] = source_query
            self.collected_articles.append(article)
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert collected articles to DataFrame."""
        rows = []
        
        for i, article in enumerate(self.collected_articles):
            # Combine description and content
            content = article.get("content", "") or ""
            description = article.get("description", "") or ""
            full_content = f"{description} {content}".strip()
            
            # Remove "[+XXXX chars]" truncation notice
            if "[+" in full_content:
                full_content = full_content.split("[+")[0].strip()
            
            row = {
                "article_id": f"ART_{datetime.now().strftime('%Y%m%d')}_{i:05d}",
                "source": article.get("source", {}).get("name", "Unknown"),
                "url": article.get("url", ""),
                "publish_date": article.get("publishedAt", "")[:10] if article.get("publishedAt") else "",
                "category": article.get("_source_query", ""),
                "headline": article.get("title", ""),
                "content": full_content,
                "author": article.get("author", "")
            }
            rows.append(row)
        
        return pd.DataFrame(rows)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect news articles using NewsAPI")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="NewsAPI key (or set NEWS_API_KEY env var)"
    )
    parser.add_argument(
        "--num-articles",
        type=int,
        default=500,
        help="Target number of articles to collect"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/articles_raw.csv",
        help="Output CSV path"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("PERSPECTIVE - News Data Collection")
    print("=" * 50)
    print("\nUsing NewsAPI to collect Indian news articles")
    print("Get your free API key at: https://newsapi.org/register\n")
    
    try:
        collector = NewsAPICollector(api_key=args.api_key)
        collector.collect_dataset(
            num_articles=args.num_articles,
            output_path=args.output
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
