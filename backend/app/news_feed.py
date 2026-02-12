"""
Live News Feed Service
=======================

Aggregates news from multiple sources for the Perspective frontend.
- RSS feeds from major Indian outlets (free, no API key)
- NewsAPI for broader coverage (optional, requires API key)

Includes caching to minimize requests and improve performance.
"""

import os
import time
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from collections import OrderedDict
import re
from urllib.parse import urlparse
import logging
from xml.etree.ElementTree import Element  # For type hints

# Use defusedxml to prevent XXE attacks
try:
    import defusedxml.ElementTree as ET
except ImportError:
    # Fallback with warning - in production, defusedxml should be installed
    import xml.etree.ElementTree as ET
    logging.warning("defusedxml not installed - using standard XML parser. Install defusedxml for production.")

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Represents a news article."""
    id: str
    title: str
    description: str
    url: str
    source: str
    source_lean: str  # "left", "center", "right", "unknown"
    published_at: str
    image_url: Optional[str] = None
    category: str = "general"
    
    def to_dict(self) -> dict:
        return asdict(self)


# RSS Feed Configuration - Major Indian News Sources
RSS_SOURCES = [
    # === MAJOR OUTLETS (Center) ===
    {
        "name": "Times of India",
        "url": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
        "lean": "center",
        "category": "general"
    },
    {
        "name": "Hindustan Times",
        "url": "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
        "lean": "center",
        "category": "india"
    },
    {
        "name": "NDTV",
        "url": "https://feeds.feedburner.com/ndtvnews-top-stories",
        "lean": "center",
        "category": "general"
    },
    {
        "name": "India Today",
        "url": "https://www.indiatoday.in/rss/home",
        "lean": "center",
        "category": "general"
    },
    {
        "name": "The Hindu",
        "url": "https://www.thehindu.com/news/national/feeder/default.rss",
        "lean": "center",
        "category": "india"
    },
    {
        "name": "Indian Express",
        "url": "https://indianexpress.com/feed/",
        "lean": "center",
        "category": "general"
    },
    
    # === LEFT-LEANING ===
    {
        "name": "News18",
        "url": "https://www.news18.com/rss/india.xml",
        "lean": "center-left",
        "category": "india"
    },
    {
        "name": "DNA India",
        "url": "https://www.dnaindia.com/feeds/india.xml",
        "lean": "center-left", 
        "category": "india"
    },
    {
        "name": "Zee News",
        "url": "https://zeenews.india.com/rss/india-national-news.xml",
        "lean": "center-left",
        "category": "india"
    },
    
    # === RIGHT-LEANING ===
    {
        "name": "Swarajya",
        "url": "https://swarajyamag.com/feed",
        "lean": "right",
        "category": "opinion"
    },
    {
        "name": "OpIndia",
        "url": "https://www.opindia.com/feed/",
        "lean": "right",
        "category": "general"
    },
    
    # === GLOBAL NEWS ===
    {
        "name": "BBC India",
        "url": "https://feeds.bbci.co.uk/news/world/asia/india/rss.xml",
        "lean": "center",
        "category": "world"
    },
    {
        "name": "Al Jazeera",
        "url": "https://www.aljazeera.com/xml/rss/all.xml",
        "lean": "center",
        "category": "world"
    },
]


class NewsFeedService:
    """
    Aggregates news from multiple sources.
    Implements caching to reduce API calls.
    """
    
    # Cache duration in seconds (5 minutes)
    CACHE_DURATION = 300
    
    # Maximum cache entries to prevent unbounded growth
    MAX_CACHE_SIZE = 100
    
    # Request timeout
    TIMEOUT = 10
    
    # Max parallel RSS fetch workers
    MAX_WORKERS = 5
    
    def __init__(self, news_api_key: Optional[str] = None):
        """
        Initialize the news feed service.
        
        Args:
            news_api_key: Optional NewsAPI key for additional sources
        """
        self.news_api_key = news_api_key or os.getenv("NEWS_API_KEY")
        self._cache: OrderedDict = OrderedDict()  # LRU cache with max size
        
    def _get_cache_key(self, *args) -> str:
        """Generate a cache key from arguments."""
        return hashlib.md5(str(args).encode()).hexdigest()
    
    def _set_cache(self, key: str, data: Any):
        """Set cache with LRU eviction."""
        # Remove oldest entries if at capacity
        while len(self._cache) >= self.MAX_CACHE_SIZE:
            self._cache.popitem(last=False)
        self._cache[key] = (time.time(), data)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
        timestamp, _ = self._cache[cache_key]
        return (time.time() - timestamp) < self.CACHE_DURATION
    
    def _parse_rss_date(self, date_str: str) -> str:
        """Parse various RSS date formats to ISO format."""
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.isoformat()
            except ValueError:
                continue
        
        # Fallback to current time
        return datetime.now().isoformat()
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""
        clean = re.sub(r'<[^>]+>', '', text)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean[:500]  # Limit description length
    
    def _extract_image(self, item: Element, namespaces: dict) -> Optional[str]:
        """Extract image URL from RSS item."""
        # Try media:content
        media = item.find('.//media:content', namespaces)
        if media is not None and media.get('url'):
            return media.get('url')
        
        # Try media:thumbnail
        thumb = item.find('.//media:thumbnail', namespaces)
        if thumb is not None and thumb.get('url'):
            return thumb.get('url')
        
        # Try enclosure
        enclosure = item.find('enclosure')
        if enclosure is not None and enclosure.get('type', '').startswith('image'):
            return enclosure.get('url')
        
        # Try to find image in content
        content = item.find('description')
        if content is not None and content.text:
            img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', content.text)
            if img_match:
                return img_match.group(1)
        
        return None
    
    def _fetch_rss_feed(self, source: dict) -> List[NewsArticle]:
        """Fetch and parse a single RSS feed."""
        articles = []
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(
                source['url'], 
                headers=headers, 
                timeout=self.TIMEOUT
            )
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            # Handle namespaces
            namespaces = {
                'media': 'http://search.yahoo.com/mrss/',
                'content': 'http://purl.org/rss/1.0/modules/content/',
            }
            
            # Find items (RSS 2.0 format)
            items = root.findall('.//item')
            
            for item in items[:10]:  # Limit to 10 per source
                try:
                    title_elem = item.find('title')
                    link_elem = item.find('link')
                    desc_elem = item.find('description')
                    pub_date_elem = item.find('pubDate')
                    
                    if title_elem is None or link_elem is None:
                        continue
                    
                    title = self._clean_html(title_elem.text or "")
                    url = link_elem.text or ""
                    description = self._clean_html(desc_elem.text if desc_elem is not None and desc_elem.text else "")
                    pub_date = pub_date_elem.text if pub_date_elem is not None and pub_date_elem.text else ""
                    
                    # Generate unique ID
                    article_id = hashlib.md5(url.encode()).hexdigest()[:12]
                    
                    article = NewsArticle(
                        id=article_id,
                        title=title,
                        description=description,
                        url=url,
                        source=source['name'],
                        source_lean=source['lean'],
                        published_at=self._parse_rss_date(pub_date),
                        image_url=self._extract_image(item, namespaces),
                        category=source['category']
                    )
                    articles.append(article)
                    
                except Exception as e:
                    logger.debug(f"Error parsing item from {source['name']}: {e}")
                    continue
                    
        except requests.RequestException as e:
            logger.warning(f"Error fetching {source['name']}: {e}")
        except ET.ParseError as e:
            logger.warning(f"Error parsing XML from {source['name']}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error from {source['name']}: {e}")
            
        return articles
    
    def _fetch_newsapi(self, query: str = "India", category: Optional[str] = None) -> List[NewsArticle]:
        """Fetch news from NewsAPI (if API key available)."""
        if not self.news_api_key:
            return []
            
        articles = []
        
        try:
            params = {
                'apiKey': self.news_api_key,
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20
            }
            
            if category:
                params['category'] = category
            
            response = requests.get(
                'https://newsapi.org/v2/everything',
                params=params,
                timeout=self.TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            for item in data.get('articles', []):
                article_id = hashlib.md5(item['url'].encode()).hexdigest()[:12]
                
                article = NewsArticle(
                    id=article_id,
                    title=item.get('title', ''),
                    description=item.get('description', '') or '',
                    url=item.get('url', ''),
                    source=item.get('source', {}).get('name', 'Unknown'),
                    source_lean='unknown',
                    published_at=item.get('publishedAt', ''),
                    image_url=item.get('urlToImage'),
                    category='newsapi'
                )
                articles.append(article)
                
        except Exception as e:
            logger.warning(f"NewsAPI error: {e}")
            
        return articles
    
    def get_feed(
        self, 
        category: Optional[str] = None,
        lean: Optional[str] = None,
        limit: int = 50
    ) -> Dict:
        """
        Get aggregated news feed.
        
        Args:
            category: Filter by category (general, india, world, opinion)
            lean: Filter by political lean (left, center, right)
            limit: Maximum number of articles
            
        Returns:
            Dict with articles and metadata
        """
        cache_key = self._get_cache_key(category, lean, limit)
        
        # Check cache
        if self._is_cache_valid(cache_key):
            _, cached_data = self._cache[cache_key]
            return cached_data
        
        all_articles = []
        
        # Filter sources based on criteria
        sources_to_fetch = [
            source for source in RSS_SOURCES
            if (not category or source['category'] == category) and
               (not lean or source['lean'] == lean or lean in source['lean'])
        ]
        
        # Fetch from RSS sources in parallel
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            future_to_source = {
                executor.submit(self._fetch_rss_feed, source): source
                for source in sources_to_fetch
            }
            for future in as_completed(future_to_source):
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                except Exception as e:
                    source = future_to_source[future]
                    logger.warning(f"Failed to fetch {source['name']}: {e}")
        
        # Add NewsAPI if available and no specific category filter
        if not category and self.news_api_key:
            newsapi_articles = self._fetch_newsapi("India news")
            all_articles.extend(newsapi_articles)
        
        # Sort by published date (newest first)
        all_articles.sort(
            key=lambda x: x.published_at if x.published_at else "",
            reverse=True
        )
        
        # Deduplicate by title similarity
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title_key = article.title.lower()[:50]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        # Apply limit
        limited_articles = unique_articles[:limit]
        
        result = {
            "articles": [a.to_dict() for a in limited_articles],
            "total": len(limited_articles),
            "sources_count": len(set(a.source for a in limited_articles)),
            "fetched_at": datetime.now().isoformat(),
            "filters": {
                "category": category,
                "lean": lean
            }
        }
        
        # Cache result with LRU eviction
        self._set_cache(cache_key, result)
        
        return result
    
    def get_source_info(self) -> List[Dict]:
        """Get information about all news sources."""
        return [
            {
                "name": s["name"],
                "lean": s["lean"],
                "category": s["category"]
            }
            for s in RSS_SOURCES
        ]


# Singleton instance
_news_feed_service: Optional[NewsFeedService] = None


def get_news_feed_service() -> NewsFeedService:
    """Get or create the news feed service singleton."""
    global _news_feed_service
    if _news_feed_service is None:
        _news_feed_service = NewsFeedService()
    return _news_feed_service
