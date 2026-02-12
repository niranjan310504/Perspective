"""
API Routes for Perspective
===========================

Endpoints:
- GET  /api/health      - Health check
- POST /api/analyze     - Analyze text for bias (with fact-check)
- GET  /api/bias-types  - Get bias type information
"""

import re
import time
from functools import wraps
from flask import Blueprint, request, jsonify, current_app, g
from typing import Optional

from backend.app.fact_check import check_facts

# Lazy model loading
_predictor = None


def get_predictor():
    """
    Lazy load the bias predictor.
    
    This ensures the model is only loaded once when first needed,
    not at import time.
    """
    global _predictor
    
    if _predictor is None:
        from model.src.inference import BiasPredictor
        
        model_dir = current_app.config.get('MODEL_DIR', 'model/checkpoints')
        device = current_app.config.get('MODEL_DEVICE', None)
        
        try:
            _predictor = BiasPredictor(model_dir=model_dir, device=device)
            current_app.logger.info(f"Model loaded from {model_dir}")
        except Exception as e:
            current_app.logger.error(f"Failed to load model: {e}")
            raise
    
    return _predictor


def rate_limit_analyze(f):
    """Apply rate limiting to expensive analysis endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if hasattr(current_app, 'limiter'):
            limit = current_app.config.get('RATE_LIMIT_ANALYZE', '30 per minute')
            return current_app.limiter.limit(limit)(f)(*args, **kwargs)
        return f(*args, **kwargs)
    return decorated_function


def sanitize_text(text: str) -> str:
    """Remove control characters and sanitize input text."""
    if not text:
        return text
    # Remove control characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text


# Create blueprint
api_bp = Blueprint('api', __name__)


@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON with status and model info
    """
    try:
        predictor = get_predictor()
        model_loaded = True
    except Exception:
        model_loaded = False
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': time.time()
    })


@api_bp.route('/analyze', methods=['POST'])
@rate_limit_analyze
def analyze_text():
    """
    Analyze text for media bias.
    
    Request Body:
        {
            "text": "Article text to analyze",
            "url": "https://example.com/article" (optional, alternative to text)
        }
    
    Returns:
        {
            "success": true,
            "data": {
                "text_preview": "First 200 chars...",
                "biases": {
                    "political": {"detected": true, "score": 0.89},
                    ...
                },
                "detected_biases": ["political", "sensationalism"],
                "summary": "Political bias and sensationalism detected."
            },
            "processing_time_ms": 234
        }
    """
    start_time = time.time()
    
    # Parse request
    data = request.get_json()
    
    if not data:
        return jsonify({
            'success': False,
            'error': 'Request body must be JSON'
        }), 400
    
    # Get text from request
    text = data.get('text')
    url = data.get('url')
    
    if not text and not url:
        return jsonify({
            'success': False,
            'error': 'Either "text" or "url" must be provided'
        }), 400
    
    # If URL provided, extract article text
    if url and not text:
        text = extract_article_from_url(url)
        if not text:
            return jsonify({
                'success': False,
                'error': 'Failed to extract article from URL'
            }), 400
    
    # Validate text length
    max_length = current_app.config.get('MAX_TEXT_LENGTH', 50000)
    if len(text) > max_length:
        return jsonify({
            'success': False,
            'error': f'Text exceeds maximum length of {max_length} characters'
        }), 400
    
    # Minimum text length
    if len(text.split()) < 10:
        return jsonify({
            'success': False,
            'error': 'Text must contain at least 10 words'
        }), 400
    
    # Sanitize text - remove control characters
    text = sanitize_text(text)
    
    try:
        # Get predictor and run inference
        predictor = get_predictor()
        result = predictor.predict(text)
        
        # Run fact-check in parallel (non-blocking for bias detection)
        fact_check_result = check_facts(text)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'success': True,
            'data': {
                'text_preview': text[:200] + '...' if len(text) > 200 else text,
                'biases': result['biases'],
                'detected_biases': result['detected_biases'],
                'summary': result['summary'],
                'fact_check': fact_check_result
            },
            'processing_time_ms': round(processing_time, 2)
        })
        
    except Exception as e:
        current_app.logger.error(f"Analysis error: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal error during analysis'
        }), 500


@api_bp.route('/analyze/batch', methods=['POST'])
@rate_limit_analyze
def analyze_batch():
    """
    Analyze multiple texts for bias.
    
    Request Body:
        {
            "texts": ["Article 1...", "Article 2..."]
        }
    
    Returns:
        {
            "success": true,
            "data": [
                {"biases": {...}, "detected_biases": [...], "summary": "..."},
                ...
            ]
        }
    """
    start_time = time.time()
    
    data = request.get_json()
    
    if not data or 'texts' not in data:
        return jsonify({
            'success': False,
            'error': 'Request must include "texts" array'
        }), 400
    
    texts = data['texts']
    
    if not isinstance(texts, list):
        return jsonify({
            'success': False,
            'error': '"texts" must be an array'
        }), 400
    
    # Limit batch size
    max_batch_size = 20
    if len(texts) > max_batch_size:
        return jsonify({
            'success': False,
            'error': f'Batch size exceeds maximum of {max_batch_size}'
        }), 400
    
    try:
        predictor = get_predictor()
        results = predictor.predict_batch(texts)
        
        processing_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'success': True,
            'data': results,
            'count': len(results),
            'processing_time_ms': round(processing_time, 2)
        })
        
    except Exception as e:
        current_app.logger.error(f"Batch analysis error: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal error during batch analysis'
        }), 500


@api_bp.route('/bias-types', methods=['GET'])
def get_bias_types():
    """
    Get information about all bias types.
    
    Returns:
        {
            "success": true,
            "data": {
                "political": {
                    "name": "Political Bias",
                    "description": "...",
                    "examples": [...],
                    "indicators": [...]
                },
                ...
            }
        }
    """
    from data.schema import BIAS_DESCRIPTIONS
    
    return jsonify({
        'success': True,
        'data': BIAS_DESCRIPTIONS
    })


@api_bp.route('/bias-types/<bias_type>', methods=['GET'])
def get_bias_type_detail(bias_type: str):
    """
    Get detailed information about a specific bias type.
    
    Args:
        bias_type: Type of bias (e.g., 'political', 'gender')
    
    Returns:
        Detailed bias information
    """
    from data.schema import BIAS_DESCRIPTIONS, BIAS_LABELS
    
    if bias_type not in BIAS_LABELS:
        return jsonify({
            'success': False,
            'error': f'Unknown bias type: {bias_type}',
            'valid_types': BIAS_LABELS
        }), 404
    
    return jsonify({
        'success': True,
        'data': BIAS_DESCRIPTIONS[bias_type]
    })


def is_safe_url(url: str) -> bool:
    """
    Check if URL is safe to fetch (SSRF protection).
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is safe, False otherwise
    """
    from urllib.parse import urlparse
    import socket
    
    try:
        parsed = urlparse(url)
        
        # Only allow http/https
        if parsed.scheme not in ('http', 'https'):
            return False
        
        hostname = parsed.hostname
        if not hostname:
            return False
        
        # Block localhost and private IPs
        blocked_hosts = ['localhost', '127.0.0.1', '0.0.0.0']
        if hostname.lower() in blocked_hosts:
            return False
        
        # Check if it resolves to a private IP
        try:
            ip = socket.gethostbyname(hostname)
            # Block private IP ranges
            if ip.startswith('127.') or ip.startswith('10.') or \
               ip.startswith('192.168.') or ip.startswith('172.16.') or \
               ip.startswith('172.17.') or ip.startswith('172.18.') or \
               ip.startswith('172.19.') or ip.startswith('172.2') or \
               ip.startswith('172.30.') or ip.startswith('172.31.') or \
               ip.startswith('169.254.') or ip == '0.0.0.0':
                return False
        except socket.gaierror:
            return False
        
        # Check against allowed domains (optional - enable in production)
        allowed_domains = current_app.config.get('ALLOWED_URL_DOMAINS', [])
        if allowed_domains:
            domain_allowed = any(
                hostname.endswith(domain) or hostname == domain 
                for domain in allowed_domains
            )
            if not domain_allowed:
                current_app.logger.warning(f"URL domain not in allowed list: {hostname}")
                return False  # Enforce domain allowlist
        
        # Block IPv6 private addresses
        if ip.startswith('::1') or ip.startswith('fc') or ip.startswith('fd') or ip.startswith('fe80'):
            return False
        
        return True
        
    except Exception as e:
        current_app.logger.error(f"URL validation error: {e}")
        return False


def extract_article_from_url(url: str) -> Optional[str]:
    """
    Extract article text from a URL with multiple fallback methods.
    Uses cloudscraper to bypass anti-bot protection (Cloudflare, etc.)
    
    Args:
        url: Article URL
        
    Returns:
        Extracted text or None if extraction fails
    """
    from bs4 import BeautifulSoup
    
    # SSRF Protection: Validate URL is safe
    if not is_safe_url(url):
        current_app.logger.warning(f"Blocked unsafe URL: {url}")
        return None
    
    # Validate URL format
    url_pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    
    if not url_pattern.match(url):
        return None
    
    # Method 1: Use cloudscraper (bypasses Cloudflare and anti-bot protection)
    try:
        import cloudscraper
        
        scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
        )
        response = scraper.get(url, timeout=20)
        
        # Check if we got redirected to homepage (bot detection)
        if response.status_code == 200 and response.url.rstrip('/') != url.rstrip('/'):
            # Check if it's just a trailing slash difference or actual redirect
            if not response.url.startswith(url.rstrip('/')):
                current_app.logger.warning(f"Redirected from {url} to {response.url}")
                raise Exception("Redirected to different page")
        
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript', 'form']):
            element.decompose()
        
        # Try multiple selectors in order of specificity
        article_selectors = [
            '.entry-content',
            '.post-content', 
            '.article-content',
            '.story-content',
            '[itemprop="articleBody"]',
            'article',
            '.content-area',
            'main',
            '#content',
        ]
        
        for selector in article_selectors:
            element = soup.select_one(selector)
            if element:
                # Get paragraphs for cleaner text
                paragraphs = element.find_all('p')
                if paragraphs:
                    article_text = ' '.join(
                        p.get_text(strip=True) for p in paragraphs 
                        if len(p.get_text(strip=True)) > 20
                    )
                    if len(article_text) > 200:
                        current_app.logger.info(f"Extracted {len(article_text)} chars from {url}")
                        return article_text
        
        # Fallback: get all text from body
        body = soup.find('body')
        if body:
            text = ' '.join(body.stripped_strings)
            if len(text) > 200:
                return text[:10000]  # Limit length
                
    except ImportError:
        current_app.logger.warning("cloudscraper not installed, falling back to requests")
    except Exception as e:
        current_app.logger.debug(f"cloudscraper extraction failed: {e}")
    
    # Method 2: Fallback to newspaper3k
    try:
        from newspaper import Article, Config
        
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        config.request_timeout = 15
        config.fetch_images = False
        
        article = Article(url, config=config)
        article.download()
        article.parse()
        
        if article.text and len(article.text) > 100:
            current_app.logger.info(f"Extracted {len(article.text)} chars using newspaper3k from {url}")
            return article.text
            
    except Exception as e:
        current_app.logger.debug(f"newspaper3k extraction failed: {e}")
    
    current_app.logger.error(f"All extraction methods failed for URL: {url}")
    return None


# =====================
# News Feed Endpoints
# =====================

@api_bp.route('/news/feed', methods=['GET'])
def get_news_feed():
    """
    Get live news feed from multiple sources.
    
    Query Parameters:
        - category: Filter by category (general, india, world, opinion)
        - lean: Filter by political lean (left, center, right)
        - limit: Maximum articles to return (default: 50, max: 100)
    
    Returns:
        {
            "success": true,
            "data": {
                "articles": [...],
                "total": 50,
                "sources_count": 12,
                "fetched_at": "2024-01-15T10:30:00",
                "filters": {...}
            }
        }
    """
    from backend.app.news_feed import get_news_feed_service
    
    try:
        # Parse query parameters
        category = request.args.get('category')
        lean = request.args.get('lean')
        
        # Parse limit with error handling
        try:
            limit = min(int(request.args.get('limit', 50)), 100)
        except (ValueError, TypeError):
            limit = 50
        
        # Validate parameters
        valid_categories = ['general', 'india', 'world', 'opinion', None]
        valid_leans = ['left', 'center', 'right', 'center-left', None]
        
        if category and category not in valid_categories:
            return jsonify({
                'success': False,
                'error': f'Invalid category. Must be one of: {valid_categories[:-1]}'
            }), 400
            
        if lean and lean not in valid_leans:
            return jsonify({
                'success': False,
                'error': f'Invalid lean. Must be one of: {valid_leans[:-1]}'
            }), 400
        
        # Get news feed
        service = get_news_feed_service()
        feed_data = service.get_feed(category=category, lean=lean, limit=limit)
        
        return jsonify({
            'success': True,
            'data': feed_data
        })
        
    except Exception as e:
        current_app.logger.error(f"News feed error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch news feed'
        }), 500


@api_bp.route('/news/sources', methods=['GET'])
def get_news_sources():
    """
    Get information about available news sources.
    
    Returns:
        {
            "success": true,
            "data": {
                "sources": [
                    {"name": "Times of India", "lean": "center", "category": "general"},
                    ...
                ]
            }
        }
    """
    from backend.app.news_feed import get_news_feed_service
    
    try:
        service = get_news_feed_service()
        sources = service.get_source_info()
        
        return jsonify({
            'success': True,
            'data': {
                'sources': sources,
                'total': len(sources)
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"News sources error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch news sources'
        }), 500


# Error handlers
@api_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        'success': False,
        'error': 'Bad request'
    }), 400


@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Resource not found'
    }), 404


@api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500
