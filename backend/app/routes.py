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
from flask import Blueprint, request, jsonify, current_app
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
                # For now, just log - uncomment below to enforce
                # return False
        
        return True
        
    except Exception as e:
        current_app.logger.error(f"URL validation error: {e}")
        return False


def extract_article_from_url(url: str) -> Optional[str]:
    """
    Extract article text from a URL.
    
    Args:
        url: Article URL
        
    Returns:
        Extracted text or None if extraction fails
    """
    try:
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
        
        # Try using newspaper3k
        try:
            from newspaper import Article
            
            article = Article(url)
            article.download()
            article.parse()
            
            return article.text
            
        except ImportError:
            # Fallback to requests + BeautifulSoup
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract text from article-like elements
            article = soup.find('article') or soup.find('main') or soup.body
            
            if article:
                return ' '.join(article.stripped_strings)
            
            return None
            
    except Exception as e:
        current_app.logger.error(f"URL extraction error: {e}")
        return None


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
