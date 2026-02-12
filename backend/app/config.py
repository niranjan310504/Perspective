"""
Configuration Classes for Flask Application
=============================================
"""

import os
from pathlib import Path

# Get project root directory (parent of backend/)
PROJECT_ROOT = Path(__file__).parent.parent.parent


# Allowed domains for URL extraction (SSRF protection)
ALLOWED_URL_DOMAINS = [
    'ndtv.com', 'thehindu.com', 'hindustantimes.com', 'indianexpress.com',
    'timesofindia.indiatimes.com', 'news18.com', 'thewire.in', 'theprint.in',
    'scroll.in', 'firstpost.com', 'swarajyamag.com', 'opindia.com',
    'livemint.com', 'economictimes.indiatimes.com', 'business-standard.com',
    'reuters.com', 'bbc.com', 'aljazeera.com', 'cnn.com', 'theguardian.com'
]


class BaseConfig:
    """Base configuration."""
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    JSON_SORT_KEYS = False
    
    # Security headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    }
    
    # Model - use absolute path
    MODEL_DIR = os.getenv('MODEL_DIR', str(PROJECT_ROOT / 'model' / 'checkpoints'))
    MODEL_DEVICE = os.getenv('MODEL_DEVICE', None)  # None = auto-detect
    CLASSIFICATION_THRESHOLD = 0.5
    
    # Request limits
    MAX_TEXT_LENGTH = 50000  # characters
    MIN_TEXT_LENGTH = 50  # minimum characters for analysis
    MIN_WORD_COUNT = 10  # minimum words for analysis
    REQUEST_TIMEOUT = 30  # seconds
    
    # URL extraction settings
    ALLOWED_URL_DOMAINS = ALLOWED_URL_DOMAINS
    URL_FETCH_TIMEOUT = 10  # seconds
    
    # Rate limiting (requests per time period)
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_DEFAULT = "100 per minute"
    RATE_LIMIT_ANALYZE = "30 per minute"
    RATE_LIMIT_BATCH = "5 per minute"
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class DevelopmentConfig(BaseConfig):
    """Development configuration."""
    
    DEBUG = True
    TESTING = False
    
    # More lenient limits for development
    RATE_LIMIT_ENABLED = False
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(BaseConfig):
    """Production configuration."""
    
    DEBUG = False
    TESTING = False
    
    # Stricter settings - SECRET_KEY must be set via environment variable
    SECRET_KEY = os.getenv('SECRET_KEY', '')
    
    # Production rate limits
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_DEFAULT = "60 per minute"
    RATE_LIMIT_ANALYZE = "20 per minute"
    RATE_LIMIT_BATCH = "3 per minute"
    
    # Stricter security headers for production
    SECURITY_HEADERS = {
        **BaseConfig.SECURITY_HEADERS,
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    }
    
    @classmethod
    def init_app(cls, app):
        """Validate production configuration."""
        if not cls.SECRET_KEY:
            raise ValueError("SECRET_KEY must be set in production!")


class TestingConfig(BaseConfig):
    """Testing configuration."""
    
    DEBUG = True
    TESTING = True
    
    RATE_LIMIT_ENABLED = False
    LOG_LEVEL = 'DEBUG'
