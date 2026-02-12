"""
Flask Application Factory for Perspective API
==============================================

Creates and configures the Flask application.
"""

import os
import time
import logging
from typing import Optional
from flask import Flask, jsonify, g
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config_name: Optional[str] = None) -> Flask:
    """
    Application factory.
    
    Args:
        config_name: Configuration to use ('development', 'production', 'testing')
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Auto-detect config from environment if not provided
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')
    
    # Load configuration
    if config_name == 'production':
        app.config.from_object('backend.app.config.ProductionConfig')
    elif config_name == 'testing':
        app.config.from_object('backend.app.config.TestingConfig')
    else:
        app.config.from_object('backend.app.config.DevelopmentConfig')
    
    # Configure logging level
    log_level = app.config.get('LOG_LEVEL', 'INFO')
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    # Get CORS origins from environment or use defaults
    cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',')
    
    # Enable CORS for frontend
    CORS(app, resources={
        r"/api/*": {
            "origins": cors_origins,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    })
    
    # Setup rate limiting
    if app.config.get('RATE_LIMIT_ENABLED', False):
        try:
            from flask_limiter import Limiter
            from flask_limiter.util import get_remote_address
            
            limiter = Limiter(
                key_func=get_remote_address,
                app=app,
                default_limits=[app.config.get('RATE_LIMIT_DEFAULT', '100 per minute')],
                storage_uri="memory://",
            )
            app.limiter = limiter
            logger.info("Rate limiting enabled")
            
            # Custom rate limit exceeded handler
            @app.errorhandler(429)
            def rate_limit_handler(e):
                return jsonify({
                    'success': False,
                    'error': 'Rate limit exceeded. Please try again later.',
                    'retry_after': e.description
                }), 429
                
        except ImportError:
            logger.warning("flask-limiter not installed, rate limiting disabled")
    
    # Add security headers
    @app.after_request
    def add_security_headers(response):
        """Add security headers to all responses."""
        security_headers = app.config.get('SECURITY_HEADERS', {})
        for header, value in security_headers.items():
            response.headers[header] = value
        return response
    
    # Request timeout handling
    @app.before_request
    def set_request_timeout():
        """Set request timeout context."""
        g.request_start_time = time.time()
        g.request_timeout = app.config.get('REQUEST_TIMEOUT', 30)
    
    @app.after_request
    def log_request_time(response):
        """Log request processing time."""
        from flask import request
        if hasattr(g, 'request_start_time'):
            elapsed = time.time() - g.request_start_time
            if elapsed > 1.0:  # Log slow requests
                logger.warning(f"Slow request: {request.path} took {elapsed:.2f}s")
        return response
    
    # Register blueprints
    from backend.app.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Global error handlers
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'success': False,
            'error': 'Bad request'
        }), 400
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'success': False,
            'error': 'Resource not found'
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500
    
    # Health check at root
    @app.route('/')
    def root():
        return {
            'name': 'Perspective API',
            'version': '1.0.0',
            'status': 'running',
            'environment': config_name
        }
    
    logger.info(f"Application created with {config_name} configuration")
    return app
