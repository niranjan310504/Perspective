"""
API Route Tests
===============

Tests for the Perspective API endpoints.
"""

import pytest
import json
from backend.app import create_app
from backend.app import routes as api_routes


@pytest.fixture
def app():
    """Create application for testing."""
    app = create_app('testing')
    yield app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get('/')
        assert response.status_code == 200
        data = response.get_json()
        assert data['name'] == 'Perspective API'
        assert data['status'] == 'running'
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'


class TestAnalyzeEndpoint:
    """Test bias analysis endpoint."""
    
    def test_analyze_valid_text(self, client, monkeypatch):
        """Test analysis with valid text input."""
        monkeypatch.setattr(api_routes, '_predictor', None)
        monkeypatch.setattr(api_routes, 'check_facts', lambda text: {
            'status': 'not_checked',
            'message': 'Fact checking not available in tests',
            'fact_checks': [],
            'claims_searched': 0
        })

        response = client.post(
            '/api/analyze',
            data=json.dumps({
                'text': 'This is a test news article about government policies. ' * 10
            }),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'data' in data
        assert 'processing_time_ms' in data
        assert data['data']['analysis_mode'] in ['heuristic_fallback', 'trained_model']
        assert 'biases' in data['data']
        assert 'detected_biases' in data['data']
        assert 'summary' in data['data']
        assert 'fact_check' in data['data']
        assert 'explanations' in data['data']
    
    def test_analyze_empty_text(self, client):
        """Test analysis with empty text."""
        response = client.post(
            '/api/analyze',
            data=json.dumps({'text': ''}),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
        assert 'error_code' in data
    
    def test_analyze_short_text(self, client):
        """Test analysis with too short text."""
        response = client.post(
            '/api/analyze',
            data=json.dumps({'text': 'Too short'}),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
        assert data['error_code'] == 'text_too_short'
    
    def test_analyze_no_json(self, client):
        """Test analysis without JSON body."""
        response = client.post(
            '/api/analyze',
            data='not json',
            content_type='text/plain'
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data['error_code'] == 'invalid_json'
    
    def test_analyze_url_valid_domain(self, client, monkeypatch):
        """Test URL analysis with valid domain."""
        monkeypatch.setattr(api_routes, '_predictor', None)
        monkeypatch.setattr(api_routes, 'check_facts', lambda text: {
            'status': 'not_checked',
            'message': 'Fact checking not available in tests',
            'fact_checks': [],
            'claims_searched': 0
        })
        monkeypatch.setattr(api_routes, 'extract_article_from_url', lambda url: (
            'The government and opposition debated policies in parliament for hours. ' * 4,
            None
        ))
        monkeypatch.setattr(api_routes, 'get_fallback_text_from_feed', lambda url: (None, {
            'code': 'feed_fallback_not_found',
            'message': 'No matching feed article was found for this URL',
            'retryable': True,
        }))

        response = client.post(
            '/api/analyze',
            data=json.dumps({
                'url': 'https://www.thehindu.com/news/some-article/'
            }),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert data['data']['analysis_mode'] in ['heuristic_fallback', 'trained_model']
        assert data['data']['fact_check']['status'] == 'not_checked'
    
    def test_analyze_url_blocked_domain(self, client):
        """Test URL analysis with blocked domain."""
        response = client.post(
            '/api/analyze',
            data=json.dumps({
                'url': 'https://malicious-site.com/article'
            }),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
        assert data['error_code'] in ['unsafe_url', 'invalid_url']


class TestInputValidation:
    """Test input validation."""
    
    def test_text_length_limit(self, client):
        """Test text exceeding maximum length."""
        # Create text that exceeds 50000 characters
        long_text = 'a' * 60000
        response = client.post(
            '/api/analyze',
            data=json.dumps({'text': long_text}),
            content_type='application/json'
        )
        assert response.status_code == 400
    
    def test_missing_both_text_and_url(self, client):
        """Test request without text or URL."""
        response = client.post(
            '/api/analyze',
            data=json.dumps({}),
            content_type='application/json'
        )
        assert response.status_code == 400
