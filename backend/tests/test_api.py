"""
API Route Tests
===============

Tests for the Perspective API endpoints.
"""

import pytest
import json
from backend.app import create_app


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
    
    def test_analyze_valid_text(self, client):
        """Test analysis with valid text input."""
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
        assert 'biases' in data
        assert 'detected_biases' in data
        assert 'summary' in data
    
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
    
    def test_analyze_no_json(self, client):
        """Test analysis without JSON body."""
        response = client.post(
            '/api/analyze',
            data='not json',
            content_type='text/plain'
        )
        assert response.status_code == 400
    
    def test_analyze_url_valid_domain(self, client):
        """Test URL analysis with valid domain."""
        # This test may fail if the URL is not accessible
        # It's mainly to test the domain validation
        response = client.post(
            '/api/analyze',
            data=json.dumps({
                'url': 'https://www.thehindu.com/news/some-article/'
            }),
            content_type='application/json'
        )
        # Either success or error fetching - not 400 for domain
        assert response.status_code in [200, 500]
    
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
        assert 'not in the allowed list' in data.get('error', '')


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
