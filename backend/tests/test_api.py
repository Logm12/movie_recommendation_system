"""
Integration tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns OK status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_endpoint(self, client):
        """Test health endpoint returns service status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "embeddings_loaded" in data
        assert "known_users" in data


class TestRecommendEndpoints:
    """Tests for recommendation endpoints."""

    def test_recommend_invalid_user(self, client):
        """Test recommendation for non-existent user returns 404."""
        response = client.get("/recommend/999999")
        # Either 404 (user not found) or 503 (model not loaded)
        assert response.status_code in [404, 503]

    def test_cold_start_empty_request(self, client):
        """Test cold start with empty request returns 400."""
        response = client.post("/recommend/cold_start", json={})
        assert response.status_code == 400

    def test_cold_start_with_genres(self, client):
        """Test cold start with genre preferences."""
        response = client.post(
            "/recommend/cold_start", json={"genres": ["Action", "Sci-Fi"], "top_k": 5}
        )
        # Should return 200 or 404 (if no matching movies)
        assert response.status_code in [200, 404]


@pytest.fixture
def client():
    """Create test client fixture."""
    from main import app

    return TestClient(app)
