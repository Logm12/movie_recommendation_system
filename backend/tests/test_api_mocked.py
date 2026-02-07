import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from main import app
from dependencies import get_recommendation_service
from services import RecommendationService
from models import Movie


# Mock Service
class MockRecommendationService:
    def __init__(self):
        self.is_ready = True

    def get_known_user_ids(self):
        return [1, 2, 3]

    def get_user_vector(self, user_id):
        if user_id == 1:
            return [0.1, 0.2]
        return None

    def recommend_for_user(self, user_id, top_k=10, **kwargs):
        if user_id == 1:
            return [
                Movie(
                    id=101,
                    title="Test Movie",
                    genres="Action",
                    poster_url=None,
                    score=0.9,
                )
            ]
        return []

    def recommend_cold_start(
        self, selected_movie_ids=None, genres=None, keywords=None, top_k=10, **kwargs
    ):
        if "Action" in (genres or []):
            return [
                Movie(
                    id=102,
                    title="Action Movie",
                    genres="Action",
                    poster_url=None,
                    score=0.8,
                )
            ]
        return []


# Pytest Fixture to override dependency
@pytest.fixture
def client_mocked():
    # Setup Override
    mock_service = MockRecommendationService()
    app.dependency_overrides[get_recommendation_service] = lambda: mock_service

    with TestClient(app) as client:
        yield client

    # Teardown
    app.dependency_overrides.clear()


def test_health_mocked(client_mocked):
    response = client_mocked.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["known_users"] == 3


def test_recommend_success(client_mocked):
    response = client_mocked.get("/recommend/1")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert len(data["recommendations"]) == 1
    assert data["recommendations"][0]["title"] == "Test Movie"


def test_recommend_not_found(client_mocked):
    response = client_mocked.get("/recommend/999")
    # Service returns None for user 999 -> Endpoint raises 404
    assert response.status_code == 404


def test_cold_start_mocked(client_mocked):
    response = client_mocked.post(
        "/recommend/cold_start", json={"genres": ["Action"], "top_k": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["recommendations"][0]["title"] == "Action Movie"
