import pytest
from starlette.testclient import TestClient
from unittest.mock import MagicMock
from main import app
from dependencies import get_movie_repository, get_recommendation_service

client = TestClient(app)


def test_health_check_healthy_flow():
    """Test that /health returns 200 when all services are healthy."""

    # Mock Repository
    mock_repo = MagicMock()
    # Mock Context Manager for _get_connection
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__.return_value = mock_cursor
    mock_conn.cursor.return_value = mock_cursor
    mock_repo._get_connection.return_value.__enter__.return_value = mock_conn

    # Mock Service
    mock_service = MagicMock()
    mock_service.is_ready = True
    mock_service.get_known_user_ids.return_value = [1, 2]

    # Override dependencies
    app.dependency_overrides[get_movie_repository] = lambda: mock_repo
    app.dependency_overrides[get_recommendation_service] = lambda: mock_service

    try:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["db_status"] == "connected"
    finally:
        app.dependency_overrides = {}


def test_health_check_db_failure_returns_503():
    """Test that /health returns 503 when DB fails."""
    mock_repo = MagicMock()
    # Mock Context Manager to succeed entering
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    # RAISE ON EXECUTE
    mock_cursor.execute.side_effect = Exception("DB Down")
    mock_cursor.__enter__.return_value = mock_cursor

    mock_conn.cursor.return_value = mock_cursor
    mock_repo._get_connection.return_value.__enter__.return_value = mock_conn

    mock_service = MagicMock()
    mock_service.is_ready = True

    app.dependency_overrides[get_movie_repository] = lambda: mock_repo
    app.dependency_overrides[get_recommendation_service] = lambda: mock_service

    try:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "unhealthy"
        assert response.json()["db_status"] == "disconnected"
    finally:
        app.dependency_overrides = {}
