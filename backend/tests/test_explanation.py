import pytest
from services.explanation_service import ExplanationService, ExplanationRequest


def test_heuristic_explanation_generation():
    """Test that heuristic logic returns a valid explanation."""
    service = ExplanationService()
    request = ExplanationRequest(
        user_id=1,
        movie_id=100,
        movie_title="Inception",
        movie_genres="Action|Sci-Fi|Thriller",
    )

    response = service.generate_explanation(request)

    assert response.method == "heuristic"
    assert len(response.explanation) > 10
    assert "Action" in response.explanation or "Sci-Fi" in response.explanation


def test_explanation_handles_empty_genres():
    """Test fallback when no genres are provided."""
    service = ExplanationService()
    request = ExplanationRequest(
        user_id=1, movie_id=101, movie_title="Unknown Movie", movie_genres=""
    )

    response = service.generate_explanation(request)
    assert response.method == "heuristic"
    assert "movies" in response.explanation  # Default fallback
