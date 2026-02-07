import pytest
from unittest.mock import MagicMock, patch

# from services.search_service import ContentSearchService # This will fail initially (Red)


def test_search_service_integrity():
    """
    Test that ContentSearchService correctly:
    1. Embeds the query using the model.
    2. Queries Qdrant.
    3. Returns list of movie IDs.
    """
    # Mock Qdrant
    mock_qdrant = MagicMock()
    mock_model = MagicMock()

    # Mock search results
    mock_point = MagicMock()
    mock_point.payload = {"movie_id": 123}
    mock_qdrant.search.return_value = [mock_point]

    # Mock embedding
    mock_model.encode.return_value = [0.1, 0.2, 0.3]

    # We expect this import to fail or class to be missing
    from services.search_service import ContentSearchService

    service = ContentSearchService(
        qdrant_client=mock_qdrant, model=mock_model, collection_name="movies_content"
    )

    results = service.search(query="cyberpunk anime", limit=5)

    # Assertions
    assert len(results) == 1
    assert results[0] == 123
    mock_model.encode.assert_called_with("cyberpunk anime")
    mock_qdrant.search.assert_called_once()
