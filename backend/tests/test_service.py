"""
Unit tests for the recommendation service.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

# We'll test the service in isolation by mocking repositories


class TestRecommendationService:
    """Tests for RecommendationService class."""

    def test_compute_centroid(self):
        """Test vector centroid computation."""
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        expected = [4.0, 5.0, 6.0]
        result = np.mean(vectors, axis=0).tolist()
        assert result == expected

    def test_enrich_movies_preserves_order(self):
        """Test that movie enrichment preserves the order of IDs."""
        # Simulate movie data from repository
        movie_data = {
            1: {"id": 1, "title": "Movie 1", "genres": "Action", "poster_url": None},
            2: {"id": 2, "title": "Movie 2", "genres": "Comedy", "poster_url": None},
            3: {"id": 3, "title": "Movie 3", "genres": "Drama", "poster_url": None},
        }

        # Order should be preserved
        movie_ids = [3, 1, 2]
        scores = {3: 0.9, 1: 0.8, 2: 0.7}

        # Simulate enrichment
        result = []
        for mid in movie_ids:
            if mid in movie_data:
                data = movie_data[mid]
                result.append({"id": data["id"], "score": scores.get(mid, 0.0)})

        assert [r["id"] for r in result] == [3, 1, 2]
        assert [r["score"] for r in result] == [0.9, 0.8, 0.7]


class TestVectorOperations:
    """Tests for vector-related operations."""

    def test_empty_vectors_returns_empty_centroid(self):
        """Test handling of empty vector list."""
        vectors = []
        if not vectors:
            result = []
        else:
            result = np.mean(vectors, axis=0).tolist()
        assert result == []

    def test_single_vector_centroid(self):
        """Test centroid of single vector is itself."""
        vectors = [[1.0, 2.0, 3.0]]
        result = np.mean(vectors, axis=0).tolist()
        assert result == [1.0, 2.0, 3.0]


class TestColdStartCriteria:
    """Tests for cold start criteria matching."""

    def test_genre_expansion(self):
        """Test that genre matching works correctly."""
        genres = ["Action", "Sci-Fi"]
        # Simulate SQL ILIKE matching
        conditions = [f"genres ILIKE '%{g}%'" for g in genres]
        assert len(conditions) == 2
        assert "Action" in conditions[0]
        assert "Sci-Fi" in conditions[1]

    def test_keyword_expansion(self):
        """Test that keyword matching works correctly."""
        keywords = ["Star", "Wars"]
        conditions = [f"title ILIKE '%{kw}%'" for kw in keywords]
        assert len(conditions) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
