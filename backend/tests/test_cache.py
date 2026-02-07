"""
Tests for Redis caching functionality.

Following @test-driven-development: Write test first, watch it fail, then implement.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestRedisCache:
    """Tests for the Redis cache decorator and cache module."""

    def test_cache_returns_cached_value_on_hit(self):
        """Cache should return cached value without calling the function."""
        # Arrange
        from cache import get_redis_client, cache_result

        mock_redis = MagicMock()
        mock_redis.get.return_value = b'[{"id": 1, "title": "Cached Movie"}]'

        call_count = 0

        @cache_result(key_prefix="rec", ttl_seconds=300)
        def get_recommendations(user_id: int):
            nonlocal call_count
            call_count += 1
            return [{"id": 2, "title": "Fresh Movie"}]

        # Act - Patch redis client
        with patch("cache._redis_client", mock_redis):
            result = get_recommendations(123)

        # Assert
        assert call_count == 0  # Function should NOT be called on cache hit
        assert result == [{"id": 1, "title": "Cached Movie"}]
        mock_redis.get.assert_called_once_with("rec:123")

    def test_cache_stores_and_returns_fresh_value_on_miss(self):
        """Cache miss should call function, store result, and return it."""
        from cache import cache_result

        mock_redis = MagicMock()
        mock_redis.get.return_value = None  # Cache miss

        @cache_result(key_prefix="rec", ttl_seconds=300)
        def get_recommendations(user_id: int):
            return [{"id": 2, "title": "Fresh Movie"}]

        with patch("cache._redis_client", mock_redis):
            result = get_recommendations(456)

        # Assert
        assert result == [{"id": 2, "title": "Fresh Movie"}]
        mock_redis.get.assert_called_once_with("rec:456")
        mock_redis.setex.assert_called_once()

    def test_cache_handles_redis_connection_error_gracefully(self):
        """If Redis is down, function should still work (graceful degradation)."""
        from cache import cache_result
        from redis.exceptions import ConnectionError

        mock_redis = MagicMock()
        mock_redis.get.side_effect = ConnectionError("Redis unavailable")

        @cache_result(key_prefix="rec", ttl_seconds=300)
        def get_recommendations(user_id: int):
            return [{"id": 3, "title": "Fallback Movie"}]

        with patch("cache._redis_client", mock_redis):
            result = get_recommendations(789)

        # Assert - Should fallback to fresh computation
        assert result == [{"id": 3, "title": "Fallback Movie"}]
