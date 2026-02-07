"""
Redis Caching Module for VDT GraphRec Pro

Provides a simple caching decorator for recommendation endpoints.
Following @observability-engineer and @kaizen principles:
- Graceful degradation if Redis is unavailable
- Structured logging for cache hits/misses
- Simple, minimal implementation (YAGNI)
"""

import json
import os
import functools
from typing import Optional, Callable, Any

import redis
from redis.exceptions import ConnectionError, TimeoutError

from logger import logger

# Global Redis client (lazy initialization)
_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> Optional[redis.Redis]:
    """
    Get or create Redis client.

    Lazily initializes the Redis connection on first use.
    Returns None if Redis is not configured or unreachable.
    """
    global _redis_client

    if _redis_client is None:
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))

        try:
            _redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                socket_timeout=2.0,  # Fast timeout for responsiveness
                socket_connect_timeout=2.0,
                decode_responses=False,  # We'll decode JSON ourselves
            )
            # Test connection
            _redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis unavailable: {e}. Caching disabled.")
            _redis_client = None

    return _redis_client


def cache_result(key_prefix: str, ttl_seconds: int = 300):
    """
    Decorator to cache function results in Redis.

    Args:
        key_prefix: Prefix for cache keys (e.g., "rec" -> "rec:123")
        ttl_seconds: Time-to-live in seconds (default: 5 minutes)

    Usage:
        @cache_result(key_prefix="rec", ttl_seconds=300)
        def get_recommendations(user_id: int):
            ...

    Behavior:
        - On cache hit: returns cached value without calling function
        - On cache miss: calls function, caches result, returns it
        - On Redis error: calls function directly (graceful degradation)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Build cache key from first positional argument (user_id, etc.)
            cache_key = f"{key_prefix}:{args[0]}" if args else f"{key_prefix}:default"

            client = _redis_client  # Use global directly for easier testing

            # Try to get from cache
            if client is not None:
                try:
                    cached = client.get(cache_key)
                    if cached is not None:
                        logger.info(f"Cache HIT for {cache_key}")
                        return json.loads(cached)
                    logger.debug(f"Cache MISS for {cache_key}")
                except (ConnectionError, TimeoutError) as e:
                    logger.warning(f"Redis read error: {e}")

            # Cache miss or Redis unavailable - compute fresh
            result = func(*args, **kwargs)

            # Try to store in cache
            if client is not None:
                try:
                    client.setex(cache_key, ttl_seconds, json.dumps(result))
                    logger.debug(f"Cached {cache_key} for {ttl_seconds}s")
                except (ConnectionError, TimeoutError) as e:
                    logger.warning(f"Redis write error: {e}")

            return result

        return wrapper

    return decorator


def invalidate_cache(key_prefix: str, key_id: Any) -> bool:
    """
    Invalidate a specific cache entry.

    Args:
        key_prefix: The prefix used in cache_result
        key_id: The ID portion of the key

    Returns:
        True if deleted, False if not found or error
    """
    client = get_redis_client()
    if client is None:
        return False

    cache_key = f"{key_prefix}:{key_id}"
    try:
        deleted = client.delete(cache_key)
        if deleted:
            logger.info(f"Invalidated cache: {cache_key}")
        return bool(deleted)
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Redis delete error: {e}")
        return False


# Export public API
__all__ = ["get_redis_client", "cache_result", "invalidate_cache"]
