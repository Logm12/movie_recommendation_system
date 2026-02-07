"""
PostgreSQL repository for movie metadata operations.
"""

import psycopg2
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

from config import get_settings


class MovieRepository:
    """Repository for movie metadata in PostgreSQL."""

    def __init__(self):
        self.settings = get_settings()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = psycopg2.connect(
            dbname=self.settings.postgres_db,
            user=self.settings.postgres_user,
            password=self.settings.postgres_password,
            host=self.settings.postgres_host,
            port=self.settings.postgres_port,
        )
        try:
            yield conn
        finally:
            conn.close()

    def get_movies_by_ids(self, movie_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Fetch movie metadata for given IDs.

        Args:
            movie_ids: List of movie IDs to fetch

        Returns:
            Dict mapping movie_id to movie data
        """
        if not movie_ids:
            return {}

        with self._get_connection() as conn:
            cur = conn.cursor()
            format_strings = ",".join(["%s"] * len(movie_ids))
            cur.execute(
                f"SELECT movie_id, title, genres, poster_url FROM movies WHERE movie_id IN ({format_strings})",
                tuple(movie_ids),
            )
            rows = cur.fetchall()
            cur.close()

        return {
            row[0]: {
                "id": row[0],
                "title": row[1],
                "genres": row[2],
                "poster_url": row[3],
            }
            for row in rows
        }

    def find_movies_by_criteria(
        self, genres: List[str] = None, keywords: List[str] = None, limit: int = 20
    ) -> List[int]:
        """
        Find movie IDs matching genres or keywords.

        Args:
            genres: List of genre strings to match
            keywords: List of keywords to search in titles
            limit: Maximum number of results

        Returns:
            List of matching movie IDs
        """
        conditions = []
        params = []

        for genre in genres or []:
            conditions.append("genres ILIKE %s")
            params.append(f"%{genre}%")

        for kw in keywords or []:
            conditions.append("title ILIKE %s")
            params.append(f"%{kw}%")

        if not conditions:
            return []

        with self._get_connection() as conn:
            cur = conn.cursor()
            query = (
                f"SELECT movie_id FROM movies WHERE {' OR '.join(conditions)} LIMIT %s"
            )
            cur.execute(query, tuple(params) + (limit,))
            rows = cur.fetchall()
            cur.close()

        return [row[0] for row in rows]


# Singleton instance
_movie_repository: Optional[MovieRepository] = None


def get_movie_repository() -> MovieRepository:
    """Get or create movie repository instance."""
    global _movie_repository
    if _movie_repository is None:
        _movie_repository = MovieRepository()
    return _movie_repository
