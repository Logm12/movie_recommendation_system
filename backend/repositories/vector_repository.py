"""
Qdrant vector database repository for embedding operations.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient

from config import get_settings


class VectorRepository:
    """Repository for vector operations in Qdrant."""

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        """Lazy initialization of Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                host=self.settings.qdrant_host, port=self.settings.qdrant_port
            )
        return self._client

    def search_similar(
        self, query_vector: List[float], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar movies using vector similarity.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results

        Returns:
            List of dicts with movie_id and score
        """
        result = self.client.query_points(
            collection_name=self.settings.qdrant_collection,
            query=query_vector,
            limit=limit,
        ).points

        return [
            {"movie_id": point.payload.get("movie_id"), "score": point.score}
            for point in result
        ]

    def get_vectors_by_ids(self, movie_ids: List[int]) -> List[List[float]]:
        """
        Retrieve vectors for given movie IDs.

        Args:
            movie_ids: List of movie IDs

        Returns:
            List of embedding vectors
        """
        if not movie_ids:
            return []

        points = self.client.retrieve(
            collection_name=self.settings.qdrant_collection,
            ids=movie_ids,
            with_vectors=True,
        )

        return [point.vector for point in points if point.vector]

    def compute_centroid(self, vectors: List[List[float]]) -> List[float]:
        """
        Compute the centroid (average) of multiple vectors.

        Args:
            vectors: List of embedding vectors

        Returns:
            Centroid vector
        """
        if not vectors:
            return []
        return np.mean(vectors, axis=0).tolist()


# Singleton instance
_vector_repository: Optional[VectorRepository] = None


def get_vector_repository() -> VectorRepository:
    """Get or create vector repository instance."""
    global _vector_repository
    if _vector_repository is None:
        _vector_repository = VectorRepository()
    return _vector_repository
