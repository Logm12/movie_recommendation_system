from qdrant_client import QdrantClient

# from sentence_transformers import SentenceTransformer # Use dependency injection or lazy load
from typing import List


class ContentSearchService:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        model,
        collection_name: str = "movies_content",
    ):
        """
        Service for content-based semantic search.

        Args:
            qdrant_client: Connected Qdrant Client
            model: SentenceTransformer model (or mock) having .encode()
            collection_name: Name of Qdrant collection
        """
        self.client = qdrant_client
        self.model = model
        self.collection_name = collection_name

    def search(self, query: str, limit: int = 10) -> List[int]:
        """
        Search for movies similar to the query string using embedding.
        """
        # Encode query
        vector = self.model.encode(query)
        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        # Search Qdrant
        # client.search() is deprecated in newer versions, use query_points()
        results = self.client.query_points(
            collection_name=self.collection_name, query=vector, limit=limit
        ).points

        # Extract Movie IDs
        return [point.payload["movie_id"] for point in results]
