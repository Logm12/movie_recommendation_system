from functools import lru_cache
from typing import Optional
from fastapi import Depends
from repositories import MovieRepository, VectorRepository
from services import RecommendationService, ExplanationService, ContentSearchService
from config import get_settings, Settings
from sentence_transformers import SentenceTransformer

# Global state for the single instance
_service_instance: Optional[RecommendationService] = None
_explanation_service_instance: Optional[ExplanationService] = None
_search_service: Optional[ContentSearchService] = None
_sbert_model: Optional[SentenceTransformer] = None


def get_sbert_model() -> SentenceTransformer:
    """Load SBERT model lazily."""
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


def get_search_service(
    settings: Settings = Depends(get_settings),
) -> ContentSearchService:
    """Get Content Search Service."""
    global _search_service
    if _search_service is None:
        model = get_sbert_model()
        from qdrant_client import QdrantClient

        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        _search_service = ContentSearchService(client, model)
    return _search_service


@lru_cache()
def get_movie_repository() -> MovieRepository:
    return MovieRepository()


# Removed lru_cache because settings might not be hashable
def get_vector_repository(
    settings: Settings = Depends(get_settings),
) -> VectorRepository:
    return VectorRepository()


# Removed lru_cache because settings might not be hashable
def get_recommendation_service(
    movie_repo: MovieRepository = Depends(get_movie_repository),
    vector_repo: VectorRepository = Depends(get_vector_repository),
    settings: Settings = Depends(get_settings),
) -> RecommendationService:
    """
    Get the singleton instance of RecommendationService.
    Initializes it if not already created.
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = RecommendationService(movie_repo, vector_repo, settings)
    return _service_instance


from services.explanation_service import ExplanationService

_explanation_service_instance: Optional[ExplanationService] = None


def get_explanation_service() -> ExplanationService:
    """Dependency to get the ExplanationService instance."""
    global _explanation_service_instance
    if _explanation_service_instance is None:
        _explanation_service_instance = ExplanationService()
    return _explanation_service_instance
