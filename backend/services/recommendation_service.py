"""
Recommendation service containing business logic.
"""

from typing import List, Optional
import torch
import random
import numpy as np

from models import Movie
from repositories.movie_repository import MovieRepository
from repositories.vector_repository import VectorRepository
from config import Settings
from logger import logger
from cache import cache_result


class RecommendationService:
    """Service for generating movie recommendations."""

    def __init__(
        self,
        movie_repo: MovieRepository,
        vector_repo: VectorRepository,
        settings: Settings,
    ):
        self.settings = settings
        self.movie_repo = movie_repo
        self.vector_repo = vector_repo
        self._user_embeddings = None
        self._user_encoder = None

    def load_embeddings(self) -> bool:
        """
        Load user embeddings from file.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            data = torch.load(
                self.settings.embeddings_path,
                map_location=torch.device("cpu"),
                weights_only=False,
            )
            self._user_embeddings = data["user_embeddings"]
            self._user_encoder = data["user_encoder"]
            return True
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False

    @property
    def is_ready(self) -> bool:
        """Check if service is ready to serve requests."""
        return self._user_embeddings is not None and self._user_encoder is not None

    def get_user_vector(self, user_id: int) -> Optional[List[float]]:
        """
        Get embedding vector for a user.

        Args:
            user_id: External user ID

        Returns:
            User embedding vector or None if not found
        """
        if not self.is_ready:
            return None

        if user_id not in self._user_encoder.classes_:
            return None

        internal_idx = self._user_encoder.transform([user_id])[0]
        return self._user_embeddings[internal_idx].tolist()

    def get_known_user_ids(self) -> List[int]:
        """Get list of all known user IDs."""
        if not self.is_ready:
            return []
        return list(self._user_encoder.classes_)

    @cache_result(key_prefix="rec", ttl_seconds=300)
    def recommend_for_user(
        self, user_id: int, top_k: int = 10, ab_group: str = "control"
    ) -> List[Movie]:
        """
        Generate recommendations for a known user.

        Experimental Variants:
        - "control": Standard LightGCN (Optimized for precision)
        - "treatment": Discovery Mode (Slightly randomized/diverse - placeholder for now)
        """
        try:
            # Check for B-Variant logic
            if ab_group == "treatment":
                # FUTURE: Implement specific Logic for B (e.g. Diversity re-ranking)
                # For now, we still return standard results but log it.
                logger.debug(
                    f"Serving recommendation for Variant B (Discovery Mode) to user {user_id}"
                )

            # 1. Get User Vector
            user_vector = self.get_user_vector(
                user_id
            )  # Changed from self.movie_repo.get_user_vector
            if user_vector is None:
                # Fallback to cold start if no vector found
                logger.warning(
                    f"User {user_id} has no vector. Falling back to cold start."
                )
                return self.recommend_cold_start(top_k=top_k)

            # 2. Search in Vector DB
            # Note: We query more candidates than needed if we want to re-rank for diversity later
            search_k = top_k * 2 if ab_group == "treatment" else top_k

            recommendations = self.vector_repo.search_similar(
                user_vector, limit=search_k
            )  # Changed from search_similar_movies

            # If "treatment", we could shuffle or filter here.
            # adhering to "Just-In-Time" (Kaizen), we start simple.
            if ab_group == "treatment":
                # Simple example: Take the top K * 2 and randomly sample K to increase diversity
                if len(recommendations) > top_k:
                    recommendations = random.sample(recommendations, top_k)

            # Extract IDs and scores
            movie_ids = [r["movie_id"] for r in recommendations]
            scores = {r["movie_id"]: r["score"] for r in recommendations}

            # Enrich with metadata
            return self._enrich_movies(movie_ids, scores)

        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            # Graceful degradation
            return []

    def recommend_cold_start(
        self,
        selected_movie_ids: Optional[List[int]] = None,  # Changed to Optional
        genres: List[str] = None,
        keywords: List[str] = None,
        query: str = None,
        search_service=None,  # ContentSearchService
        top_k: int = 10,
    ) -> List[Movie]:
        """
        Cold start recommendation using vector arithmetic and optional Neural Search.
        """
        selected_movie_ids = (
            selected_movie_ids if selected_movie_ids is not None else []
        )  # Ensure it's a list

        # 0. Neural Search (Phase 4.2)
        if query and search_service:
            logger.info(f"Performing Neural Search for: '{query}'")
            try:
                search_ids = search_service.search(query, limit=5)
                # Add these as if they were selected movies
                selected_movie_ids.extend(
                    search_ids
                )  # Use extend to add to existing list
                logger.info(f"Neural Search found: {search_ids}")
            except Exception as e:
                logger.error(f"Neural Search failed: {e}")

        # 1. Fetch vectors for selected movies
        selected_vectors = []
        if selected_movie_ids:
            selected_vectors = self.vector_repo.get_vectors_by_ids(selected_movie_ids)

        # Determine candidate pool and target vector
        if selected_vectors:
            target_vector = np.mean(selected_vectors, axis=0)
            # Search for similar movies using the target vector
            results = self.vector_repo.search_similar(
                target_vector, limit=top_k * 2
            )  # Search more to filter out selected

            # Filter out already selected movies and enrich
            recommended_movie_ids = [
                r["movie_id"]
                for r in results
                if r["movie_id"] not in selected_movie_ids
            ]
            scores = {r["movie_id"]: r["score"] for r in results}

            # Apply genre filtering if specified
            enriched_movies = self._enrich_movies(recommended_movie_ids, scores)
            if genres:
                enriched_movies = self._rank_by_genre_similarity(
                    enriched_movies, genres
                )

            return enriched_movies[:top_k]

        # Fallback if no selected movies or neural search results, but genres/keywords are provided
        candidates = set()
        if genres or keywords:
            candidates_list = self.movie_repo.find_movies_by_criteria(
                genres=genres, keywords=keywords, limit=50
            )
            candidates.update(candidates_list)

        if candidates:
            # Just return metadata for candidates, sorted by some default (e.g., popularity if available, or just as is)
            # For now, we'll just get the metadata for the first `top_k` candidates
            candidate_movie_ids = list(candidates)[:top_k]
            return list(self.movie_repo.get_movies_by_ids(candidate_movie_ids).values())

        return []

    def _get_seed_movie_ids(
        self,
        movie_ids: List[int] = None,
        genres: List[str] = None,
        keywords: List[str] = None,
    ) -> List[int]:
        """Get seed movie IDs from explicit IDs and/or criteria matching."""
        seed_ids = set(movie_ids or [])

        if genres or keywords:
            matched = self.movie_repo.find_movies_by_criteria(
                genres=genres, keywords=keywords, limit=50
            )
            seed_ids.update(matched)

        return list(seed_ids)

    def _find_similar_movies(self, seed_ids: List[int], limit: int) -> List[Movie]:
        """Find movies similar to seed movies using vector search."""
        vectors = self.vector_repo.get_vectors_by_ids(seed_ids)
        if not vectors:
            return []

        centroid = self.vector_repo.compute_centroid(vectors)
        results = self.vector_repo.search_similar(centroid, limit=limit)

        movie_ids = [r["movie_id"] for r in results]
        scores = {r["movie_id"]: r["score"] for r in results}

        return self._enrich_movies(movie_ids, scores)

    def _rank_by_genre_similarity(
        self, movies: List[Movie], target_genres: List[str]
    ) -> List[Movie]:
        """
        Rank movies by Jaccard similarity of genres.
        Score = intersection / union

        Args:
            movies: List of candidate movies
            target_genres: User's selected genres

        Returns:
            List of movies sorted by Jaccard score (desc) then specific score (desc)
        """
        if not target_genres:
            return movies

        target_set = {g.lower() for g in target_genres}
        ranked_movies = []

        for movie in movies:
            movie_genres = {g.strip().lower() for g in movie.genres.split("|")}
            intersection = len(movie_genres & target_set)
            union = len(movie_genres | target_set)

            jaccard_score = intersection / union if union > 0 else 0.0

            if intersection > 0:  # Only keep matches
                ranked_movies.append(
                    {
                        "movie": movie,
                        "jaccard": jaccard_score,
                        "original_score": movie.score,
                    }
                )

        # Sort by Jaccard score (desc), then by original vector score (desc)
        ranked_movies.sort(
            key=lambda x: (x["jaccard"], x["original_score"]), reverse=True
        )

        return [item["movie"] for item in ranked_movies]

    def _enrich_movies(self, movie_ids: List[int], scores: dict) -> List[Movie]:
        """
        Enrich movie IDs with metadata from database.

        Args:
            movie_ids: Ordered list of movie IDs
            scores: Dict mapping movie_id to similarity score

        Returns:
            List of Movie objects with metadata
        """
        movie_data = self.movie_repo.get_movies_by_ids(movie_ids)

        movies = []
        for mid in movie_ids:
            if mid in movie_data:
                data = movie_data[mid]
                movies.append(
                    Movie(
                        id=data["id"],
                        title=data["title"],
                        genres=data["genres"],
                        poster_url=data["poster_url"],
                        score=scores.get(mid, 0.0),
                    )
                )

        return movies
