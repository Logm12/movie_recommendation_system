"""
Pydantic models for API request/response schemas.
"""

from pydantic import BaseModel, field_validator
from typing import List, Optional


class Movie(BaseModel):
    """Movie data model with metadata."""

    id: int
    title: str
    genres: str
    poster_url: Optional[str] = None
    score: float = 0.0


class RecommendationResponse(BaseModel):
    """Response model for recommendation endpoints."""

    user_id: int
    recommendations: List[Movie]
    ab_group: str = "control"


# Valid genres for validation
VALID_GENRES = {
    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "film-noir",
    "horror",
    "musical",
    "mystery",
    "romance",
    "sci-fi",
    "thriller",
    "war",
    "western",
}


class ColdStartRequest(BaseModel):
    """Request model for cold start/guest recommendations."""

    selected_movie_ids: List[int] = []
    genres: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    query: Optional[str] = None  # Natural Language Query
    top_k: int = 10

    @field_validator("genres", mode="before")
    @classmethod
    def normalize_genres(cls, v):
        """Normalize genres to title case and filter valid ones."""
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        # Clean and normalize
        cleaned = []
        for g in v:
            if isinstance(g, str) and g.strip():
                normalized = g.strip().title()
                # Handle special cases
                if normalized.lower() == "sci-fi":
                    normalized = "Sci-Fi"
                elif normalized.lower() == "film-noir":
                    normalized = "Film-Noir"
                cleaned.append(normalized)
        return cleaned

    @field_validator("keywords", mode="before")
    @classmethod
    def clean_keywords(cls, v):
        """Clean and filter keywords."""
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        return [kw.strip() for kw in v if isinstance(kw, str) and kw.strip()]

    @field_validator("selected_movie_ids", mode="before")
    @classmethod
    def clean_movie_ids(cls, v):
        """Clean and filter movie IDs."""
        if not v:
            return []
        return [mid for mid in v if isinstance(mid, int) and mid > 0]

    @field_validator("top_k", mode="before")
    @classmethod
    def validate_top_k(cls, v):
        """Ensure top_k is within reasonable bounds."""
        if v is None:
            return 10
        return max(1, min(50, int(v)))

    def has_valid_input(self) -> bool:
        """Check if request has at least one valid input."""
        return bool(self.selected_movie_ids or self.genres or self.keywords)
