from typing import List, Optional
from pydantic import BaseModel
import random


class ExplanationRequest(BaseModel):
    user_id: int
    movie_id: int
    movie_title: str
    movie_genres: str


class ExplanationResponse(BaseModel):
    explanation: str
    method: str  # "heuristic" or "genai"


class ExplanationService:
    """Service for generating explanations for recommendations."""

    def generate_explanation(self, request: ExplanationRequest) -> ExplanationResponse:
        """
        Generate a natural language explanation for why a movie was recommended.

        Current Strategy (Heuristic):
        - Analyze genres of the recommended movie.
        - Look at common patterns (hardcoded for MVP, ideally connected to user history).
        - Return a template-based string.
        """
        # Heuristic Logic
        genres = [g.strip() for g in request.movie_genres.split("|") if g.strip()]
        main_genre = genres[0] if genres else "movies"

        templates = [
            f"Because you seem to enjoy {main_genre} movies.",
            f"This is a top-rated {main_genre} film similar to others you've liked.",
            f"We thought you'd like this because it features elements of {', '.join(genres[:2]) if len(genres) > 1 else main_genre}.",
            f"Popular among fans of {main_genre}.",
            f"Since you like {main_genre}, '{request.movie_title}' might be right up your alley.",
            f"If you're in the mood for {main_genre}, this is a great pick.",
            f"Based on your taste, this {main_genre} movie is a strong match.",
            f"Fans of {main_genre} often enjoy this film.",
        ]

        explanation = random.choice(templates)

        return ExplanationResponse(explanation=explanation, method="heuristic")
