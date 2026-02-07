"""
VDT GraphRec API - Main FastAPI Application

This module provides the REST API endpoints for the movie recommendation system.
It uses a clean architecture with separate layers for:
- Configuration (config.py)
- Models/Schemas (models/)
- Repositories (repositories/)
- Services (services/)
"""

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import get_settings, Settings
from models import Movie, RecommendationResponse, ColdStartRequest
from services import (
    RecommendationService,
    ExplanationService,
    ExplanationRequest,
    ExplanationResponse as RecommendationExplanationResponse,
    ContentSearchService,
)
from dependencies import (
    get_recommendation_service,
    get_movie_repository,
    get_vector_repository,
    get_explanation_service,
    get_search_service,
)
from logger import logger, configure_logging
from middleware.middleware import RequestIDMiddleware, ABTestingMiddleware
from middleware.error_handler import GlobalExceptionHandlerMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info("Starting VDT GraphRec API...")

    # Initialize singleton instance manually for startup
    settings = get_settings()
    repo = get_movie_repository()
    vec_repo = get_vector_repository(settings)

    # This initializes the internal singleton in dependencies.py
    service = get_recommendation_service(repo, vec_repo, settings)

    if service.load_embeddings():
        logger.info(f"Loaded embeddings for {len(service.get_known_user_ids())} users.")
    else:
        logger.warning("Failed to load embeddings. Some features may be unavailable.")

    yield

    # Shutdown
    logger.info("Shutting down VDT GraphRec API...")


def create_app() -> FastAPI:
    """Application factory for creating FastAPI instance."""
    # Configure structured logging early
    configure_logging(log_level="INFO", json_logs=False)

    settings = get_settings()

    app = FastAPI(
        title="Movie Recommendation System",
        version="2.0",
        description="Hybrid Movie Recommendation System using LightGCN and Vector Search",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID Middleware for tracing
    app.add_middleware(RequestIDMiddleware)

    return app


app = create_app()


# --- Dependency Injection ---
# Re-exporting for compatibility if needed, but better to use directly
# get_service = get_recommendation_service

# --- Health Check ---


@app.get("/", tags=["Health"])
def read_root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Movie Recommendation System Backend",
        "version": "2.0",
    }


@app.get("/health", tags=["Health"])
def health_check(
    response: Response,
    movie_repo=Depends(get_movie_repository),
    service: RecommendationService = Depends(get_recommendation_service),
):
    """
    Health check endpoint.
    Verifies Database and AI Engine connectivity.
    """
    health_status = {
        "status": "healthy",
        "embeddings_loaded": service.is_ready,
        "known_users": len(service.get_known_user_ids()) if service.is_ready else 0,
        "db_status": "connected",
        "qdrant_status": "connected",
    }

    # Check Database
    try:
        with movie_repo._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
    except Exception:
        health_status["db_status"] = "disconnected"
        health_status["status"] = "unhealthy"
        response.status_code = 503

    return health_status


# --- Recommendation Endpoints ---


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
)
def recommend(
    request: Request,
    user_id: int,
    top_k: int = 10,
    service: RecommendationService = Depends(get_recommendation_service),
):
    """
    Get personalized recommendations for a known user.

    Args:
        user_id: The user's ID from the training dataset
        top_k: Number of recommendations to return (default: 10)

    Returns:
        RecommendationResponse with user_id and list of recommended movies
    """
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Check if user exists
    if service.get_user_vector(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Get A/B Group from middleware
    ab_group = request.scope.get("ab_group", "control")

    try:
        movies = service.recommend_for_user(user_id, top_k=top_k, ab_group=ab_group)
        return RecommendationResponse(
            user_id=user_id, recommendations=movies, ab_group=ab_group
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")


@app.post(
    "/recommend/cold_start",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
)
def recommend_cold_start(
    request: ColdStartRequest,
    service: RecommendationService = Depends(get_recommendation_service),
    search_service: ContentSearchService = Depends(get_search_service),
):
    """
    Get recommendations for guest/new users using cold start approach.

    This endpoint uses vector arithmetic and optionally Neural Search (if query provided).

    Args:
        request: ColdStartRequest with preferences (including query)

    Returns:
        RecommendationResponse with user_id=0 (guest) and recommendations
    """
    # Log incoming request for debugging
    print(
        f"[Cold Start] Request received: genres={request.genres}, keywords={request.keywords}, query={request.query}, movie_ids={request.selected_movie_ids}"
    )

    # Use model's validation method
    if not request.has_valid_input():
        # Check if query is also missing (new field)
        if not request.query:
            print(f"[Cold Start] Validation failed: no valid input provided")
            raise HTTPException(
                status_code=400,
                detail="At least one of selected_movie_ids, genres, keywords, or query must be provided",
            )

    try:
        print(
            f"[Cold Start] Calling service with: genres={request.genres}, keywords={request.keywords}, query={request.query}"
        )
        movies = service.recommend_cold_start(
            selected_movie_ids=request.selected_movie_ids,
            genres=request.genres,
            keywords=request.keywords,
            query=request.query,
            search_service=search_service,
            top_k=request.top_k,
        )

        print(f"[Cold Start] Service returned {len(movies)} movies")

        if not movies:
            print(f"[Cold Start] No movies found for criteria")
            raise HTTPException(
                status_code=404,
                detail="No matching movies found for the given criteria",
            )

        return RecommendationResponse(user_id=0, recommendations=movies)
    except Exception as e:
        print(f"[Cold Start] Exception occurred: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Cold start recommendation failed", "message": str(e)},
        )


@app.post(
    "/recommend/explain",
    response_model=RecommendationExplanationResponse,
    tags=["Recommendations"],
)
def explain_recommendation(
    request: ExplanationRequest,
    service: ExplanationService = Depends(get_explanation_service),
):
    """
    Generate an explanation for why a movie was recommended.

    Currently uses heuristic logic (Phase 3.3).
    Future: Will use GenAI (LLM) if API key is provided.
    """
    try:
        return service.generate_explanation(request)
    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        raise HTTPException(status_code=500, detail="Could not generate explanation")


# --- Entry Point ---

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
