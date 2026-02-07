"""
Configuration module for VDT GraphRec Backend.
Centralizes all environment variables and settings.
"""

import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    postgres_db: str = "movies_db"
    postgres_user: str = "admin"
    postgres_password: str = "password"
    postgres_host: str = "localhost"
    postgres_port: str = "5432"

    # Qdrant Vector DB
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "movies"

    # Model
    embeddings_path: str = "../user_embeddings.pt"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
