"""Configuration management for the application."""

import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Database Configuration
    sql_database_url: str = "postgresql+asyncpg://user:password@localhost:5432/financial_news"
    vector_db_path: str = "./data/chroma_db"

    # Application Configuration
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Similarity Thresholds
    deduplication_threshold: float = 0.90
    query_similarity_threshold: float = 0.65  # Lowered to get more query results

    # Confidence Scores
    direct_mention_confidence: float = 1.0
    sector_impact_confidence_min: float = 0.6
    sector_impact_confidence_max: float = 0.8
    regulatory_impact_confidence_min: float = 0.5
    regulatory_impact_confidence_max: float = 0.9

    @property
    def database_dir(self) -> str:
        """Get the directory for database files (for SQLite compatibility if used)."""
        # PostgreSQL is the primary database choice. This property is kept for SQLite compatibility
        # if someone wants to use SQLite instead
        if "sqlite" in self.sql_database_url:
            db_path = os.path.dirname(self.sql_database_url.replace("sqlite+aiosqlite:///", ""))
            if not os.path.exists(db_path):
                os.makedirs(db_path, exist_ok=True)
            return db_path
        return "./data"  # Default fallback

    @property
    def vector_db_dir(self) -> str:
        """Get the directory for vector database."""
        db_dir = os.path.dirname(self.vector_db_path) or "."
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        return db_dir


# Global settings instance
settings = Settings()

