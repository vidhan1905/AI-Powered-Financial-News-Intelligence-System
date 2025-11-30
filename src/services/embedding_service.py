"""Embedding service using OpenAI's text-embedding-3-small model."""

import logging
from typing import List, Optional

import numpy as np
from openai import AsyncOpenAI, OpenAI

from src.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using OpenAI API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the embedding service.

        Args:
            api_key: OpenAI API key. If None, uses settings.openai_api_key.
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = settings.openai_embedding_model
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self._cache: dict[str, List[float]] = {}

    def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed.
            use_cache: Whether to use cached embeddings.

        Returns:
            Embedding vector as a list of floats (1536 dimensions).
        """
        if use_cache and text in self._cache:
            return self._cache[text]

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            embedding = response.data[0].embedding

            if use_cache:
                self._cache[text] = embedding

            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def embed_batch(self, texts: List[str], batch_size: int = 2048) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches.

        Args:
            texts: List of input texts to embed.
            batch_size: Maximum number of texts per batch (OpenAI limit: 2048).

        Returns:
            List of embedding vectors.
        """
        all_embeddings = []
        batch_size = min(batch_size, 2048)  # OpenAI's limit

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                # Cache embeddings
                for text, embedding in zip(batch, batch_embeddings):
                    self._cache[text] = embedding

            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                raise

        return all_embeddings

    async def embed_text_async(self, text: str, use_cache: bool = True) -> List[float]:
        """Async version of embed_text.

        Args:
            text: Input text to embed.
            use_cache: Whether to use cached embeddings.

        Returns:
            Embedding vector as a list of floats.
        """
        if use_cache and text in self._cache:
            return self._cache[text]

        try:
            response = await self.async_client.embeddings.create(
                model=self.model,
                input=text,
            )
            embedding = response.data[0].embedding

            if use_cache:
                self._cache[text] = embedding

            return embedding
        except Exception as e:
            logger.error(f"Error generating async embedding: {e}")
            raise

    async def embed_batch_async(
        self, texts: List[str], batch_size: int = 2048
    ) -> List[List[float]]:
        """Async version of embed_batch.

        Args:
            texts: List of input texts to embed.
            batch_size: Maximum number of texts per batch.

        Returns:
            List of embedding vectors.
        """
        all_embeddings = []
        batch_size = min(batch_size, 2048)

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = await self.async_client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                # Cache embeddings
                for text, embedding in zip(batch, batch_embeddings):
                    self._cache[text] = embedding

            except Exception as e:
                logger.error(f"Error generating async batch embeddings: {e}")
                raise

        return all_embeddings

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score between 0 and 1.
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()


# Global instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

