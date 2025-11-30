"""Vector database service using ChromaDB."""

import logging
import uuid
from typing import List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.config import settings

logger = logging.getLogger(__name__)


class VectorDatabase:
    """Service for vector database operations using ChromaDB."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the vector database.

        Args:
            db_path: Path to ChromaDB. If None, uses settings.vector_db_path.
        """
        self.db_path = db_path or settings.vector_db_path
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="news_articles",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Initialized ChromaDB at {self.db_path}")

    def add_news(
        self,
        article_id: int,
        title: str,
        content: str,
        embedding: List[float],
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a news article with its embedding.

        Args:
            article_id: Article ID.
            title: Article title.
            content: Article content.
            embedding: Article embedding vector.
            metadata: Optional metadata dictionary.
        """
        try:
            doc_id = str(article_id)
            document = f"{title}\n{content}"

            if metadata is None:
                metadata = {}

            metadata["article_id"] = article_id
            metadata["title"] = title

            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata],
            )
            logger.debug(f"Added article {article_id} to vector DB")
        except Exception as e:
            logger.error(f"Error adding news to vector DB: {e}")
            raise

    def search_similar(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        threshold: float = 0.85,
        where: Optional[dict] = None,
    ) -> List[Tuple[int, float, dict]]:
        """Search for similar articles.

        Args:
            query_embedding: Query embedding vector.
            n_results: Number of results to return.
            threshold: Minimum similarity threshold.
            where: Optional metadata filter.

        Returns:
            List of tuples (article_id, similarity_score, metadata).
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
            )

            similar_articles = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i, doc_id in enumerate(results["ids"][0]):
                    article_id = int(doc_id)
                    # ChromaDB returns distances, convert to similarity (1 - distance for cosine)
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    similarity = 1.0 - distance  # Cosine distance to similarity

                    if similarity >= threshold:
                        metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                        similar_articles.append((article_id, similarity, metadata))
                    else:
                        logger.debug(f"Article {article_id} below threshold: {similarity:.3f} < {threshold}")

            return similar_articles
        except Exception as e:
            logger.error(f"Error searching vector DB: {e}")
            return []

    def get_by_id(self, article_id: int) -> Optional[dict]:
        """Get article embedding and metadata by ID.

        Args:
            article_id: Article ID.

        Returns:
            Dictionary with embedding and metadata, or None.
        """
        try:
            results = self.collection.get(ids=[str(article_id)])
            if results["ids"] and len(results["ids"]) > 0:
                return {
                    "id": results["ids"][0],
                    "embedding": results["embeddings"][0] if results["embeddings"] else None,
                    "document": results["documents"][0] if results["documents"] else None,
                    "metadata": results["metadatas"][0] if results["metadatas"] else {},
                }
            return None
        except Exception as e:
            logger.error(f"Error getting article from vector DB: {e}")
            return None

    def delete_by_id(self, article_id: int) -> None:
        """Delete article from vector DB.

        Args:
            article_id: Article ID.
        """
        try:
            self.collection.delete(ids=[str(article_id)])
            logger.debug(f"Deleted article {article_id} from vector DB")
        except Exception as e:
            logger.error(f"Error deleting article from vector DB: {e}")

    def get_collection_size(self) -> int:
        """Get the number of articles in the collection.

        Returns:
            Number of articles.
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting collection size: {e}")
            return 0


# Global instance
_vector_db: Optional[VectorDatabase] = None


def get_vector_db() -> VectorDatabase:
    """Get or create the global vector database instance."""
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDatabase()
    return _vector_db

