"""Deduplication Agent for identifying duplicate news articles."""

import logging
from typing import List, Tuple

from src.core.config import settings
from src.core.state import AgentState
from src.database.vector_db import get_vector_db
from src.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


def deduplication_agent(state: AgentState) -> AgentState:
    """Check if article is a duplicate of existing articles.

    Args:
        state: Current agent state.

    Returns:
        Updated state with deduplication decision.
    """
    try:
        article = state.get("article")
        if not article:
            state["errors"].append("No article provided to deduplication agent")
            return state

        # Generate embedding for the article
        embedding_service = get_embedding_service()
        article_text = f"{article['title']}\n{article['content']}"
        embedding = embedding_service.embed_text(article_text)

        # Search for similar articles in vector DB
        vector_db = get_vector_db()
        threshold = settings.deduplication_threshold
        similar_articles = vector_db.search_similar(
            query_embedding=embedding,
            n_results=5,
            threshold=threshold,
        )

        if similar_articles:
            # Found similar article - mark as duplicate
            duplicate_of_id, similarity_score, _ = similar_articles[0]
            state["is_duplicate"] = True
            state["duplicate_of_id"] = duplicate_of_id
            logger.info(
                f"Article identified as duplicate of {duplicate_of_id} "
                f"(similarity: {similarity_score:.3f})"
            )
        else:
            # No similar article found - unique
            state["is_duplicate"] = False
            state["duplicate_of_id"] = None
            logger.info("Article is unique")

        return state

    except Exception as e:
        logger.error(f"Error in deduplication agent: {e}")
        state["errors"].append(f"Deduplication error: {str(e)}")
        # On error, assume unique to continue processing
        state["is_duplicate"] = False
        return state

