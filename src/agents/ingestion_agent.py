"""News Ingestion Agent for processing incoming news articles."""

import logging
from datetime import datetime
from typing import Any, Dict

from src.core.state import AgentState
from src.utils.text_processing import clean_text, extract_title_from_content
from src.utils.validators import validate_news_article

logger = logging.getLogger(__name__)


def ingestion_agent(state: AgentState) -> AgentState:
    """Process and normalize incoming news article.

    Args:
        state: Current agent state.

    Returns:
        Updated state with processed article.
    """
    try:
        article = state.get("article")
        if not article:
            state["errors"].append("No article provided to ingestion agent")
            return state

        # Validate article
        is_valid, error_msg = validate_news_article(article)
        if not is_valid:
            state["errors"].append(f"Invalid article: {error_msg}")
            return state

        # Clean and normalize
        title = clean_text(article.get("title", ""))
        content = clean_text(article.get("content", ""))

        # Extract title if not provided
        if not title:
            title = extract_title_from_content(content)

        # Extract metadata
        source = article.get("source", "unknown")
        timestamp = article.get("timestamp")
        if timestamp is None:
            timestamp = datetime.utcnow()
        elif isinstance(timestamp, str):
            # Try to parse timestamp string
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except Exception:
                timestamp = datetime.utcnow()

        # Create processed article
        processed_article = {
            "title": title,
            "content": content,
            "source": source,
            "timestamp": timestamp,
            "original_data": article,  # Keep original for reference
        }

        state["article"] = processed_article
        logger.info(f"Ingested article: {title[:50]}...")

        return state

    except Exception as e:
        logger.error(f"Error in ingestion agent: {e}")
        state["errors"].append(f"Ingestion error: {str(e)}")
        return state

