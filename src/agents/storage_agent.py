"""Storage & Indexing Agent for storing processed articles."""

import asyncio
import logging

from src.core.state import AgentState
from src.database.sql_db import get_sql_db
from src.database.vector_db import get_vector_db
from src.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


def storage_agent(state: AgentState) -> AgentState:
    """Store article, entities, and stock impacts in databases.

    Args:
        state: Current agent state.

    Returns:
        Updated state with article_id.
    """
    try:
        article = state.get("article")
        if not article:
            state["errors"].append("No article provided to storage agent")
            return state

        # Skip storage if duplicate
        if state.get("is_duplicate", False):
            duplicate_of_id = state.get("duplicate_of_id")
            logger.info(f"Skipping storage for duplicate article (duplicate of {duplicate_of_id})")
            state["article_id"] = duplicate_of_id
            return state

        # Run async storage operations
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to use a different approach
                # Create a new thread-safe event loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _store_article_async(state))
                    future.result()
            else:
                loop.run_until_complete(_store_article_async(state))
        except RuntimeError:
            # No event loop exists, create a new one
            asyncio.run(_store_article_async(state))

        return state

    except Exception as e:
        logger.error(f"Error in storage agent: {e}")
        state["errors"].append(f"Storage error: {str(e)}")
        return state


async def _store_article_async(state: AgentState) -> None:
    """Async helper to store article in databases."""
    article = state["article"]
    entities = state.get("entities", [])
    stock_impacts = state.get("stock_impacts", [])

    sql_db = get_sql_db()
    vector_db = get_vector_db()
    embedding_service = get_embedding_service()

    # Store in SQL database
    article_id = await sql_db.store_article(
        title=article["title"],
        content=article["content"],
        source=article.get("source", "unknown"),
        is_duplicate=state.get("is_duplicate", False),
        duplicate_of_id=state.get("duplicate_of_id"),
    )

    state["article_id"] = article_id

    # Store entities
    if entities:
        await sql_db.store_entities(article_id, entities)

    # Store stock impacts
    if stock_impacts:
        await sql_db.store_stock_impacts(article_id, stock_impacts)

    # Generate embedding and store in vector DB
    article_text = f"{article['title']}\n{article['content']}"
    embedding = embedding_service.embed_text(article_text)

    vector_db.add_news(
        article_id=article_id,
        title=article["title"],
        content=article["content"],
        embedding=embedding,
        metadata={
            "source": article.get("source", "unknown"),
            "timestamp": article.get("timestamp").isoformat() if article.get("timestamp") else None,
        },
    )

    logger.info(f"Stored article {article_id} in databases")

