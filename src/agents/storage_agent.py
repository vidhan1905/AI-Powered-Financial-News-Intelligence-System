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

        # Skip storage if duplicate, but verify the original exists in SQL DB
        if state.get("is_duplicate", False):
            duplicate_of_id = state.get("duplicate_of_id")
            
            # Verify the duplicate article exists in SQL DB
            # If it doesn't exist, treat as new article and store it
            def verify_and_set_duplicate():
                """Verify duplicate exists in SQL DB."""
                import asyncio
                from src.database.sql_db import SQLDatabase
                
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    sql_db = SQLDatabase()
                    # Check if article exists
                    article = new_loop.run_until_complete(sql_db.get_article(duplicate_of_id))
                    if article:
                        logger.info(f"Skipping storage for duplicate article (duplicate of {duplicate_of_id})")
                        state["article_id"] = duplicate_of_id
                    else:
                        logger.warning(
                            f"Duplicate article {duplicate_of_id} not found in SQL DB. "
                            f"Removing orphaned embedding and treating as new article."
                        )
                        # Remove orphaned embedding from vector DB
                        vector_db = get_vector_db()
                        try:
                            vector_db.delete_by_id(duplicate_of_id)
                            logger.debug(f"Removed orphaned embedding for article {duplicate_of_id}")
                        except Exception as e:
                            logger.warning(f"Could not remove orphaned embedding: {e}")
                        
                        state["is_duplicate"] = False
                        state["duplicate_of_id"] = None
                except Exception as e:
                    logger.warning(f"Error verifying duplicate article: {e}. Treating as new article.")
                    state["is_duplicate"] = False
                    state["duplicate_of_id"] = None
                finally:
                    new_loop.close()
            
            import threading
            thread = threading.Thread(target=verify_and_set_duplicate)
            thread.start()
            thread.join()
            
            # If still marked as duplicate, return early
            if state.get("is_duplicate", False):
                return state

        # Run async storage operations
        # Since LangGraph agents are sync but we need async DB operations,
        # we'll use a thread pool to run the async code
        import concurrent.futures
        import threading

        def run_async_in_thread():
            """Run async function in a new event loop in a thread."""
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(_store_article_async(state))
            finally:
                new_loop.close()

        # Run in a separate thread to avoid event loop conflicts
        thread = threading.Thread(target=run_async_in_thread)
        thread.start()
        thread.join()
        
        # Longer delay to ensure transaction is committed and visible across connection pools
        import time
        time.sleep(0.5)  # Increased delay for cross-connection visibility

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

    # Create new database connection for this event loop
    from src.database.sql_db import SQLDatabase
    sql_db = SQLDatabase()  # Create new instance for this event loop
    vector_db = get_vector_db()
    embedding_service = get_embedding_service()

    try:
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
        # Note: We don't close the database connection here as it's a new instance
        # The connection pool will handle cleanup. The transaction is already committed.
    except Exception as e:
        logger.error(f"Error storing article in async function: {e}")
        raise

