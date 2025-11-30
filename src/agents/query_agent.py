"""Query Processing Agent for handling user queries."""

import asyncio
import logging
from typing import Dict, List

from src.core.config import settings
from src.core.state import AgentState
from src.database.sql_db import get_sql_db
from src.database.vector_db import get_vector_db
from src.services.embedding_service import get_embedding_service
from src.services.ner_service import get_ner_service
from src.services.stock_mapper import get_stock_mapper

logger = logging.getLogger(__name__)


def query_agent(state: AgentState) -> AgentState:
    """Process user query and retrieve relevant news.

    Args:
        state: Current agent state with query.

    Returns:
        Updated state with query results.
    """
    try:
        query = state.get("query")
        if not query:
            state["errors"].append("No query provided to query agent")
            return state

        # Extract entities from query
        ner_service = get_ner_service()
        query_entities_dict = ner_service.extract_entities(query)
        query_entities = []
        for entity_type, entity_values in query_entities_dict.items():
            for entity_value in entity_values:
                query_entities.append({"entity_type": entity_type, "entity_value": entity_value})

        state["query_entities"] = query_entities

        # Determine query type
        query_type = _determine_query_type(query_entities)
        state["query_type"] = query_type

        # Run async query operations
        # Since LangGraph agents are sync but we need async DB operations,
        # we'll use a thread pool to run the async code
        import threading

        def run_async_in_thread():
            """Run async function in a new event loop in a thread."""
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(_process_query_async(state))
            finally:
                new_loop.close()

        # Run in a separate thread to avoid event loop conflicts
        thread = threading.Thread(target=run_async_in_thread)
        thread.start()
        thread.join()

        return state

    except Exception as e:
        logger.error(f"Error in query agent: {e}")
        state["errors"].append(f"Query error: {str(e)}")
        state["results"] = []
        return state


def _determine_query_type(query_entities: List[Dict]) -> str:
    """Determine the type of query.

    Args:
        query_entities: Entities extracted from query.

    Returns:
        Query type: company, sector, regulator, or theme.
    """
    entity_types = [e["entity_type"] for e in query_entities]

    if "companies" in entity_types:
        return "company"
    elif "sectors" in entity_types:
        return "sector"
    elif "regulators" in entity_types:
        return "regulator"
    else:
        return "theme"  # Semantic/thematic query


async def _process_query_async(state: AgentState) -> None:
    """Async helper to process query and retrieve results."""
    query = state["query"]
    query_entities = state.get("query_entities", [])
    query_type = state.get("query_type", "theme")

    # Create new database connection for this event loop
    from src.database.sql_db import SQLDatabase
    sql_db = SQLDatabase()  # Create new instance for this event loop
    vector_db = get_vector_db()
    embedding_service = get_embedding_service()
    stock_mapper = get_stock_mapper()

    results = []

    # Generate query embedding for semantic search
    query_embedding = embedding_service.embed_text(query)

    # Semantic search in vector DB
    similar_articles = vector_db.search_similar(
        query_embedding=query_embedding,
        n_results=20,
        threshold=settings.query_similarity_threshold,
    )

    # Get article details from SQL DB
    article_ids = [article_id for article_id, _, _ in similar_articles]
    articles_by_id = {}
    for article_id in article_ids:
        article = await sql_db.get_article(article_id)
        if article:
            articles_by_id[article_id] = article

    # Entity-based filtering and expansion
    if query_type == "company":
        # Expand to include sector news
        for entity in query_entities:
            if entity["entity_type"] == "companies":
                company_name = entity["entity_value"]
                # Get direct mentions
                direct_articles = await sql_db.get_articles_by_entity("companies", company_name)
                for article in direct_articles:
                    if article.id not in articles_by_id:
                        articles_by_id[article.id] = article

                # Get sector for company
                symbol = stock_mapper.map_company_to_symbol(company_name)
                if symbol:
                    # Get sector-wide news (would need reverse mapping)
                    # For now, include all banking/financial news if it's a bank
                    pass

    elif query_type == "sector":
        # Get all sector-related articles
        for entity in query_entities:
            if entity["entity_type"] == "sectors":
                sector = entity["entity_value"]
                sector_articles = await sql_db.get_articles_by_entity("sectors", sector)
                for article in sector_articles:
                    if article.id not in articles_by_id:
                        articles_by_id[article.id] = article

    elif query_type == "regulator":
        # Get regulator-specific articles
        for entity in query_entities:
            if entity["entity_type"] == "regulators":
                regulator = entity["entity_value"]
                regulator_articles = await sql_db.get_articles_by_entity("regulators", regulator)
                for article in regulator_articles:
                    if article.id not in articles_by_id:
                        articles_by_id[article.id] = article

    # Build results with relevance scores
    for article_id, similarity, _ in similar_articles:
        if article_id in articles_by_id:
            article = articles_by_id[article_id]
            # Get entities and stock impacts
            entities = await sql_db.get_entities(article_id)
            stock_impacts = await sql_db.get_stock_impacts(article_id)

            results.append(
                {
                    "article_id": article_id,
                    "title": article.title,
                    "content": article.content[:500] + "..." if len(article.content) > 500 else article.content,
                    "source": article.source,
                    "timestamp": article.timestamp.isoformat() if article.timestamp else None,
                    "similarity_score": round(similarity, 3),
                    "entities": [
                        {"type": e.entity_type, "value": e.entity_value} for e in entities
                    ],
                    "stock_impacts": [
                        {
                            "symbol": si.stock_symbol,
                            "confidence": si.confidence,
                            "type": si.impact_type,
                        }
                        for si in stock_impacts
                    ],
                }
            )

    # Sort by similarity score
    results.sort(key=lambda x: x["similarity_score"], reverse=True)

    state["results"] = results
    logger.info(f"Query returned {len(results)} results")

