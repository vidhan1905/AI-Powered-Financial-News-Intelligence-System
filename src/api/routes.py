"""API routes for the FastAPI application."""

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    NewsArticleInput,
    NewsArticleResponse,
    QueryRequest,
    QueryResponse,
)
from src.core.graph import get_news_processing_graph, get_query_graph
from src.core.state import create_article_state, create_query_state
from src.database.sql_db import get_sql_db
from src.database.vector_db import get_vector_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/news/ingest", response_model=NewsArticleResponse, status_code=201)
async def ingest_news(article_input: NewsArticleInput):
    """Ingest a news article through the processing pipeline.

    Args:
        article_input: News article input.

    Returns:
        Processed article with entities and stock impacts.
    """
    try:
        # Create initial state
        article_dict = {
            "title": article_input.title,
            "content": article_input.content,
            "source": article_input.source,
            "timestamp": article_input.timestamp,
        }
        state = create_article_state(article_dict)

        # Run through processing graph
        graph = get_news_processing_graph()
        final_state = graph.invoke(state)

        # Check for errors
        if final_state.get("errors"):
            raise HTTPException(
                status_code=500,
                detail=f"Processing errors: {', '.join(final_state['errors'])}",
            )

        article_id = final_state.get("article_id")
        if not article_id:
            raise HTTPException(status_code=500, detail="Failed to store article")

        # Get stored article
        sql_db = get_sql_db()
        article = await sql_db.get_article(article_id)

        if not article:
            raise HTTPException(status_code=404, detail="Article not found after storage")

        # Build response
        entities = await sql_db.get_entities(article_id)
        stock_impacts = await sql_db.get_stock_impacts(article_id)

        return NewsArticleResponse(
            article_id=article.id,
            title=article.title,
            content=article.content,
            source=article.source,
            timestamp=article.timestamp,
            is_duplicate=article.is_duplicate,
            duplicate_of_id=article.duplicate_of_id,
            entities=[
                {
                    "entity_type": e.entity_type,
                    "entity_value": e.entity_value,
                    "confidence": e.confidence,
                }
                for e in entities
            ],
            stock_impacts=[
                {
                    "stock_symbol": si.stock_symbol,
                    "confidence": si.confidence,
                    "impact_type": si.impact_type,
                }
                for si in stock_impacts
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting news: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def process_query(query_request: QueryRequest):
    """Process a user query and return relevant news.

    Args:
        query_request: Query request.

    Returns:
        Query response with relevant articles.
    """
    try:
        # Create query state
        state = create_query_state(query_request.query)

        # Run through query graph
        graph = get_query_graph()
        final_state = graph.invoke(state)

        # Check for errors
        if final_state.get("errors"):
            raise HTTPException(
                status_code=500,
                detail=f"Query errors: {', '.join(final_state['errors'])}",
            )

        results = final_state.get("results", [])

        return QueryResponse(
            query=query_request.query,
            query_type=final_state.get("query_type"),
            results_count=len(results),
            results=results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/news/{article_id}", response_model=NewsArticleResponse)
async def get_article(article_id: int):
    """Get an article by ID.

    Args:
        article_id: Article ID.

    Returns:
        Article details.
    """
    try:
        sql_db = get_sql_db()
        article = await sql_db.get_article(article_id)

        if not article:
            raise HTTPException(status_code=404, detail="Article not found")

        entities = await sql_db.get_entities(article_id)
        stock_impacts = await sql_db.get_stock_impacts(article_id)

        return NewsArticleResponse(
            article_id=article.id,
            title=article.title,
            content=article.content,
            source=article.source,
            timestamp=article.timestamp,
            is_duplicate=article.is_duplicate,
            duplicate_of_id=article.duplicate_of_id,
            entities=[
                {
                    "entity_type": e.entity_type,
                    "entity_value": e.entity_value,
                    "confidence": e.confidence,
                }
                for e in entities
            ],
            stock_impacts=[
                {
                    "stock_symbol": si.stock_symbol,
                    "confidence": si.confidence,
                    "impact_type": si.impact_type,
                }
                for si in stock_impacts
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting article: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/stocks/{symbol}/news", response_model=List[NewsArticleResponse])
async def get_stock_news(symbol: str, limit: int = 50):
    """Get news articles for a specific stock symbol.

    Args:
        symbol: Stock symbol.
        limit: Maximum number of articles to return.

    Returns:
        List of articles affecting the stock.
    """
    try:
        sql_db = get_sql_db()
        articles = await sql_db.get_articles_by_stock(symbol, limit=limit)

        results = []
        for article in articles:
            entities = await sql_db.get_entities(article.id)
            stock_impacts = await sql_db.get_stock_impacts(article.id)

            results.append(
                NewsArticleResponse(
                    article_id=article.id,
                    title=article.title,
                    content=article.content,
                    source=article.source,
                    timestamp=article.timestamp,
                    is_duplicate=article.is_duplicate,
                    duplicate_of_id=article.duplicate_of_id,
                    entities=[
                        {
                            "entity_type": e.entity_type,
                            "entity_value": e.entity_value,
                            "confidence": e.confidence,
                        }
                        for e in entities
                    ],
                    stock_impacts=[
                        {
                            "stock_symbol": si.stock_symbol,
                            "confidence": si.confidence,
                            "impact_type": si.impact_type,
                        }
                        for si in stock_impacts
                    ],
                )
            )

        return results

    except Exception as e:
        logger.error(f"Error getting stock news: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.

    Returns:
        Health status information.
    """
    try:
        vector_db = get_vector_db()
        vector_db_size = vector_db.get_collection_size()

        # Try to connect to SQL DB
        sql_db = get_sql_db()
        database_connected = True
        try:
            # Simple check - try to get session
            async with sql_db.get_session() as session:
                pass
        except Exception:
            database_connected = False

        return HealthResponse(
            status="healthy",
            version="0.1.0",
            database_connected=database_connected,
            vector_db_size=vector_db_size,
        )

    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return HealthResponse(
            status="unhealthy",
            version="0.1.0",
            database_connected=False,
            vector_db_size=0,
        )

