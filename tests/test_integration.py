"""Integration tests for end-to-end workflow."""

import pytest

from src.core.graph import get_news_processing_graph, get_query_graph
from src.core.state import create_article_state, create_query_state


@pytest.mark.asyncio
async def test_full_processing_pipeline():
    """Test the full news processing pipeline."""
    article = {
        "title": "RBI Raises Repo Rate, Impacting Banking Sector",
        "content": "The Reserve Bank of India has increased the repo rate by 25 basis points. This affects HDFC Bank, ICICI Bank, and other banking stocks.",
        "source": "Test Source",
    }

    graph = get_news_processing_graph()
    state = create_article_state(article)
    final_state = graph.invoke(state)

    assert final_state.get("article_id") is not None or final_state.get("is_duplicate", False)
    assert "entities" in final_state
    assert "stock_impacts" in final_state


@pytest.mark.asyncio
async def test_query_pipeline():
    """Test the query processing pipeline."""
    query = "What is the impact of RBI rate hike on banking stocks?"

    graph = get_query_graph()
    state = create_query_state(query)
    final_state = graph.invoke(state)

    assert "results" in final_state
    assert isinstance(final_state["results"], list)

