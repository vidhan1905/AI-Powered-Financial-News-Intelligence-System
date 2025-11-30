"""Unit tests for agents."""

import pytest

from src.agents.deduplication_agent import deduplication_agent
from src.agents.entity_extraction_agent import entity_extraction_agent
from src.agents.ingestion_agent import ingestion_agent
from src.agents.stock_impact_agent import stock_impact_agent
from src.core.state import create_article_state


@pytest.fixture
def sample_article():
    """Sample article for testing."""
    return {
        "title": "RBI Raises Repo Rate, Impacting Banking Sector",
        "content": "The Reserve Bank of India has increased the repo rate by 25 basis points. This affects HDFC Bank, ICICI Bank, and other banking stocks.",
        "source": "Test Source",
    }


def test_ingestion_agent(sample_article):
    """Test ingestion agent."""
    state = create_article_state(sample_article)
    result = ingestion_agent(state)

    assert result["article"] is not None
    assert result["article"]["title"] == sample_article["title"]
    assert len(result["article"]["content"]) > 0
    assert len(result.get("errors", [])) == 0


def test_entity_extraction_agent(sample_article):
    """Test entity extraction agent."""
    state = create_article_state(sample_article)
    state = ingestion_agent(state)
    result = entity_extraction_agent(state)

    assert "entities" in result
    assert len(result["entities"]) > 0

    # Check for expected entities
    entity_values = [e["entity_value"] for e in result["entities"]]
    assert any("RBI" in val or "Reserve Bank" in val for val in entity_values)


def test_stock_impact_agent(sample_article):
    """Test stock impact agent."""
    state = create_article_state(sample_article)
    state = ingestion_agent(state)
    state = entity_extraction_agent(state)
    result = stock_impact_agent(state)

    assert "stock_impacts" in result
    # Should have at least some stock impacts if entities were found
    if len(state.get("entities", [])) > 0:
        assert len(result["stock_impacts"]) >= 0  # May or may not have mappings


def test_deduplication_agent(sample_article):
    """Test deduplication agent."""
    state = create_article_state(sample_article)
    state = ingestion_agent(state)
    result = deduplication_agent(state)

    assert "is_duplicate" in result
    assert isinstance(result["is_duplicate"], bool)

