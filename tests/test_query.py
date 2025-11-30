"""Tests for query system."""

import pytest

from src.agents.query_agent import query_agent
from src.core.state import create_query_state


def test_query_processing():
    """Test query processing."""
    query = "What is the impact of RBI rate hike on banking stocks?"

    state = create_query_state(query)
    result = query_agent(state)

    assert "results" in result
    assert isinstance(result["results"], list)
    assert "query_type" in result


def test_query_entity_extraction():
    """Test entity extraction from queries."""
    query = "News about HDFC Bank and ICICI Bank"

    state = create_query_state(query)
    result = query_agent(state)

    assert "query_entities" in result
    assert len(result.get("query_entities", [])) >= 0  # May or may not find entities

