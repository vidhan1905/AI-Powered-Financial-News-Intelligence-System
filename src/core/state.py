"""State management for LangGraph agents."""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    """State structure for the multi-agent system."""

    # News processing state
    article: Optional[Dict[str, Any]]  # Current article being processed
    article_id: Optional[int]  # Database ID of the article
    is_duplicate: bool  # Whether the article is a duplicate
    duplicate_of_id: Optional[int]  # ID of the original article if duplicate

    # Entity extraction state
    entities: List[Dict[str, Any]]  # Extracted entities

    # Stock impact state
    stock_impacts: List[Dict[str, Any]]  # Stock impact mappings

    # Query state
    query: Optional[str]  # User query
    query_entities: Optional[List[Dict[str, Any]]]  # Entities extracted from query
    query_type: Optional[str]  # Type of query (company, sector, regulator, theme)

    # Results state
    results: Optional[List[Dict[str, Any]]]  # Query results

    # Metadata
    errors: List[str]  # List of errors encountered
    metadata: Dict[str, Any]  # Additional metadata


def create_initial_state() -> AgentState:
    """Create an initial empty state.

    Returns:
        Initial AgentState.
    """
    return AgentState(
        article=None,
        article_id=None,
        is_duplicate=False,
        duplicate_of_id=None,
        entities=[],
        stock_impacts=[],
        query=None,
        query_entities=None,
        query_type=None,
        results=None,
        errors=[],
        metadata={},
    )


def create_article_state(article: Dict[str, Any]) -> AgentState:
    """Create state for processing a new article.

    Args:
        article: Article dictionary with title, content, source, etc.

    Returns:
        Initialized AgentState for article processing.
    """
    return AgentState(
        article=article,
        article_id=None,
        is_duplicate=False,
        duplicate_of_id=None,
        entities=[],
        stock_impacts=[],
        query=None,
        query_entities=None,
        query_type=None,
        results=None,
        errors=[],
        metadata={},
    )


def create_query_state(query: str) -> AgentState:
    """Create state for processing a query.

    Args:
        query: User query string.

    Returns:
        Initialized AgentState for query processing.
    """
    return AgentState(
        article=None,
        article_id=None,
        is_duplicate=False,
        duplicate_of_id=None,
        entities=[],
        stock_impacts=[],
        query=query,
        query_entities=None,
        query_type=None,
        results=[],
        errors=[],
        metadata={},
    )

