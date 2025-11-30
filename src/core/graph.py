"""LangGraph workflow definition for the multi-agent system."""

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from src.agents.deduplication_agent import deduplication_agent
from src.agents.entity_extraction_agent import entity_extraction_agent
from src.agents.ingestion_agent import ingestion_agent
from src.agents.query_agent import query_agent
from src.agents.stock_impact_agent import stock_impact_agent
from src.agents.storage_agent import storage_agent
from src.core.state import AgentState

logger = logging.getLogger(__name__)


def should_continue(state: AgentState) -> Literal["store", "skip"]:
    """Determine if processing should continue or skip.

    Args:
        state: Current agent state.

    Returns:
        "store" if article is unique, "skip" if duplicate.
    """
    if state.get("is_duplicate", False):
        return "skip"
    return "store"


def create_news_processing_graph() -> StateGraph:
    """Create the LangGraph workflow for news processing.

    Returns:
        Compiled StateGraph for news processing.
    """
    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes (agents)
    workflow.add_node("ingestion", ingestion_agent)
    workflow.add_node("deduplication", deduplication_agent)
    workflow.add_node("entity_extraction", entity_extraction_agent)
    workflow.add_node("stock_impact", stock_impact_agent)
    workflow.add_node("storage", storage_agent)

    # Set entry point
    workflow.set_entry_point("ingestion")

    # Add edges
    workflow.add_edge("ingestion", "deduplication")
    workflow.add_edge("deduplication", "entity_extraction")
    workflow.add_conditional_edges(
        "entity_extraction",
        should_continue,
        {
            "store": "stock_impact",
            "skip": "storage",  # Still store duplicate info
        },
    )
    workflow.add_edge("stock_impact", "storage")
    workflow.add_edge("storage", END)

    # Compile graph
    app = workflow.compile()

    return app


def create_query_graph() -> StateGraph:
    """Create the LangGraph workflow for query processing.

    Returns:
        Compiled StateGraph for query processing.
    """
    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("query", query_agent)

    # Set entry point
    workflow.set_entry_point("query")

    # Add edge to end
    workflow.add_edge("query", END)

    # Compile graph
    app = workflow.compile()

    return app


# Global graph instances
_news_graph = None
_query_graph = None


def get_news_processing_graph():
    """Get or create the news processing graph."""
    global _news_graph
    if _news_graph is None:
        _news_graph = create_news_processing_graph()
    return _news_graph


def get_query_graph():
    """Get or create the query graph."""
    global _query_graph
    if _query_graph is None:
        _query_graph = create_query_graph()
    return _query_graph

