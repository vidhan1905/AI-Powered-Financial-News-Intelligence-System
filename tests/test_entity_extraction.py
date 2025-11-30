"""Tests for entity extraction."""

import pytest

from src.agents.entity_extraction_agent import entity_extraction_agent
from src.core.state import create_article_state
from src.services.ner_service import get_ner_service


def test_entity_extraction_companies():
    """Test extraction of company entities."""
    article = {
        "title": "HDFC Bank and ICICI Bank Report Strong Growth",
        "content": "HDFC Bank and ICICI Bank have reported strong quarterly results. Both banks showed improved asset quality.",
        "source": "Test",
    }

    state = create_article_state(article)
    result = entity_extraction_agent(state)

    entities = result.get("entities", [])
    company_entities = [e for e in entities if e.get("entity_type") == "companies"]

    assert len(company_entities) > 0


def test_entity_extraction_regulators():
    """Test extraction of regulator entities."""
    article = {
        "title": "RBI Announces New Policy",
        "content": "The Reserve Bank of India (RBI) has announced new monetary policy measures.",
        "source": "Test",
    }

    state = create_article_state(article)
    result = entity_extraction_agent(state)

    entities = result.get("entities", [])
    regulator_entities = [e for e in entities if e.get("entity_type") == "regulators"]

    assert len(regulator_entities) > 0


def test_ner_service():
    """Test NER service directly."""
    ner_service = get_ner_service()
    text = "HDFC Bank and ICICI Bank are major banks in India. The RBI regulates the banking sector."

    entities = ner_service.extract_entities(text)

    assert "companies" in entities
    assert "regulators" in entities
    assert len(entities["companies"]) > 0

