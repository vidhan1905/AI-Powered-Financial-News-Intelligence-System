"""Entity Extraction Agent for extracting structured entities from news."""

import logging
from typing import Dict, List

from src.core.state import AgentState
from src.services.llm_service import get_llm_service
from src.services.ner_service import get_ner_service
from src.utils.text_processing import normalize_entity_type

logger = logging.getLogger(__name__)


def entity_extraction_agent(state: AgentState) -> AgentState:
    """Extract entities from the article.

    Args:
        state: Current agent state.

    Returns:
        Updated state with extracted entities.
    """
    try:
        article = state.get("article")
        if not article:
            state["errors"].append("No article provided to entity extraction agent")
            return state

        content = f"{article['title']}\n{article['content']}"

        # Use NER service for initial extraction
        ner_service = get_ner_service()
        entities_dict = ner_service.extract_entities(content)

        # Convert to structured format and normalize entity types
        entities = []
        entity_set = set()  # Track unique entities to avoid duplicates
        for entity_type, entity_values in entities_dict.items():
            # Normalize entity type to singular form
            normalized_type = normalize_entity_type(entity_type)
            for entity_value in entity_values:
                if entity_value:  # Skip empty values
                    entity_key = (normalized_type, entity_value.lower().strip())
                    if entity_key not in entity_set:
                        entity_set.add(entity_key)
                        entities.append(
                            {
                                "entity_type": normalized_type,
                                "entity_value": entity_value,
                                "confidence": 0.9,  # Default confidence for NER
                            }
                        )

        # Always use LLM for better entity extraction (especially for financial entities)
        # LLM is better at recognizing company names, financial terms, etc.
        logger.debug("Using LLM for entity extraction to improve accuracy")
        llm_service = get_llm_service()
        llm_entities = llm_service.extract_entities(content)
        for entity_type, entity_values in llm_entities.items():
            # Normalize entity type to singular form
            normalized_type = normalize_entity_type(entity_type)
            for entity_value in entity_values:
                if entity_value:
                    entity_key = (normalized_type, entity_value.lower().strip())
                    # Only add if not already present (LLM results take precedence for companies)
                    if entity_key not in entity_set:
                        entity_set.add(entity_key)
                        entities.append(
                            {
                                "entity_type": normalized_type,
                                "entity_value": entity_value,
                                "confidence": 0.95,  # Higher confidence for LLM (better accuracy)
                            }
                        )
                    elif normalized_type == "company":
                        # For companies, prefer LLM results (update existing if found)
                        for i, e in enumerate(entities):
                            if (e["entity_type"] == normalized_type and 
                                e["entity_value"].lower().strip() == entity_value.lower().strip()):
                                entities[i]["confidence"] = 0.95  # Update confidence
                                break

        state["entities"] = entities
        logger.info(f"Extracted {len(entities)} entities from article")

        return state

    except Exception as e:
        logger.error(f"Error in entity extraction agent: {e}")
        state["errors"].append(f"Entity extraction error: {str(e)}")
        state["entities"] = []
        return state

