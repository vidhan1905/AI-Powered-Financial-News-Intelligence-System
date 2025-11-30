"""Stock Impact Analysis Agent for mapping entities to stock symbols."""

import logging
from typing import Dict, List

from src.core.config import settings
from src.core.state import AgentState
from src.services.stock_mapper import get_stock_mapper

logger = logging.getLogger(__name__)


def stock_impact_agent(state: AgentState) -> AgentState:
    """Map entities to stock symbols with confidence scores.

    Args:
        state: Current agent state.

    Returns:
        Updated state with stock impact mappings.
    """
    try:
        entities = state.get("entities", [])
        if not entities:
            logger.warning("No entities to map to stocks")
            state["stock_impacts"] = []
            return state

        stock_mapper = get_stock_mapper()
        stock_impacts = []
        processed_stocks = set()  # Avoid duplicates

        for entity in entities:
            entity_type = entity["entity_type"]
            entity_value = entity["entity_value"]

            # Map entity to stocks
            stock_mappings = stock_mapper.map_entity_to_stocks(entity_type, entity_value)

            for stock_symbol, base_confidence, impact_type in stock_mappings:
                # Avoid duplicate entries for same stock
                stock_key = (stock_symbol, impact_type)
                if stock_key in processed_stocks:
                    continue

                # Adjust confidence based on impact type
                if impact_type == "direct":
                    confidence = settings.direct_mention_confidence
                elif impact_type == "sector":
                    # Use entity confidence to adjust sector confidence
                    confidence = (
                        settings.sector_impact_confidence_min
                        + (settings.sector_impact_confidence_max - settings.sector_impact_confidence_min)
                        * entity.get("confidence", 0.9)
                    )
                elif impact_type == "regulatory":
                    # Regulatory impact varies
                    confidence = (
                        settings.regulatory_impact_confidence_min
                        + (settings.regulatory_impact_confidence_max - settings.regulatory_impact_confidence_min)
                        * entity.get("confidence", 0.9)
                    )
                else:
                    confidence = base_confidence

                stock_impacts.append(
                    {
                        "stock_symbol": stock_symbol,
                        "confidence": round(confidence, 3),
                        "impact_type": impact_type,
                        "source_entity": entity_value,
                        "source_entity_type": entity_type,
                    }
                )
                processed_stocks.add(stock_key)

        # Sort by confidence (highest first)
        stock_impacts.sort(key=lambda x: x["confidence"], reverse=True)

        state["stock_impacts"] = stock_impacts
        logger.info(f"Mapped {len(stock_impacts)} stock impacts from {len(entities)} entities")

        return state

    except Exception as e:
        logger.error(f"Error in stock impact agent: {e}")
        state["errors"].append(f"Stock impact error: {str(e)}")
        state["stock_impacts"] = []
        return state

