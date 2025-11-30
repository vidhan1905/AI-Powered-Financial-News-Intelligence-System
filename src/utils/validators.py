"""Input validation utilities."""

from typing import Any, Dict, List, Optional


def validate_news_article(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate news article data.

    Args:
        data: Dictionary with article data.

    Returns:
        Tuple of (is_valid, error_message).
    """
    required_fields = ["title", "content"]
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"

    if not isinstance(data["title"], str) or len(data["title"].strip()) == 0:
        return False, "Title must be a non-empty string"

    if not isinstance(data["content"], str) or len(data["content"].strip()) == 0:
        return False, "Content must be a non-empty string"

    return True, None


def validate_query(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate query data.

    Args:
        data: Dictionary with query data.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if "query" not in data:
        return False, "Missing required field: query"

    if not isinstance(data["query"], str) or len(data["query"].strip()) == 0:
        return False, "Query must be a non-empty string"

    return True, None


def validate_entity(entity: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate entity data.

    Args:
        entity: Dictionary with entity data.

    Returns:
        Tuple of (is_valid, error_message).
    """
    required_fields = ["entity_type", "entity_value"]
    for field in required_fields:
        if field not in entity:
            return False, f"Missing required field: {field}"

    valid_types = ["company", "sector", "regulator", "person", "event"]
    if entity["entity_type"].lower() not in valid_types:
        return False, f"Invalid entity_type. Must be one of: {valid_types}"

    return True, None


def validate_stock_impact(impact: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate stock impact data.

    Args:
        impact: Dictionary with impact data.

    Returns:
        Tuple of (is_valid, error_message).
    """
    required_fields = ["stock_symbol", "confidence", "impact_type"]
    for field in required_fields:
        if field not in impact:
            return False, f"Missing required field: {field}"

    if not isinstance(impact["confidence"], (int, float)) or not (0.0 <= impact["confidence"] <= 1.0):
        return False, "Confidence must be a float between 0.0 and 1.0"

    valid_types = ["direct", "sector", "regulatory"]
    if impact["impact_type"].lower() not in valid_types:
        return False, f"Invalid impact_type. Must be one of: {valid_types}"

    return True, None

