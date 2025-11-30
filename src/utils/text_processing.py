"""Text processing utilities."""

import re
from typing import Optional


def clean_text(text: str) -> str:
    """Clean and normalize text.

    Args:
        text: Input text.

    Returns:
        Cleaned text.
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^\w\s.,!?;:()\-'\"%$]", " ", text)

    # Strip whitespace
    text = text.strip()

    return text


def extract_title_from_content(content: str, max_length: int = 200) -> str:
    """Extract a title from content if title is not provided.

    Args:
        content: Article content.
        max_length: Maximum title length.

    Returns:
        Extracted title.
    """
    if not content:
        return "Untitled"

    # Take first sentence or first N characters
    sentences = content.split(".")
    if sentences:
        title = sentences[0].strip()
        if len(title) > max_length:
            title = title[:max_length].rsplit(" ", 1)[0] + "..."
        return title

    # Fallback to first N characters
    return content[:max_length].strip() + "..." if len(content) > max_length else content.strip()


def normalize_company_name(name: str) -> str:
    """Normalize company name for matching.

    Args:
        name: Company name.

    Returns:
        Normalized name.
    """
    if not name:
        return ""

    # Remove common suffixes
    name = re.sub(r"\s+(Ltd|Limited|Inc|Incorporated|Corp|Corporation|LLC)\s*\.?$", "", name, flags=re.IGNORECASE)

    # Remove extra whitespace
    name = re.sub(r"\s+", " ", name).strip()

    return name

