"""LLM service abstraction for OpenAI GPT-4o-mini."""

import logging
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM interactions using OpenAI."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the LLM service.

        Args:
            api_key: OpenAI API key. If None, uses settings.openai_api_key.
            model: Model name. If None, uses settings.openai_model.
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self.llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            temperature=0.0,  # Deterministic for entity extraction
        )
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate text using the LLM.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text.
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    async def generate_async(self, prompt: str, max_tokens: int = 1000) -> str:
        """Async version of generate.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text.
        """
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error generating async text: {e}")
            raise

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text using LLM.

        Args:
            text: Input text to extract entities from.

        Returns:
            Dictionary with entity types as keys and lists of entities as values.
        """
        prompt = f"""Extract financial entities from the following news article. 
Return a JSON object with the following structure:
{{
    "companies": ["list of company names"],
    "sectors": ["list of sectors"],
    "regulators": ["list of regulatory bodies"],
    "people": ["list of people mentioned"],
    "events": ["list of events"]
}}

News article:
{text}

Return only valid JSON, no additional text."""

        try:
            response = self.generate(prompt)
            # Parse JSON response
            import json

            # Clean response if it has markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            entities = json.loads(response)
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            # Return empty structure on error
            return {
                "companies": [],
                "sectors": [],
                "regulators": [],
                "people": [],
                "events": [],
            }

    def classify(self, text: str, categories: List[str]) -> str:
        """Classify text into one of the given categories.

        Args:
            text: Input text to classify.
            categories: List of possible categories.

        Returns:
            Selected category.
        """
        categories_str = ", ".join(categories)
        prompt = f"""Classify the following text into one of these categories: {categories_str}

Text: {text}

Return only the category name, nothing else."""

        try:
            response = self.generate(prompt, max_tokens=50)
            category = response.strip()
            # Validate category
            if category in categories:
                return category
            # Try to find partial match
            for cat in categories:
                if cat.lower() in category.lower() or category.lower() in cat.lower():
                    return cat
            return categories[0]  # Default to first category
        except Exception as e:
            logger.error(f"Error classifying text: {e}")
            return categories[0] if categories else "unknown"


# Global instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service

