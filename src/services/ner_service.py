"""NER service using spaCy for entity extraction."""

import logging
from typing import Dict, List, Optional

try:
    import spacy
    from spacy import displacy
except ImportError:
    spacy = None

logger = logging.getLogger(__name__)


class NERService:
    """Service for Named Entity Recognition using spaCy."""

    def __init__(self, model_name: str = "en_core_web_lg"):
        """Initialize the NER service.

        Args:
            model_name: spaCy model name. Defaults to en_core_web_lg for better accuracy.
        """
        if spacy is None:
            raise ImportError(
                "spaCy is not installed. Install it with: uv pip install spacy && python -m spacy download en_core_web_lg"
            )

        # Try to load the model, with fallbacks
        models_to_try = [model_name, "en_core_web_sm", "en_core_web_md"]
        self.nlp = None
        
        for model in models_to_try:
            try:
                self.nlp = spacy.load(model)
                logger.info(f"Loaded spaCy model: {model}")
                break
            except (OSError, IOError) as e:
                if model == models_to_try[-1]:
                    # Last model failed, log warning but don't raise
                    # The service will use LLM fallback instead
                    logger.warning(
                        f"spaCy models not found. Tried: {models_to_try}. "
                        f"Entity extraction will use LLM fallback. "
                        f"To install: uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
                    )
                    self.nlp = None
                else:
                    logger.warning(f"Model {model} not found, trying next...")
                    continue

        # Financial entity patterns
        self._setup_financial_patterns()

    def _setup_financial_patterns(self) -> None:
        """Setup custom patterns for financial entities."""
        # Skip if model not loaded
        if self.nlp is None:
            return
            
        # Add patterns for Indian financial terms
        patterns = [
            # RBI variations
            {"label": "REGULATOR", "pattern": [{"LOWER": {"IN": ["rbi", "reserve", "bank"]}}]},
            # Stock exchanges
            {"label": "ORG", "pattern": [{"LOWER": {"IN": ["nse", "bse", "bombay", "national"]}}]},
        ]

        # Add patterns to the pipeline if not already present
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(patterns)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text.

        Args:
            text: Input text to extract entities from.

        Returns:
            Dictionary with entity types as keys and lists of entities as values.
        """
        # If model not loaded, return empty entities (will use LLM fallback)
        if self.nlp is None:
            logger.warning("spaCy model not available, returning empty entities (LLM will be used)")
            return {
                "companies": [],
                "sectors": [],
                "regulators": [],
                "people": [],
                "events": [],
            }
        
        doc = self.nlp(text)

        entities = {
            "companies": [],
            "sectors": [],
            "regulators": [],
            "people": [],
            "events": [],
        }

        # Map spaCy labels to our entity types
        label_mapping = {
            "ORG": "companies",
            "PERSON": "people",
            "GPE": "companies",  # Sometimes companies are tagged as GPE
            "EVENT": "events",
        }

        # Extract entities
        for ent in doc.ents:
            entity_text = ent.text.strip()
            label = ent.label_

            # Map to our categories
            if label in label_mapping:
                category = label_mapping[label]
                if entity_text and entity_text not in entities[category]:
                    entities[category].append(entity_text)
            elif label == "REGULATOR" or "rbi" in entity_text.lower() or "sebi" in entity_text.lower():
                if entity_text not in entities["regulators"]:
                    entities["regulators"].append(entity_text)

        # Extract sectors using keyword matching
        sectors = self._extract_sectors(text)
        entities["sectors"].extend(sectors)

        return entities

    def _extract_sectors(self, text: str) -> List[str]:
        """Extract sector mentions from text.

        Args:
            text: Input text.

        Returns:
            List of sectors mentioned.
        """
        sector_keywords = {
            "Banking": ["banking", "bank", "financial services", "lending", "financial sector"],
            "IT": ["it", "information technology", "software", "tech", "technology sector"],
            "Pharma": ["pharma", "pharmaceutical", "drug", "medicine", "pharmaceutical sector"],
            "Auto": ["automobile", "auto", "vehicle", "car", "automotive"],
            "Oil": ["oil", "petroleum", "refinery", "crude", "oil and gas"],
            "Telecom": ["telecom", "telecommunication", "mobile", "network", "telecommunications"],
            "Retail": ["retail", "retailer", "store", "shopping", "retail sector"],
            "Steel": ["steel", "iron", "metal", "steel sector"],
            "Cement": ["cement", "construction", "cement sector"],
            "Power": ["power", "energy", "electricity", "power sector"],
            "FMCG": ["fmcg", "fast moving consumer goods", "consumer goods"],
        }

        text_lower = text.lower()
        found_sectors = []

        for sector, keywords in sector_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                if sector not in found_sectors:
                    found_sectors.append(sector)

        return found_sectors

    def get_entities_with_confidence(
        self, text: str
    ) -> List[Dict[str, any]]:
        """Extract entities with confidence scores.

        Args:
            text: Input text.

        Returns:
            List of entity dictionaries with text, label, and confidence.
        """
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 0.9,  # spaCy doesn't provide confidence, use default
                }
            )

        return entities


# Global instance
_ner_service: Optional[NERService] = None


def get_ner_service() -> NERService:
    """Get or create the global NER service instance."""
    global _ner_service
    if _ner_service is None:
        _ner_service = NERService()
    return _ner_service

