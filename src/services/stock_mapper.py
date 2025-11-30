"""Stock mapper service for mapping companies to stock symbols."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.core.config import settings
from src.utils.text_processing import normalize_company_name, normalize_entity_type

logger = logging.getLogger(__name__)


class StockMapper:
    """Service for mapping company names to stock symbols."""

    def __init__(self, mappings_file: Optional[str] = None):
        """Initialize the stock mapper.

        Args:
            mappings_file: Path to JSON file with stock mappings. If None, uses default.
        """
        if mappings_file is None:
            # Default to data/stock_mappings.json
            project_root = Path(__file__).parent.parent.parent
            mappings_file = project_root / "data" / "stock_mappings.json"

        self.mappings_file = Path(mappings_file)
        self.company_to_symbol: Dict[str, str] = {}
        self.sector_to_stocks: Dict[str, List[str]] = {}
        self.regulator_to_sectors: Dict[str, List[str]] = {}
        self._load_mappings()

    def _load_mappings(self) -> None:
        """Load stock mappings from JSON file."""
        if not self.mappings_file.exists():
            logger.warning(f"Mappings file not found: {self.mappings_file}. Creating default.")
            self._create_default_mappings()
            return

        try:
            with open(self.mappings_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load company to symbol mappings
            self.company_to_symbol = data.get("companies", {})
            self.sector_to_stocks = data.get("sectors", {})
            self.regulator_to_sectors = data.get("regulators", {})

            logger.info(f"Loaded {len(self.company_to_symbol)} company mappings")
        except Exception as e:
            logger.error(f"Error loading mappings: {e}")
            self._create_default_mappings()

    def _create_default_mappings(self) -> None:
        """Create default stock mappings for Indian stocks."""
        # Default Indian stock mappings
        self.company_to_symbol = {
            "HDFC Bank": "HDFCBANK",
            "HDFC": "HDFCBANK",
            "ICICI Bank": "ICICIBANK",
            "ICICI": "ICICIBANK",
            "State Bank of India": "SBIN",
            "SBI": "SBIN",
            "Reliance Industries": "RELIANCE",
            "Reliance": "RELIANCE",
            "RIL": "RELIANCE",
            "Infosys": "INFY",
            "TCS": "TCS",
            "Tata Consultancy Services": "TCS",
            "Wipro": "WIPRO",
            "Bharti Airtel": "BHARTIARTL",
            "Airtel": "BHARTIARTL",
            "ITC": "ITC",
            "Hindustan Unilever": "HINDUNILVR",
            "HUL": "HINDUNILVR",
            "Axis Bank": "AXISBANK",
            "Kotak Mahindra Bank": "KOTAKBANK",
            "Kotak Bank": "KOTAKBANK",
        }

        # Sector to stocks mapping
        self.sector_to_stocks = {
            "Banking": [
                "HDFCBANK",
                "ICICIBANK",
                "SBIN",
                "AXISBANK",
                "KOTAKBANK",
                "INDUSINDBK",
                "PNB",
                "BANKBARODA",
            ],
            "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM"],
            "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "AUROPHARMA"],
            "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO"],
            "Oil": ["RELIANCE", "ONGC", "IOC", "BPCL", "HPCL"],
            "Telecom": ["BHARTIARTL", "IDEA"],
            "Retail": ["RELIANCE", "DMART"],
            "Steel": ["TATASTEEL", "JSWSTEEL", "SAIL"],
            "Cement": ["ULTRACEMCO", "SHREECEM", "ACC"],
            "Power": ["NTPC", "POWERGRID", "TATAPOWER"],
        }

        # Regulator to sectors mapping
        self.regulator_to_sectors = {
            "RBI": ["Banking", "Financial Services"],
            "SEBI": ["All Sectors"],  # SEBI regulates all listed companies
            "IRDA": ["Insurance"],
            "TRAI": ["Telecom"],
        }

        # Save default mappings
        self._save_mappings()

    def _save_mappings(self) -> None:
        """Save mappings to JSON file."""
        try:
            self.mappings_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "companies": self.company_to_symbol,
                "sectors": self.sector_to_stocks,
                "regulators": self.regulator_to_sectors,
            }
            with open(self.mappings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving mappings: {e}")

    def map_company_to_symbol(self, company_name: str) -> Optional[str]:
        """Map company name to stock symbol.

        Args:
            company_name: Company name to map.

        Returns:
            Stock symbol if found, None otherwise.
        """
        if not company_name:
            return None

        # Normalize company name
        normalized_name = normalize_company_name(company_name)

        # Direct lookup
        if normalized_name in self.company_to_symbol:
            return self.company_to_symbol[normalized_name]

        # Case-insensitive lookup
        normalized_lower = normalized_name.lower()
        for key, value in self.company_to_symbol.items():
            if key.lower() == normalized_lower:
                return value

        # Partial match (e.g., "HDFC Bank Ltd" -> "HDFC Bank")
        for key, value in self.company_to_symbol.items():
            key_lower = key.lower()
            if key_lower in normalized_lower or normalized_lower in key_lower:
                # Prefer longer matches
                if len(key) > 3:  # Avoid very short partial matches
                    return value

        # Try matching with original name as fallback
        company_lower = company_name.lower()
        for key, value in self.company_to_symbol.items():
            key_lower = key.lower()
            if key_lower in company_lower or company_lower in key_lower:
                if len(key) > 3:
                    return value

        return None

    def get_stocks_for_sector(self, sector: str) -> List[str]:
        """Get all stock symbols for a given sector.

        Args:
            sector: Sector name.

        Returns:
            List of stock symbols.
        """
        if not sector:
            return []

        # Normalize sector name - remove common suffixes
        sector_clean = sector.replace(" sector", "").replace(" Sector", "").strip()
        sector_key = sector_clean.title()

        # Try exact match
        stocks = self.sector_to_stocks.get(sector_key, [])

        # If no match, try case-insensitive lookup
        if not stocks:
            sector_lower = sector_key.lower()
            for key, value in self.sector_to_stocks.items():
                if key.lower() == sector_lower:
                    return value

        return stocks

    def get_sectors_for_regulator(self, regulator: str) -> List[str]:
        """Get sectors affected by a regulator.

        Args:
            regulator: Regulator name (e.g., "RBI").

        Returns:
            List of sector names.
        """
        regulator_key = regulator.upper()
        return self.regulator_to_sectors.get(regulator_key, [])

    def map_entity_to_stocks(
        self, entity_type: str, entity_value: str
    ) -> List[Tuple[str, float, str]]:
        """Map an entity to stock symbols with confidence.

        Args:
            entity_type: Type of entity (company/companies, sector/sectors, regulator/regulators).
            entity_value: Value of the entity.

        Returns:
            List of tuples (stock_symbol, confidence, impact_type).
        """
        results = []

        # Normalize entity type to handle both plural and singular forms
        normalized_type = normalize_entity_type(entity_type)

        if normalized_type == "company":
            symbol = self.map_company_to_symbol(entity_value)
            if symbol:
                results.append((symbol, 1.0, "direct"))
            else:
                logger.debug(f"Could not map company '{entity_value}' to stock symbol")

        elif normalized_type == "sector":
            stocks = self.get_stocks_for_sector(entity_value)
            if stocks:
                for stock in stocks:
                    results.append((stock, 0.7, "sector"))  # Default sector confidence
            else:
                logger.debug(f"Could not find stocks for sector '{entity_value}'")

        elif normalized_type == "regulator":
            # Handle regulator name variations
            regulator_upper = entity_value.upper()
            regulator_lower = entity_value.lower()
            
            # Try exact match first
            sectors = self.get_sectors_for_regulator(entity_value)
            
            # If no match, try common variations
            if not sectors:
                if "rbi" in regulator_lower or "reserve" in regulator_lower:
                    sectors = self.get_sectors_for_regulator("RBI")
                elif "sebi" in regulator_lower:
                    sectors = self.get_sectors_for_regulator("SEBI")
                elif "irda" in regulator_lower:
                    sectors = self.get_sectors_for_regulator("IRDA")
                elif "trai" in regulator_lower:
                    sectors = self.get_sectors_for_regulator("TRAI")

            for sector in sectors:
                if sector == "All Sectors":
                    # If regulator affects all sectors, get all stocks
                    all_stocks = []
                    for sector_stocks in self.sector_to_stocks.values():
                        all_stocks.extend(sector_stocks)
                    for stock in set(all_stocks):
                        results.append((stock, 0.6, "regulatory"))
                else:
                    stocks = self.get_stocks_for_sector(sector)
                    for stock in stocks:
                        results.append((stock, 0.6, "regulatory"))

        return results


# Global instance
_stock_mapper: Optional[StockMapper] = None


def get_stock_mapper() -> StockMapper:
    """Get or create the global stock mapper instance."""
    global _stock_mapper
    if _stock_mapper is None:
        _stock_mapper = StockMapper()
    return _stock_mapper

