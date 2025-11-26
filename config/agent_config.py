"""
Configuration for agents - centralized location for all hardcoded values.
This allows easy updates without modifying agent code.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional


CONFIG_DIR = Path(__file__).parent
DATA_DIR = CONFIG_DIR.parent / "brand_extraction" / "data"
CONFIG_FILE = CONFIG_DIR / "agent_config.json"


def load_json_file(filepath: Path, default: dict = None) -> dict:
    """Load JSON configuration file with fallback to default."""
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return default or {}


# Load external config if available
_external_config = load_json_file(CONFIG_FILE, {})

# Known brands - can be loaded from file or environment variable
KNOWN_BRANDS = _external_config.get("known_brands") or os.getenv(
    "KNOWN_BRANDS",
    "Samsung,Apple,LG,Sony,Philips,Xiaomi,OnePlus,Google,Nokia,Motorola,Dell,HP,Lenovo,Asus,Acer,Microsoft,Realme,Oppo,Vivo,TCL,BenQ,Epson,TikTok,YouTube,Netflix,Amazon,Meta,Facebook,Twitter,Instagram"
).split(",")

KNOWN_BRANDS = [b.strip() for b in KNOWN_BRANDS if b.strip()]

# Common product categories
COMMON_CATEGORIES = _external_config.get("common_categories") or [
    "electronics", "technology", "smartphones", "televisions", "computers", "laptops",
    "fashion", "clothing", "accessories", "media", "entertainment",
    "home appliances", "audio", "gaming", "software", "streaming",
    "news", "social media", "advertising", "marketing", "e-commerce",
    "automotive", "healthcare", "finance", "education", "travel"
]

# Category-specific keywords for confidence boosting
CATEGORY_KEYWORDS = _external_config.get("category_keywords") or {
    "electronics": ["tv", "television", "phone", "smartphone", "computer", "laptop", "tablet", "device"],
    "media": ["streaming", "youtube", "netflix", "podcast", "video", "content", "channel"],
    "technology": ["ai", "algorithm", "software", "digital", "smart", "app", "platform"],
    "fashion": ["clothing", "apparel", "wear", "style", "fashion", "designer"],
    "home appliances": ["appliance", "refrigerator", "washer", "dryer", "oven", "microwave"]
}

# Campaign keywords for confidence scoring
CAMPAIGN_KEYWORDS = _external_config.get("campaign_keywords") or [
    "campaign", "advertisement", "marketing", "promotion", "launch",
    "brand awareness", "ad campaign", "marketing campaign", "commercial"
]

# Brand tier mappings for reach estimation
BRAND_TIERS = _external_config.get("brand_tiers") or {
    "tier1": ["Samsung", "Apple", "Google", "Microsoft", "Amazon", "Meta"],
    "tier2": ["LG", "Sony", "Xiaomi", "OnePlus", "HP", "Dell"],
    "tier3": ["Philips", "TCL", "Motorola", "Nokia", "Asus", "Acer"]
}

# Confidence thresholds (configurable)
CONFIDENCE_THRESHOLDS = _external_config.get("confidence_thresholds") or {
    "very_high": 0.9,
    "high": 0.75,
    "medium": 0.5,
    "low": 0.3,
    "very_low": 0.1
}

# Confidence weights for different factors
CONFIDENCE_WEIGHTS = _external_config.get("confidence_weights") or {
    "ner_entity_count": 0.1,      # Weight per NER entity found
    "known_brand_match": 0.3,      # Weight for matching known brands
    "keyword_match": 0.2,          # Weight for keyword presence
    "extraction_ratio": 0.15,      # Weight for brands/candidates ratio
    "base_confidence": 0.5,        # Base confidence when entities found
    "no_entity_base": 0.3          # Base confidence when no entities
}


def get_known_brands() -> List[str]:
    """Get list of known brands (allows runtime updates)."""
    return KNOWN_BRANDS.copy()


def get_category_keywords() -> Dict[str, List[str]]:
    """Get category keywords mapping."""
    return CATEGORY_KEYWORDS.copy()


def get_brand_tiers() -> Dict[str, List[str]]:
    """Get brand tier mappings."""
    return BRAND_TIERS.copy()


# TODO: External call opportunity - Load brands from API/database
# Could fetch from:
# - Company database API
# - Brand registry service
# - Knowledge graph (Wikidata, DBpedia)
# Function: load_brands_from_external_source() -> List[str]


# TODO: External call opportunity - Update category taxonomy
# Could fetch from:
# - Product taxonomy API
# - Industry-standard classifications (UNSPSC, eClass)
# Function: load_categories_from_taxonomy_service() -> List[str]


# TODO: External call opportunity - Dynamic keyword extraction
# Could use:
# - NLP keyword extraction service
# - ML-based keyword relevance scoring
# - Historical query analysis
# Function: extract_keywords_with_ml(prompt: str) -> Dict[str, float]

