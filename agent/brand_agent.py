"""
Brand extraction agent with improved confidence scoring and reduced hardcoding.
"""
import dspy
from typing import List, Dict, Optional
from brand_extraction.entity_extractor import extract_entities
from config import get_known_brands
from utils.confidence_scoring import calculate_brand_confidence, fuzzy_match_in_list


class BrandValidationSignature(dspy.Signature):
    """Given entity candidates, return only real brands."""
    entities = dspy.InputField(desc="List of entity candidates extracted using NER")
    text = dspy.InputField(desc="Original user query")
    brands = dspy.OutputField(desc="A clean list of brand names present in the text")


class BrandAgent:
    """
    Brand extraction agent with advanced confidence scoring.
    
    Uses:
    - SpaCy NER for entity extraction
    - DSPy for brand validation
    - Advanced confidence scoring based on multiple factors
    """
    
    def __init__(self):
        self.validator = dspy.Predict(BrandValidationSignature)
        self.known_brands = get_known_brands()
    
    def extract_brand(self, prompt: str) -> Dict[str, any]:
        """
        Extract brands from prompt with sophisticated confidence scoring.
        
        Returns: {
            'brands': List[str],
            'confidence': float (0-1),
            'ner_entity_count': int,
            'matched_entities': int
        }
        """
        # 1. Extract raw entities (SpaCy)
        ner_result = extract_entities(prompt)
        candidates = [
            ent["text"] for ent in ner_result.get("entities", [])
            if ent.get("label") in ("ORG", "PRODUCT")
        ]

        # Remove duplicates
        candidates = list(set(candidates))
        ner_entity_count = len(candidates)
        total_candidates = ner_entity_count

        # If no NER entities found, fallback to DSPy directly
        if not candidates:
            result = self.validator(entities=[], text=prompt)
            brands_str = result.brands if isinstance(result.brands, str) else (
                ", ".join(result.brands) if isinstance(result.brands, list) else ""
            )
            
            if isinstance(brands_str, str) and brands_str:
                brands = [b.strip() for b in brands_str.split(",") if b.strip()]
            else:
                brands = []
            
            # Calculate confidence for no-entity case
            context = {
                "ner_confidences": [],
                "signal_sources": ["dspy_only"] if brands else []
            }
            confidence = calculate_brand_confidence(
                brands=brands,
                ner_entity_count=0,
                total_candidates=0,
                prompt=prompt,
                context=context
            )
            
            return {
                "brands": brands,
                "confidence": confidence,
                "ner_entity_count": 0,
                "matched_entities": len(brands)
            }

        # 2. Ask DSPy to filter which of these are actual brands
        result = self.validator(
            entities=candidates,
            text=prompt
        )

        # Parse DSPy output
        if isinstance(result.brands, str):
            brands = [b.strip() for b in result.brands.split(",") if b.strip()]
        elif isinstance(result.brands, list):
            brands = [str(b).strip() for b in result.brands if b]
        else:
            brands = []
        
        # 3. Normalize and validate brands against known brands
        normalized_brands = []
        ner_confidences = []
        
        for brand in brands:
            # Try to match against known brands (fuzzy matching)
            matched_brand, match_score = fuzzy_match_in_list(
                brand,
                self.known_brands,
                threshold=0.6
            )
            
            if matched_brand:
                normalized_brands.append(matched_brand)
                ner_confidences.append(match_score)
            else:
                # Keep original if no match (might be a new/unknown brand)
                normalized_brands.append(brand)
                ner_confidences.append(0.5)  # Medium confidence for unknown
        
        # Remove duplicates while preserving order
        seen = set()
        unique_brands = []
        for brand in normalized_brands:
            brand_lower = brand.lower()
            if brand_lower not in seen:
                seen.add(brand_lower)
                unique_brands.append(brand)
        
        brands = unique_brands
        
        # 4. Calculate sophisticated confidence score
        context = {
            "ner_confidences": ner_confidences,
            "signal_sources": ["spacy_ner", "dspy_validation"]
        }
        
        confidence = calculate_brand_confidence(
            brands=brands,
            ner_entity_count=ner_entity_count,
            total_candidates=total_candidates,
            prompt=prompt,
            context=context
        )
        
        return {
            "brands": brands,
            "confidence": confidence,
            "ner_entity_count": ner_entity_count,
            "matched_entities": len(brands)
        }


# TODO: External call opportunity - Brand validation via API
# Could use:
# - Brand registry API (Trademark databases, BrandWatch, SimilarWeb)
# - Wikidata/DBpedia to verify brand existence and get metadata
# - Social media APIs to check brand mentions/verification
# Function: validate_brand_with_external_api(brand_name: str) -> Tuple[bool, float, Dict]

# TODO: External call opportunity - Product-to-brand mapping service
# Could use:
# - Product databases (GS1, Amazon Product API)
# - E-commerce APIs (to map product names to brands)
# Function: get_brand_from_product(product_name: str) -> Optional[str]

# TODO: External call opportunity - Historical brand extraction accuracy
# Could track:
# - Per-brand accuracy rates
# - Confidence calibration data
# - Common false positives/negatives
# Function: get_calibrated_confidence(raw_confidence: float, brand: str) -> float
