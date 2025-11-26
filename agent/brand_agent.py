"""
Brand extraction agent with improved confidence scoring and reduced hardcoding.
"""
import dspy
from typing import List, Dict, Optional, Set
from brand_extraction.entity_extractor import extract_entities
from config import get_known_brands, get_brand_relationships
from utils.confidence_scoring import calculate_brand_confidence, fuzzy_match_in_list


class BrandValidationSignature(dspy.Signature):
    """Given entity candidates, return only real brands."""
    entities = dspy.InputField(desc="List of entity candidates extracted using NER")
    text = dspy.InputField(desc="Original user query")
    brands = dspy.OutputField(desc="A clean list of brand names present in the text")


class BrandExpansionSignature(dspy.Signature):
    """Given explicit brands and context, identify related brands (competitors, alternatives, parent companies)."""
    explicit_brands = dspy.InputField(desc="Brands explicitly mentioned in the text")
    text = dspy.InputField(desc="Original user query")
    context_keywords = dspy.InputField(desc="Key context words from the text (e.g., video, streaming, social media)")
    related_brands = dspy.OutputField(desc="Related brands that are contextually relevant: competitors, parent companies, or alternatives in the same domain")


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
        self.expander = dspy.Predict(BrandExpansionSignature)
        self.known_brands = get_known_brands()
        self.brand_relationships = get_brand_relationships()
    
    def _calculate_individual_match_score(
        self, 
        brand: str, 
        fuzzy_match_score: float, 
        prompt: str,
        is_ner_entity: bool
    ) -> float:
        """
        Calculate individual match score for a brand (0.0-1.0).
        
        Factors:
        1. Fuzzy match score (0.0-1.0)
        2. Prompt prominence (exact mentions, capitalization)
        3. NER entity boost (if found via NER)
        """
        # Base score from fuzzy matching
        base_score = fuzzy_match_score
        
        # Prompt prominence boost
        prompt_lower = prompt.lower()
        brand_lower = brand.lower()
        
        # Exact match in prompt (case-insensitive)
        if brand_lower in prompt_lower:
            # Check if it appears as a standalone word or with proper capitalization
            if brand in prompt:  # Exact case match
                base_score = max(base_score, 1.0)
            else:
                # Case-insensitive match - boost but not full score
                base_score = max(base_score, 0.95)
        
        # NER entity boost (if found via NER, it's more reliable)
        if is_ner_entity:
            base_score = min(1.0, base_score + 0.05)
        
        # Normalize to [0, 1]
        return round(min(1.0, max(0.0, base_score)), 2)
    
    def _infer_related_brands(
        self, 
        explicit_brands: List[str], 
        prompt: str
    ) -> List[Dict[str, any]]:
        """
        Infer related brands using knowledge graph and contextual inference.
        
        Returns list of related brands with match scores:
        - Parent companies: 0.2-0.3
        - Competitors: 0.5-0.7
        - Category-related: 0.3-0.5
        """
        related_brands = []
        seen_brands = set(b.lower() for b in explicit_brands)
        prompt_lower = prompt.lower()
        
        # Extract context keywords for domain inference
        context_keywords = []
        if any(kw in prompt_lower for kw in ["video", "streaming", "content", "platform"]):
            context_keywords.append("video_platforms")
        if any(kw in prompt_lower for kw in ["social", "media", "network", "sharing"]):
            context_keywords.append("social_media")
        if any(kw in prompt_lower for kw in ["streaming", "watch", "movie", "show"]):
            context_keywords.append("streaming")
        if any(kw in prompt_lower for kw in ["phone", "smartphone", "mobile", "device"]):
            context_keywords.append("smartphones")
        
        # 1. Find parent companies
        parent_companies = self.brand_relationships.get("parent_companies", {})
        for brand in explicit_brands:
            # Check if brand has a parent company
            parent = parent_companies.get(brand)
            if parent and parent.lower() not in seen_brands:
                related_brands.append({
                    "name": parent,
                    "match_score": 0.2,  # Lower score for parent companies
                    "type": "parent"
                })
                seen_brands.add(parent.lower())
        
        # 2. Find competitors/alternatives
        competitors = self.brand_relationships.get("competitors", {})
        for brand in explicit_brands:
            brand_competitors = competitors.get(brand, [])
            for competitor in brand_competitors:
                if competitor.lower() not in seen_brands:
                    # Check if competitor is contextually relevant
                    score = 0.6  # Base competitor score
                    # Boost if competitor appears in context keywords
                    if any(comp.lower() in prompt_lower for comp in [competitor]):
                        score = 0.7
                    related_brands.append({
                        "name": competitor,
                        "match_score": round(score, 2),
                        "type": "competitor"
                    })
                    seen_brands.add(competitor.lower())
        
        # 3. Contextual inference using category_brands
        category_brands = self.brand_relationships.get("category_brands", {})
        for category, brands_in_category in category_brands.items():
            if category in context_keywords:
                for cat_brand in brands_in_category:
                    if cat_brand.lower() not in seen_brands:
                        # Check if brand is mentioned in prompt context
                        score = 0.4  # Base category-related score
                        if cat_brand.lower() in prompt_lower:
                            score = 0.5
                        related_brands.append({
                            "name": cat_brand,
                            "match_score": round(score, 2),
                            "type": "category_related"
                        })
                        seen_brands.add(cat_brand.lower())
        
        # 4. Use DSPy for additional contextual inference
        if explicit_brands:
            try:
                explicit_brands_str = ", ".join(explicit_brands)
                context_keywords_str = ", ".join(context_keywords) if context_keywords else "general"
                result = self.expander(
                    explicit_brands=explicit_brands_str,
                    text=prompt,
                    context_keywords=context_keywords_str
                )
                
                # Parse DSPy output for related brands
                if result.related_brands:
                    dspy_brands_str = result.related_brands
                    if isinstance(dspy_brands_str, str):
                        dspy_brands = [b.strip() for b in dspy_brands_str.split(",") if b.strip()]
                    elif isinstance(dspy_brands_str, list):
                        dspy_brands = [str(b).strip() for b in dspy_brands_str if b]
                    else:
                        dspy_brands = []
                    
                    for dspy_brand in dspy_brands:
                        # Normalize brand name
                        matched_brand, _ = fuzzy_match_in_list(
                            dspy_brand,
                            self.known_brands,
                            threshold=0.6
                        )
                        final_brand = matched_brand if matched_brand else dspy_brand
                        
                        if final_brand.lower() not in seen_brands:
                            # DSPy inferred brands get medium score
                            score = 0.5
                            if final_brand.lower() in prompt_lower:
                                score = 0.6
                            related_brands.append({
                                "name": final_brand,
                                "match_score": round(score, 2),
                                "type": "dspy_inferred"
                            })
                            seen_brands.add(final_brand.lower())
            except Exception:
                # If DSPy expansion fails, continue with knowledge graph results
                pass
        
        return related_brands
    
    def extract_brand(self, prompt: str) -> Dict[str, any]:
        """
        Extract brands from prompt with sophisticated confidence scoring.
        
        Returns: {
            'brands': List[Dict[str, Any]] where each dict has 'name' and 'match_score',
            'confidence': float (0-1) - overall confidence,
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
        ner_entities_set = set(c.lower() for c in candidates)

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
            
            # Calculate individual match scores for each brand
            brands_with_scores = []
            for brand in brands:
                matched_brand, fuzzy_score = fuzzy_match_in_list(
                    brand,
                    self.known_brands,
                    threshold=0.6
                )
                final_brand = matched_brand if matched_brand else brand
                is_ner_entity = False  # No NER entities found
                match_score = self._calculate_individual_match_score(
                    final_brand, 
                    fuzzy_score if matched_brand else 0.5,
                    prompt,
                    is_ner_entity
                )
                brands_with_scores.append({
                    "name": final_brand,
                    "match_score": match_score
                })
            
            # Entity expansion: Infer related brands
            explicit_brand_names = [b["name"] for b in brands_with_scores]
            related_brands = self._infer_related_brands(explicit_brand_names, prompt)
            
            # Add related brands to the list
            for related in related_brands:
                if related["name"].lower() not in [b["name"].lower() for b in brands_with_scores]:
                    brands_with_scores.append(related)
            
            # Sort by match score
            brands_with_scores.sort(key=lambda x: x["match_score"], reverse=True)
            
            # Calculate overall confidence for backward compatibility
            brand_names = [b["name"] for b in brands_with_scores]
            context = {
                "ner_confidences": [b["match_score"] for b in brands_with_scores],
                "signal_sources": ["dspy_only"] if brands else []
            }
            confidence = calculate_brand_confidence(
                brands=brand_names,
                ner_entity_count=0,
                total_candidates=0,
                prompt=prompt,
                context=context
            )
            
            return {
                "brands": brands_with_scores,
                "confidence": confidence,
                "ner_entity_count": 0,
                "matched_entities": len(brands_with_scores)
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
        
        # 3. Normalize and validate brands against known brands, calculate individual scores
        brands_with_scores = []
        ner_confidences = []
        seen = set()
        
        for brand in brands:
            # Try to match against known brands (fuzzy matching)
            matched_brand, fuzzy_score = fuzzy_match_in_list(
                brand,
                self.known_brands,
                threshold=0.6
            )
            
            final_brand = matched_brand if matched_brand else brand
            brand_lower = final_brand.lower()
            
            # Skip duplicates
            if brand_lower in seen:
                continue
            seen.add(brand_lower)
            
            # Check if this brand was found via NER
            is_ner_entity = brand_lower in ner_entities_set or any(
                brand_lower in c.lower() or c.lower() in brand_lower 
                for c in candidates
            )
            
            # Calculate individual match score
            match_score = self._calculate_individual_match_score(
                final_brand,
                fuzzy_score if matched_brand else 0.5,
                prompt,
                is_ner_entity
            )
            
            brands_with_scores.append({
                "name": final_brand,
                "match_score": match_score
            })
            ner_confidences.append(match_score)
        
        # Sort by match score (descending) for better display
        brands_with_scores.sort(key=lambda x: x["match_score"], reverse=True)
        
        # 4. Entity expansion: Infer related brands
        explicit_brand_names = [b["name"] for b in brands_with_scores]
        related_brands = self._infer_related_brands(explicit_brand_names, prompt)
        
        # Add related brands to the list (with lower scores)
        for related in related_brands:
            # Avoid duplicates
            if related["name"].lower() not in [b["name"].lower() for b in brands_with_scores]:
                brands_with_scores.append(related)
        
        # Re-sort by match score after adding related brands
        brands_with_scores.sort(key=lambda x: x["match_score"], reverse=True)
        
        # 5. Calculate overall confidence score for backward compatibility
        brand_names = [b["name"] for b in brands_with_scores]
        context = {
            "ner_confidences": ner_confidences,
            "signal_sources": ["spacy_ner", "dspy_validation"]
        }
        
        confidence = calculate_brand_confidence(
            brands=brand_names,
            ner_entity_count=ner_entity_count,
            total_candidates=total_candidates,
            prompt=prompt,
            context=context
        )
        
        return {
            "brands": brands_with_scores,
            "confidence": confidence,
            "ner_entity_count": ner_entity_count,
            "matched_entities": len(brands_with_scores)
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
