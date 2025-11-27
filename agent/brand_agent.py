"""
Brand extraction agent with improved confidence scoring and reduced hardcoding.
"""
import re
import dspy
from typing import List, Dict, Optional, Set
from brand_extraction.entity_extractor import extract_entities
from config import get_known_brands, get_brand_relationships
from utils.confidence_scoring import calculate_brand_confidence, fuzzy_match_in_list


class BrandValidationSignature(dspy.Signature):
    """Given entity candidates, return only real brands."""
    entities = dspy.InputField(desc="List of entity candidates extracted using NER")
    text = dspy.InputField(desc="Original user query")
    brands = dspy.OutputField(desc="A comma-separated list of brand names only (e.g., 'Samsung, Apple, Sony'). Do not include explanations, Python code, or structures.")


class BrandExpansionSignature(dspy.Signature):
    """Given explicit brands and context, identify related brands (competitors, alternatives, parent companies)."""
    explicit_brands = dspy.InputField(desc="Brands explicitly mentioned in the text")
    text = dspy.InputField(desc="Original user query")
    context_keywords = dspy.InputField(desc="Key context words from the text (e.g., video, streaming, social media)")
    related_brands = dspy.OutputField(desc="A comma-separated list of related brand names only (e.g., 'YouTube, ByteDance'). Do not include explanations, Python code, dictionaries, or nested structures.")


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
        is_ner_entity: bool,
        is_inferred: bool = False
    ) -> float:
        """
        Calculate individual match score for a brand (0.0-1.0).
        
        Factors:
        1. Fuzzy match score (0.0-1.0)
        2. Prompt prominence (exact mentions, capitalization)
        3. NER entity boost (if found via NER)
        4. Inferred brands are capped at 0.7 (never 100%)
        
        Only brands explicitly mentioned in the prompt can get 100% confidence.
        """
        # Inferred brands (from relationships) are capped at 0.7 maximum
        if is_inferred:
            max_score = 0.7
        else:
            max_score = 1.0
        
        # Base score from fuzzy matching (capped by inferred status)
        base_score = min(fuzzy_match_score, max_score)
        
        # Prompt prominence check - only for explicitly mentioned brands
        prompt_lower = prompt.lower()
        brand_lower = brand.lower()
        
        # Check for exact word match (word boundaries to avoid partial matches)
        # Pattern matches brand as a whole word (case-insensitive)
        word_boundary_pattern = r'\b' + re.escape(brand_lower) + r'\b'
        exact_match_found = bool(re.search(word_boundary_pattern, prompt_lower))
        
        # Check for exact case-sensitive match
        exact_case_match = brand in prompt
        
        # PRIORITY: If brand is explicitly mentioned in prompt, it should get 1.0
        if exact_match_found:
            # Brand is explicitly mentioned in prompt - give it 1.0
            # Only exception: if it's a parent/competitor relationship AND not the primary brand
            # But if it's mentioned, it should still get high score
            if is_inferred:
                # Even if inferred, if mentioned in prompt, give it high score (but not perfect)
                # This handles cases where a brand is both mentioned and inferred as a relationship
                base_score = 0.95
            else:
                # Explicitly mentioned brand (not inferred) gets perfect score
                base_score = 1.0
        else:
            # Brand not explicitly mentioned - score based on fuzzy match, but cap if inferred
            if is_inferred:
                base_score = min(0.7, fuzzy_match_score)
            else:
                # Not inferred, but also not in prompt - reduce score
                base_score = min(0.8, fuzzy_match_score * 0.8)
        
        # NER entity boost (if found via NER, it's more reliable)
        if is_ner_entity and not is_inferred:
            base_score = min(max_score, base_score + 0.05)
        
        # Final cap: Only truly inferred brands (not mentioned) should be capped at 0.7
        # If brand is mentioned in prompt, it should get at least 0.95
        if is_inferred and not exact_match_found:
            # Truly inferred brand (not mentioned) - cap at 0.7
            base_score = min(0.7, base_score)
        
        # Normalize to [0, max_score]
        return round(min(max_score, max(0.0, base_score)), 2)
    
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
                # Check if parent company is also mentioned in prompt
                parent_lower = parent.lower()
                # Check for exact word match of parent in prompt
                word_pattern = r'\b' + re.escape(parent_lower) + r'\b'
                parent_mentioned = bool(re.search(word_pattern, prompt_lower))
                
                if parent_mentioned:
                    # Parent is explicitly mentioned - give it 100% match
                    score = 1.0
                    is_inferred = False  # Not inferred if mentioned
                else:
                    # Parent not mentioned - but if the product (brand) is a well-known product
                    # of the parent (like iPad -> Apple), the parent should get 100% match
                    # because mentioning the product implies the brand
                    major_products = ["ipad", "iphone", "galaxy", "pixel", "surface", "airpods", "macbook"]
                    if brand.lower() in major_products:
                        # Major product mentioned (like iPad Pro), parent (Apple) gets 100% match
                        # This is because mentioning "iPad Pro" clearly refers to Apple
                        score = 1.0
                        is_inferred = False  # Not really inferred, it's the main brand
                    else:
                        # Regular parent relationship
                        score = 0.2
                        is_inferred = True
                
                related_brands.append({
                    "name": parent,
                    "match_score": round(score, 2),
                    "type": "parent",
                    "is_inferred": is_inferred
                })
                seen_brands.add(parent.lower())
        
        # 2. Find competitors/alternatives
        competitors = self.brand_relationships.get("competitors", {})
        for brand in explicit_brands:
            brand_competitors = competitors.get(brand, [])
            for competitor in brand_competitors:
                if competitor.lower() not in seen_brands:
                    # Check if competitor appears in prompt (but still inferred)
                    if competitor.lower() in prompt_lower:
                        score = 0.7  # Maximum for inferred brands
                    else:
                        score = 0.6  # Base competitor score
                    # Cap at 0.7 for all inferred brands
                    score = min(score, 0.7)
                    related_brands.append({
                        "name": competitor,
                        "match_score": round(score, 2),
                        "type": "competitor",
                        "is_inferred": True
                    })
                    seen_brands.add(competitor.lower())
        
        # 3. Contextual inference using category_brands
        category_brands = self.brand_relationships.get("category_brands", {})
        for category, brands_in_category in category_brands.items():
            if category in context_keywords:
                for cat_brand in brands_in_category:
                    if cat_brand.lower() not in seen_brands:
                        # Check if brand is mentioned in prompt context (but still inferred)
                        if cat_brand.lower() in prompt_lower:
                            score = 0.7  # Maximum for inferred brands
                        else:
                            score = 0.4  # Base category-related score
                        # Cap at 0.7 for all inferred brands
                        score = min(score, 0.7)
                        related_brands.append({
                            "name": cat_brand,
                            "match_score": round(score, 2),
                            "type": "category_related",
                            "is_inferred": True
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
                
                # Parse DSPy output for related brands - clean up Python structures
                if result.related_brands:
                    dspy_brands = []
                    dspy_brands_str = result.related_brands
                    
                    if isinstance(dspy_brands_str, str):
                        # Split and clean
                        raw = [b.strip() for b in dspy_brands_str.split(",") if b.strip()]
                        for b in raw:
                            cleaned = b.strip("'\"[]{}")
                            # Skip Python structures and invalid patterns
                            invalid_patterns = ["{", "}", "[", "]", "dict", "list", "competitors", "alternatives", 
                                               "examples", "related_brands", "related", "brands"]
                            if not any(pattern in cleaned.lower() for pattern in invalid_patterns):
                                if cleaned and len(cleaned) > 1:
                                    dspy_brands.append(cleaned)
                    elif isinstance(dspy_brands_str, list):
                        for b in dspy_brands_str:
                            b_str = str(b).strip()
                            cleaned = b_str.strip("'\"[]{}")
                            # Skip Python structures and invalid patterns
                            invalid_patterns = ["{", "}", "[", "]", "dict", "list", "competitors", "alternatives", 
                                               "examples", "related_brands", "related", "brands"]
                            if not any(pattern in cleaned.lower() for pattern in invalid_patterns):
                                if cleaned and len(cleaned) > 1:
                                    dspy_brands.append(cleaned)
                    
                    for dspy_brand in dspy_brands:
                        # Skip if it's a description or explanation text
                        if len(dspy_brand) > 50 or "these are" in dspy_brand.lower() or "examples" in dspy_brand.lower():
                            continue
                            
                        # Normalize brand name
                        matched_brand, _ = fuzzy_match_in_list(
                            dspy_brand,
                            self.known_brands,
                            threshold=0.6
                        )
                        final_brand = matched_brand if matched_brand else dspy_brand
                        
                        # Final validation - must be a reasonable brand name
                        final_brand_lower = final_brand.lower() if final_brand else ""
                        invalid_names = ["related_brands", "related", "brands", "competitors", "alternatives"]
                        
                        if (final_brand and len(final_brand) <= 30 and 
                            final_brand_lower not in seen_brands and
                            final_brand_lower not in invalid_names and
                            not any(inv in final_brand_lower for inv in invalid_names)):
                            # DSPy inferred brands get medium score, capped at 0.7
                            if final_brand_lower in prompt_lower:
                                score = 0.7  # Maximum for inferred brands
                            else:
                                score = 0.5
                            # Cap at 0.7 for all inferred brands
                            score = min(score, 0.7)
                            related_brands.append({
                                "name": final_brand,
                                "match_score": round(score, 2),
                                "type": "dspy_inferred",
                                "is_inferred": True
                            })
                            seen_brands.add(final_brand_lower)
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
            
            # Parse and clean DSPy output
            brands = []
            if isinstance(result.brands, str):
                raw_brands = [b.strip() for b in result.brands.split(",") if b.strip()]
                for b in raw_brands:
                    cleaned = b.strip("'\"[]{}")
                    # Skip Python structures
                    if not any(char in cleaned for char in ["{", "}", "[", "]", "dict", "list", "competitors", "alternatives"]):
                        if cleaned and len(cleaned) > 1:
                            brands.append(cleaned)
            elif isinstance(result.brands, list):
                for b in result.brands:
                    b_str = str(b).strip()
                    cleaned = b_str.strip("'\"[]{}")
                    if not any(char in cleaned for char in ["{", "}", "[", "]", "dict", "list", "competitors", "alternatives"]):
                        if cleaned and len(cleaned) > 1:
                            brands.append(cleaned)
            
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
                    is_ner_entity,
                    is_inferred=False  # Explicit brands from DSPy
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

        # Parse DSPy output - handle various formats and clean up
        brands = []
        if isinstance(result.brands, str):
            # Split by comma and clean
            raw_brands = [b.strip() for b in result.brands.split(",") if b.strip()]
            for b in raw_brands:
                # Remove quotes, brackets, and other Python structure markers
                cleaned = b.strip("'\"[]{}")
                # Skip if it looks like a Python structure
                if not any(char in cleaned for char in ["{", "}", "[", "]", "dict", "list", "competitors", "alternatives"]):
                    if cleaned and len(cleaned) > 1:
                        brands.append(cleaned)
        elif isinstance(result.brands, list):
            for b in result.brands:
                b_str = str(b).strip()
                # Clean up the string
                cleaned = b_str.strip("'\"[]{}")
                # Skip Python structures
                if not any(char in cleaned for char in ["{", "}", "[", "]", "dict", "list", "competitors", "alternatives"]):
                    if cleaned and len(cleaned) > 1:
                        brands.append(cleaned)
        else:
            brands = []
        
        # Additional cleanup: remove any remaining Python structure artifacts
        cleaned_brands = []
        for brand in brands:
            # Remove any text that looks like Python code
            if "(" in brand and ")" in brand:
                # Might be a function call, extract just the brand name part
                parts = brand.split("(")[0].strip()
                if parts:
                    brand = parts
            # Remove common Python structure keywords
            if brand.lower() not in ["dict", "list", "competitors", "alternatives", "examples", "these are"]:
                cleaned_brands.append(brand)
        brands = cleaned_brands
        
        # 3. FIRST: Extract full product names with versions (iPad Pro, Galaxy Tab) and parent brands
        prompt_lower = prompt.lower()
        parent_companies_map = self.brand_relationships.get("parent_companies", {})
        product_to_parent_map = {}  # Track which products were found
        full_product_names = []  # Track full product names with versions (iPad Pro, Galaxy Tab)
        
        # Detect OR questions early
        is_or_question = " or " in prompt_lower or " vs " in prompt_lower or " versus " in prompt_lower
        
        # Extract full product names with versions from prompt
        # Look for patterns like "iPad Pro", "Galaxy Tab"
        # For "Samsung Galaxy Tab", extract "Galaxy Tab" (not "Samsung Galaxy Tab")
        for product, parent_brand in parent_companies_map.items():
            product_lower = product.lower()
            # Pattern: product name + optional version word (Pro, Tab, Air, etc.)
            # Match case-insensitive but preserve original case from prompt
            product_pattern = r'\b' + re.escape(product) + r'(\s+[A-Z][a-z]+)?\b'
            matches = re.finditer(product_pattern, prompt, re.IGNORECASE)
            
            for match in matches:
                full_match = match.group(0)  # Full match (e.g., "iPad Pro", "Galaxy Tab")
                # The regex with word boundaries already extracts just the product+version
                # "Samsung Galaxy Tab" -> regex matches "Galaxy Tab" (word boundary after Samsung)
                
                # Store the product name (iPad Pro, Galaxy Tab)
                if full_match and full_match.lower() not in [f.lower() for f in full_product_names]:
                    full_product_names.append(full_match)
                    product_to_parent_map[product_lower] = parent_brand
        
        # 4. Build brands list: products with versions + parent brands
        brands_with_scores = []
        ner_confidences = []
        seen = set()
        
        # Add full product names with versions (iPad Pro, Galaxy Tab) with 0.9 score
        for full_product in full_product_names:
            product_lower = full_product.lower()
            if product_lower not in seen:
                brands_with_scores.append({
                    "name": full_product,
                    "match_score": 0.9,  # Product with version gets 0.9
                    "type": "product_with_version"
                })
                seen.add(product_lower)
                ner_confidences.append(0.9)
        
        # Add parent brands - always 1.0 (they're the main brands)
        for product_lower, parent_brand in product_to_parent_map.items():
            parent_lower = parent_brand.lower()
            if parent_lower not in seen:
                # Parent brands always get 1.0 (they're the actual brands)
                brands_with_scores.append({
                    "name": parent_brand,
                    "match_score": 1.0,  # Parent brands always 100%
                    "type": "parent_from_product"
                })
                seen.add(parent_lower)
                ner_confidences.append(1.0)
        
        # Now process extracted brands from NER/DSPy
        for brand in brands:
            brand_lower = brand.lower()
            brand_original = brand
            
            # Check if this is a base product name (iPad, Galaxy) - skip these, we have full versions
            is_base_product = brand in parent_companies_map
            if is_base_product:
                continue  # Skip base product names - we have full versions (iPad Pro, Galaxy Tab)
            
            # Check if brand contains a product name - might be a full product name
            # If it's already in our full_product_names, skip it
            if any(brand_lower == fp.lower() for fp in full_product_names):
                continue
            
            # Try to match against known brands (fuzzy matching)
            matched_brand, fuzzy_score = fuzzy_match_in_list(
                brand,
                self.known_brands,
                threshold=0.6
            )
            
            final_brand = matched_brand if matched_brand else brand
            brand_lower_final = final_brand.lower()
            
            # Skip if it's a product name after matching
            if final_brand in parent_companies_map:
                continue
            
            # Check if brand already exists (from product mapping)
            existing_brand = next(
                (b for b in brands_with_scores if b["name"].lower() == brand_lower_final),
                None
            )
            
            if existing_brand:
                # Brand already exists - check if this is an explicit mention
                # If explicitly mentioned, upgrade score to 100%
                word_pattern = r'\b' + re.escape(brand_lower_final) + r'\b'
                if re.search(word_pattern, prompt_lower):
                    # Explicitly mentioned - upgrade to 100%
                    existing_brand["match_score"] = 1.0
                    existing_brand["type"] = "explicit"
                # Otherwise keep existing score (95% from product mapping)
                continue
            
            seen.add(brand_lower_final)
            
            # Check if this brand was found via NER
            is_ner_entity = brand_lower_final in ner_entities_set or any(
                brand_lower_final in c.lower() or c.lower() in brand_lower_final 
                for c in candidates
            )
            
            # Calculate individual match score
            match_score = self._calculate_individual_match_score(
                final_brand,
                fuzzy_score if matched_brand else 0.5,
                prompt,
                is_ner_entity,
                is_inferred=False
            )
            
            brands_with_scores.append({
                "name": final_brand,
                "match_score": match_score
            })
            ner_confidences.append(match_score)
        
        # Sort by match score (descending)
        brands_with_scores.sort(key=lambda x: x["match_score"], reverse=True)
        
        # 5. For OR questions: Keep products and their parent brands, filter out irrelevant ones
        # But be LESS strict - allow products and their parents, just filter out completely unrelated brands
        if is_or_question:
            # Get parent brands of mentioned products
            mentioned_parent_brands = set(product_to_parent_map.values())
            mentioned_parent_brands_lower = {p.lower() for p in mentioned_parent_brands}
            
            # Get full product names (for filtering)
            full_product_names_lower = {fp.lower() for fp in full_product_names}
            
            # Keep: full product names, parent brands, and explicitly mentioned brands
            # Filter out: unrelated brands like Google, etc.
            filtered_brands = []
            for brand_data in brands_with_scores:
                brand_name = brand_data.get("name", "")
                brand_lower = brand_name.lower()
                
                # Keep if it's a product with version (iPad Pro, Galaxy Tab)
                if brand_data.get("type") == "product_with_version":
                    filtered_brands.append(brand_data)
                # Keep if it's a parent brand of a mentioned product
                elif brand_lower in mentioned_parent_brands_lower:
                    filtered_brands.append(brand_data)
                # Keep if it's explicitly mentioned in prompt
                else:
                    word_pattern = r'\b' + re.escape(brand_lower) + r'\b'
                    if re.search(word_pattern, prompt_lower):
                        filtered_brands.append(brand_data)
            
            brands_with_scores = filtered_brands
            brands_with_scores.sort(key=lambda x: x["match_score"], reverse=True)
        
        # 6. Entity expansion: Infer related brands (SKIP for OR questions)
        explicit_brand_names = [b["name"] for b in brands_with_scores]
        # For OR questions, don't infer any related brands - only show what's directly relevant
        related_brands = self._infer_related_brands(explicit_brand_names, prompt) if not is_or_question else []
        
        # Add related brands to the list (with lower scores)
        # IMPORTANT: Don't override explicit brands with inferred ones
        for related in related_brands:
            related_name_lower = related["name"].lower()
            # Check if this brand already exists
            existing_brand = next(
                (b for b in brands_with_scores if b["name"].lower() == related_name_lower),
                None
            )
            
            if existing_brand:
                # Brand already exists - only update if the new score is higher
                # But never override an explicit brand (score >= 0.95) with an inferred one
                if existing_brand["match_score"] < 0.95 and related["match_score"] > existing_brand["match_score"]:
                    existing_brand["match_score"] = related["match_score"]
            else:
                # New brand - add it
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
