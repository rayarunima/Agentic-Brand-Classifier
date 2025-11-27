"""
Category extraction agent with improved confidence scoring and reduced hardcoding.
"""
import dspy
from typing import Dict, Optional, List
from collections import Counter
from config import COMMON_CATEGORIES, get_category_keywords, get_brand_to_category
from utils.confidence_scoring import calculate_category_confidence, fuzzy_match_in_list, count_keyword_matches


class CategoryExtractionSignature(dspy.Signature):
    """Extract the product or service category from user text"""
    prompt = dspy.InputField(desc="The text containing product or service info")
    category = dspy.OutputField(desc="A single product category name only (e.g., 'Electronics', 'Audio Equipment'). Do not include placeholders like {category}, explanations, or Python code.")


class CategoryAgent:
    """
    Category extraction agent with advanced confidence scoring.
    
    Uses:
    - DSPy for category extraction
    - Fuzzy matching against known categories
    - Keyword-based confidence boosting
    """
    
    def __init__(self):
        self.predictor = dspy.Predict(CategoryExtractionSignature)
        self.common_categories = COMMON_CATEGORIES
        self.category_keywords = get_category_keywords()
        self.brand_to_category = get_brand_to_category()
    
    def _infer_category_from_brands(self, brands: List[str]) -> Optional[str]:
        """
        Infer category from brand list using brand-to-category mappings.
        
        Args:
            brands: List of brand names (can be list of strings or list of dicts with 'name' key)
        
        Returns:
            Most common category inferred from brands, or None if no mapping found
        """
        if not brands:
            return None
        
        # Extract brand names if brands are in dict format
        brand_names = []
        for brand in brands:
            if isinstance(brand, dict):
                brand_names.append(brand.get("name", ""))
            elif isinstance(brand, str):
                brand_names.append(brand)
        
        # Map brands to categories
        categories = []
        for brand_name in brand_names:
            if brand_name and brand_name in self.brand_to_category:
                categories.append(self.brand_to_category[brand_name])
        
        if not categories:
            return None
        
        # Return the most common category
        category_counts = Counter(categories)
        most_common_category = category_counts.most_common(1)[0][0]
        
        return most_common_category
    
    def extract_category(self, prompt: str, context: Optional[Dict] = None) -> Dict[str, any]:
        """
        Extract category from prompt with sophisticated confidence scoring.
        Uses brand context to infer category when available.
        
        Args:
            prompt: User prompt text
            context: Optional context dict with 'brands' key containing list of brands
        
        Returns: {
            'category': str,
            'confidence': float (0-1),
            'normalized_category': str
        }
        """
        # Step 1: Try to infer category from brand context if available
        brand_inferred_category = None
        brands = None
        if context:
            brands = context.get("brands") or context.get("brand_names")
            if brands:
                brand_inferred_category = self._infer_category_from_brands(brands)
        
        # Step 2: Use DSPy to extract category from prompt
        result = self.predictor(prompt=prompt)
        dspy_category = result.category.strip() if result.category else ""
        
        # Clean DSPy category - remove placeholders and invalid values
        if dspy_category:
            dspy_category = str(dspy_category).strip()
            # Remove common placeholders
            if dspy_category in ["{category}", "category", "None", "null", ""]:
                dspy_category = ""
            # Remove Python structure markers
            dspy_category = dspy_category.strip("'\"[]{}")
            # Skip if it looks like a Python structure
            if any(char in dspy_category for char in ["{", "}", "[", "]", "dict", "list"]):
                dspy_category = ""
        
        # Step 3: Decide on final category (prefer brand-inferred if available and clear)
        # Also handle multiple categories (e.g., "Electronics" + "Media")
        categories_to_combine = []
        
        if brand_inferred_category:
            categories_to_combine.append(brand_inferred_category)
        
        if dspy_category:
            # Check if DSPy category is different from brand-inferred
            if not brand_inferred_category or brand_inferred_category.lower() != dspy_category.lower():
                categories_to_combine.append(dspy_category)
        
        # Combine categories if multiple are found (e.g., "Electronics, Media")
        if len(categories_to_combine) > 1:
            # Remove duplicates (case-insensitive)
            unique_categories = []
            seen = set()
            for cat in categories_to_combine:
                cat_lower = cat.lower()
                if cat_lower not in seen:
                    unique_categories.append(cat)
                    seen.add(cat_lower)
            final_category = ", ".join(unique_categories)
        elif len(categories_to_combine) == 1:
            final_category = categories_to_combine[0]
        else:
            final_category = ""
        
        # Normalize category
        normalized = final_category.lower().strip() if final_category else ""
        
        # Find matched category from known categories (fuzzy matching)
        matched_category = None
        if normalized:
            matched_category, match_score = fuzzy_match_in_list(
                normalized,
                [cat.lower() for cat in self.common_categories],
                threshold=0.6
            )
            # Convert back to original case
            if matched_category:
                # Find original case version
                for cat in self.common_categories:
                    if cat.lower() == matched_category:
                        matched_category = cat
                        break
        
        # If matched_category exists but final_category doesn't match, use matched_category
        if matched_category and final_category.lower() != matched_category.lower():
            final_category = matched_category
        
        # Count keyword matches
        keywords_found = 0
        category_for_keywords = matched_category or final_category
        if category_for_keywords and category_for_keywords.lower() in self.category_keywords:
            relevant_keywords = self.category_keywords[category_for_keywords.lower()]
            keywords_found = count_keyword_matches(prompt.lower(), relevant_keywords)
        
        # Build enhanced context for confidence calculation
        enhanced_context = context.copy() if context else {}
        enhanced_context["brand_context"] = bool(brands)
        enhanced_context["brand_inferred_category"] = brand_inferred_category
        if brand_inferred_category and final_category.lower() == brand_inferred_category.lower():
            enhanced_context["brand_category_agreement"] = True
        
        # Calculate confidence score
        confidence = calculate_category_confidence(
            category=final_category,
            prompt=prompt,
            matched_category=matched_category,
            keywords_found=keywords_found,
            context=enhanced_context
        )
        
        return {
            "category": final_category,
            "confidence": confidence,
            "normalized_category": matched_category or normalized
        }


# TODO: External call opportunity - Category taxonomy validation
# Could use:
# - UNSPSC (United Nations Standard Products and Services Code) API
# - eClass classification system
# - Google Product Taxonomy API
# - Industry-specific taxonomies
# Function: validate_category_with_taxonomy(category: str, industry: str) -> Tuple[bool, str, float]

# TODO: External call opportunity - Context-aware category extraction
# Could use:
# - Knowledge graphs to infer category from brand relationships
# - Historical category assignments for similar prompts
# - ML models trained on product categorization datasets
# Function: extract_category_with_context(prompt: str, brands: List[str], context: Dict) -> Dict

# TODO: External call opportunity - Multi-level category hierarchy
# Could use:
# - Hierarchical category databases
# - E-commerce category trees (Amazon, eBay APIs)
# Function: get_category_hierarchy(category: str) -> Dict[str, List[str]]
