"""
Category extraction agent with improved confidence scoring and reduced hardcoding.
"""
import dspy
from typing import Dict, Optional
from config import COMMON_CATEGORIES, get_category_keywords
from utils.confidence_scoring import calculate_category_confidence, fuzzy_match_in_list, count_keyword_matches


class CategoryExtractionSignature(dspy.Signature):
    """Extract the product or service category from user text"""
    prompt = dspy.InputField(desc="The text containing product or service info")
    category = dspy.OutputField(desc="Product category like electronics, fashion, etc.")


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
    
    def extract_category(self, prompt: str, context: Optional[Dict] = None) -> Dict[str, any]:
        """
        Extract category from prompt with sophisticated confidence scoring.
        
        Args:
            prompt: User prompt text
            context: Optional context dict (e.g., extracted brands for context)
        
        Returns: {
            'category': str,
            'confidence': float (0-1),
            'normalized_category': str
        }
        """
        result = self.predictor(prompt=prompt)
        category = result.category.strip() if result.category else ""
        
        # Normalize category
        normalized = category.lower().strip()
        
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
        
        # Count keyword matches
        keywords_found = 0
        if matched_category and matched_category.lower() in self.category_keywords:
            relevant_keywords = self.category_keywords[matched_category.lower()]
            keywords_found = count_keyword_matches(prompt.lower(), relevant_keywords)
        
        # Calculate confidence score
        confidence = calculate_category_confidence(
            category=category,
            prompt=prompt,
            matched_category=matched_category,
            keywords_found=keywords_found,
            context=context or {}
        )
        
        return {
            "category": category,
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
