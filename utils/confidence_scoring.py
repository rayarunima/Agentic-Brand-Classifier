"""
Advanced confidence scoring utilities for agent predictions.
Provides sophisticated confidence calculation without external API calls.
"""
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher
import re
from config.agent_config import (
    get_known_brands, get_category_keywords, CONFIDENCE_WEIGHTS,
    CONFIDENCE_THRESHOLDS, CAMPAIGN_KEYWORDS
)


def similarity_score(str1: str, str2: str) -> float:
    """Calculate similarity between two strings (0-1)."""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def fuzzy_match_in_list(item: str, reference_list: List[str], threshold: float = 0.7) -> Tuple[Optional[str], float]:
    """
    Find fuzzy match in reference list.
    Returns: (matched_item, similarity_score) or (None, 0.0)
    """
    item_lower = item.lower().strip()
    best_match = None
    best_score = 0.0
    
    for ref in reference_list:
        ref_lower = ref.lower().strip()
        # Exact match
        if item_lower == ref_lower:
            return (ref, 1.0)
        
        # Substring match
        if item_lower in ref_lower or ref_lower in item_lower:
            score = min(len(item_lower), len(ref_lower)) / max(len(item_lower), len(ref_lower))
            if score > best_score:
                best_score = score
                best_match = ref
        
        # Similarity match
        sim = similarity_score(item_lower, ref_lower)
        if sim > best_score:
            best_score = sim
            best_match = ref
    
    if best_score >= threshold:
        return (best_match, best_score)
    return (None, best_score)


def count_keyword_matches(text: str, keywords: List[str], case_sensitive: bool = False) -> int:
    """Count how many keywords appear in text."""
    text_normalized = text if case_sensitive else text.lower()
    keywords_normalized = keywords if case_sensitive else [k.lower() for k in keywords]
    
    matches = 0
    for keyword in keywords_normalized:
        if keyword in text_normalized:
            matches += 1
    return matches


def calculate_brand_confidence(
    brands: List[str],
    ner_entity_count: int,
    total_candidates: int,
    prompt: str = "",
    context: Optional[Dict] = None
) -> float:
    """
    Calculate confidence score for brand extraction.
    
    Factors considered:
    1. NER entity count (more entities = higher confidence)
    2. Known brand matching (fuzzy matching)
    3. Extraction ratio (brands/candidates)
    4. Keyword presence in prompt
    5. Contextual signals
    
    Args:
        brands: List of extracted brands
        ner_entity_count: Number of NER entities found
        total_candidates: Total number of candidate entities
        prompt: Original prompt text
        context: Optional context dict with additional signals
    
    Returns:
        Confidence score (0.0 - 1.0)
    """
    if not brands:
        return 0.0
    
    weights = CONFIDENCE_WEIGHTS
    known_brands = get_known_brands()
    
    # Factor 1: Base confidence
    base_conf = weights["base_confidence"] if ner_entity_count > 0 else weights["no_entity_base"]
    
    # Factor 2: Known brand matching
    matched_known = 0
    match_scores = []
    for brand in brands:
        matched, score = fuzzy_match_in_list(brand, known_brands, threshold=0.6)
        if matched:
            matched_known += 1
            match_scores.append(score)
    
    known_brand_score = (matched_known / len(brands)) * weights["known_brand_match"]
    avg_match_score = sum(match_scores) / len(match_scores) if match_scores else 0.0
    known_brand_score *= (0.7 + 0.3 * avg_match_score)  # Boost if high similarity
    
    # Factor 3: NER entity count boost
    entity_boost = min(ner_entity_count * weights["ner_entity_count"], 0.2)
    
    # Factor 4: Extraction ratio (more selective = higher confidence)
    if total_candidates > 0:
        extraction_ratio = len(brands) / total_candidates
        # Ideal ratio is around 0.3-0.7 (not too many, not too few)
        if 0.3 <= extraction_ratio <= 0.7:
            ratio_score = weights["extraction_ratio"]
        elif extraction_ratio < 0.3:
            ratio_score = weights["extraction_ratio"] * 0.7  # Might be too selective
        else:
            ratio_score = weights["extraction_ratio"] * 0.5  # Too many extracted
    else:
        ratio_score = weights["extraction_ratio"] * 0.5
    
    # Factor 5: Keyword presence
    keyword_score = 0.0
    if prompt:
        # Check if brand names appear in prompt (exact or partial)
        prompt_lower = prompt.lower()
        brand_mentions = sum(1 for brand in brands if brand.lower() in prompt_lower)
        if brand_mentions > 0:
            keyword_score = (brand_mentions / len(brands)) * weights["keyword_match"]
    
    # Factor 6: Contextual signals (if provided)
    context_boost = 0.0
    if context:
        # Boost if entities have high NER confidence
        if "ner_confidences" in context:
            avg_ner_conf = sum(context["ner_confidences"]) / len(context["ner_confidences"])
            context_boost = avg_ner_conf * 0.1
        
        # Boost if multiple signal sources agree
        if "signal_sources" in context and len(context["signal_sources"]) > 1:
            context_boost += 0.05
    
    # Combine all factors
    confidence = (
        base_conf +
        known_brand_score +
        entity_boost +
        ratio_score +
        keyword_score +
        context_boost
    )
    
    # Penalty for low quality matches
    if matched_known == 0 and ner_entity_count > 0:
        confidence *= 0.7  # Penalty if no known brands matched but entities found
    
    # Penalty for too many brands (likely noise)
    if len(brands) > 5:
        confidence *= 0.9
    
    # Normalize to [0, 1]
    confidence = min(1.0, max(0.0, confidence))
    
    return round(confidence, 3)


def calculate_category_confidence(
    category: str,
    prompt: str,
    matched_category: Optional[str] = None,
    keywords_found: int = 0,
    context: Optional[Dict] = None
) -> float:
    """
    Calculate confidence score for category extraction.
    
    Factors considered:
    1. Category normalization (matches known categories)
    2. Keyword presence
    3. Category length and quality
    4. Contextual signals
    
    Args:
        category: Extracted category string
        prompt: Original prompt text
        matched_category: Matched category from known list
        keywords_found: Number of relevant keywords found
        context: Optional context dict
    
    Returns:
        Confidence score (0.0 - 1.0)
    """
    weights = CONFIDENCE_WEIGHTS
    category_keywords = get_category_keywords()
    
    # Base confidence
    if not category or len(category) < 3:
        return 0.2
    
    base_conf = 0.5
    
    # Factor 1: Known category match
    category_match_score = 0.0
    if matched_category:
        # Exact match
        if category.lower() == matched_category.lower():
            category_match_score = 0.4
        else:
            # Fuzzy match
            sim = similarity_score(category, matched_category)
            category_match_score = 0.3 * sim
    
    # Factor 2: Keyword presence
    keyword_score = 0.0
    prompt_lower = prompt.lower()
    
    # Check category-specific keywords
    if matched_category and matched_category.lower() in category_keywords:
        relevant_keywords = category_keywords[matched_category.lower()]
        keyword_matches = count_keyword_matches(prompt_lower, relevant_keywords)
        if keyword_matches > 0:
            keyword_score = min(keyword_matches * 0.1, 0.3)
    
    # General category-related keywords
    general_keywords = ["category", "type", "kind", "product", "class", "group"]
    if any(kw in prompt_lower for kw in general_keywords):
        keyword_score += 0.1
    
    # Factor 3: Category quality (length, capitalization)
    quality_score = 0.0
    if len(category) >= 5:  # Longer categories are usually more specific
        quality_score = 0.1
    if category[0].isupper():  # Proper capitalization
        quality_score += 0.05
    
    # Factor 4: Contextual signals
    context_boost = 0.0
    if context:
        if "brand_context" in context and context["brand_context"]:
            context_boost += 0.05  # Brand context helps category identification
        if "previous_category" in context:
            # Consistency with previous predictions
            if similarity_score(category, context["previous_category"]) > 0.8:
                context_boost += 0.05
    
    # Combine factors
    confidence = (
        base_conf +
        category_match_score +
        keyword_score +
        quality_score +
        context_boost
    )
    
    # Penalties
    if not matched_category and len(category) < 5:
        confidence *= 0.8  # Short, unknown category
    
    # Normalize
    confidence = min(1.0, max(0.0, confidence))
    
    return round(confidence, 3)


def calculate_campaign_confidence(
    campaigns: List[Dict],
    prompt: str,
    brand_name: str = "",
    context: Optional[Dict] = None
) -> float:
    """
    Calculate confidence score for campaign extraction.
    
    Args:
        campaigns: List of campaign dicts
        prompt: Original prompt text
        brand_name: Brand name if available
        context: Optional context
    
    Returns:
        Confidence score (0.0 - 1.0)
    """
    if not campaigns:
        return 0.3  # Low confidence if no campaigns found
    
    base_conf = 0.5
    
    # Factor 1: Campaign count (more = higher confidence)
    count_score = min(len(campaigns) * 0.1, 0.3)
    
    # Factor 2: Keyword presence
    keyword_matches = count_keyword_matches(prompt.lower(), CAMPAIGN_KEYWORDS)
    keyword_score = min(keyword_matches * 0.15, 0.3)
    
    # Factor 3: Campaign quality (has name, description, etc.)
    quality_score = 0.0
    for campaign in campaigns:
        if campaign.get("name") and len(campaign.get("name", "")) > 3:
            quality_score += 0.1
        if campaign.get("description") and len(campaign.get("description", "")) > 10:
            quality_score += 0.1
    quality_score = min(quality_score / len(campaigns), 0.2)
    
    # Factor 4: Brand context
    brand_score = 0.0
    if brand_name and brand_name.lower() in prompt.lower():
        brand_score = 0.1
    
    confidence = base_conf + count_score + keyword_score + quality_score + brand_score
    
    return min(1.0, max(0.0, round(confidence, 3)))


def calculate_numeric_confidence(
    value: float,
    expected_range: Tuple[float, float],
    context: Optional[Dict] = None
) -> float:
    """
    Calculate confidence for numeric predictions (reach, brand lift, etc.).
    
    Args:
        value: Predicted numeric value
        expected_range: (min, max) expected range
        context: Optional context
    
    Returns:
        Confidence score
    """
    min_val, max_val = expected_range
    
    # Check if value is in reasonable range
    if min_val <= value <= max_val:
        range_score = 0.7
    elif value < min_val * 0.5 or value > max_val * 1.5:
        range_score = 0.3  # Out of range
    else:
        range_score = 0.5  # Borderline
    
    # Context boost
    context_boost = 0.0
    if context:
        if "similar_values" in context:
            # Check consistency with similar predictions
            avg_similar = sum(context["similar_values"]) / len(context["similar_values"])
            if abs(value - avg_similar) / avg_similar < 0.2:  # Within 20%
                context_boost = 0.2
    
    confidence = range_score + context_boost
    return min(1.0, max(0.0, round(confidence, 3)))


# TODO: External call opportunity - Use ML model for confidence scoring
# Could use:
# - Trained confidence prediction model (XGBoost, Neural Network)
# - Ensemble of multiple confidence estimators
# - Historical accuracy data for similar predictions
# Function: predict_confidence_with_ml(features: Dict) -> float


# TODO: External call opportunity - Cross-validation with external knowledge base
# Could use:
# - Wikidata/DBpedia to verify brand existence
# - Product database to validate categories
# - Industry databases for reach/brand lift benchmarks
# Function: validate_with_knowledge_base(entity: str, entity_type: str) -> Tuple[bool, float]


# TODO: External call opportunity - Historical performance tracking
# Could track:
# - Confidence calibration over time
# - Per-agent accuracy metrics
# - Confidence distribution analysis
# Function: get_calibrated_confidence(raw_confidence: float, agent_name: str) -> float

