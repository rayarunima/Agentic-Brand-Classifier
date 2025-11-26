"""
Reach estimation agent with improved confidence scoring and reduced hardcoding.
"""
import dspy
import re
from typing import Dict, Optional
from config import get_brand_tiers
from utils.confidence_scoring import calculate_numeric_confidence


class ReachEstimationSignature(dspy.Signature):
    """Estimate campaign reach based on brand, category, and campaign type."""
    brand_name = dspy.InputField(desc="Brand name")
    category = dspy.InputField(desc="Product category")
    campaign_type = dspy.InputField(desc="Campaign type")
    prompt = dspy.InputField(desc="User prompt or campaign context")
    estimated_reach = dspy.OutputField(desc="Estimated reach in number of users (as integer)")
    reach_category = dspy.OutputField(desc="Reach category: local, regional, national, international")
    confidence_level = dspy.OutputField(desc="Confidence level: low, medium, high")


class ReachEstimatorAgent:
    """
    Agent that calculates estimated reach for campaigns with advanced confidence scoring.
    
    Uses:
    - DSPy for reach estimation
    - Brand tier-based fallback calculations
    - Advanced confidence scoring based on input quality
    """
    
    def __init__(self):
        self.reach_estimator = dspy.Predict(ReachEstimationSignature)
        self.brand_tiers = get_brand_tiers()
    
    def calculate_reach(
        self,
        brand_name: str,
        category: str = "",
        campaign_type: str = "",
        prompt: str = ""
    ) -> Dict[str, any]:
        """
        Calculate estimated reach with sophisticated confidence scoring.
        
        Args:
            brand_name: Brand name
            category: Product category
            campaign_type: Type of campaign
            prompt: User prompt text
        
        Returns: {
            'brand_name': str,
            'estimated_reach': int,
            'reach_category': str,
            'confidence': float (0-1),
            'confidence_level': str
        }
        """
        try:
            result = self.reach_estimator(
                brand_name=brand_name,
                category=category or "General",
                campaign_type=campaign_type or "Brand Awareness",
                prompt=prompt or f"Campaign for {brand_name}"
            )
            
            # Parse estimated_reach (handle both string and int)
            reach_str = str(result.estimated_reach).strip()
            numbers = re.findall(r'\d+', reach_str)
            if numbers:
                estimated_reach = int(numbers[0])
            else:
                estimated_reach = self._get_brand_based_reach(brand_name)
            
            reach_category = result.reach_category.strip() if result.reach_category else "regional"
            confidence_level = result.confidence_level.strip().lower() if result.confidence_level else "medium"
            
        except Exception:
            # Fallback calculation
            estimated_reach = self._get_brand_based_reach(brand_name)
            reach_category = "regional"
            confidence_level = "medium"
        
        # Determine expected range based on brand tier and reach category
        expected_range = self._get_expected_range(brand_name, reach_category)
        
        # Calculate sophisticated confidence score
        context = {
            "brand_name": brand_name,
            "category": category,
            "campaign_type": campaign_type,
            "reach_category": reach_category
        }
        
        numeric_confidence = calculate_numeric_confidence(
            value=float(estimated_reach),
            expected_range=expected_range,
            context=context
        )
        
        # Map confidence level to numeric and combine
        confidence_map = {"low": 0.5, "medium": 0.7, "high": 0.9}
        level_confidence = confidence_map.get(confidence_level, 0.7)
        
        # Combine numeric and level confidence (weighted average)
        combined_confidence = (numeric_confidence * 0.6) + (level_confidence * 0.4)
        
        # Boost confidence based on input quality
        if brand_name and category:
            combined_confidence = min(0.95, combined_confidence + 0.1)
        if prompt and any(kw in prompt.lower() for kw in ["reach", "audience", "viewers", "users"]):
            combined_confidence = min(0.95, combined_confidence + 0.1)
        
        return {
            "brand_name": brand_name,
            "estimated_reach": estimated_reach,
            "reach_category": reach_category,
            "confidence": round(combined_confidence, 3),
            "confidence_level": confidence_level
        }
    
    def _get_brand_based_reach(self, brand_name: str) -> int:
        """Get base reach estimate based on brand tier."""
        for tier, brands in self.brand_tiers.items():
            if brand_name in brands:
                if tier == "tier1":
                    return 250000 + (abs(hash(brand_name)) % 750000)  # 250k-1M
                elif tier == "tier2":
                    return 50000 + (abs(hash(brand_name)) % 450000)  # 50k-500k
                else:
                    return 10000 + (abs(hash(brand_name)) % 90000)  # 10k-100k
        # Default for unknown brands
        return 25000 + (abs(hash(brand_name)) % 75000)  # 25k-100k
    
    def _get_expected_range(self, brand_name: str, reach_category: str) -> tuple:
        """Get expected reach range for confidence calculation."""
        # Base ranges by category
        category_ranges = {
            "local": (1000, 50000),
            "regional": (10000, 500000),
            "national": (100000, 5000000),
            "international": (500000, 50000000)
        }
        
        base_range = category_ranges.get(reach_category.lower(), (10000, 500000))
        
        # Adjust based on brand tier
        for tier, brands in self.brand_tiers.items():
            if brand_name in brands:
                if tier == "tier1":
                    # Multiply range by 3-5x for tier1 brands
                    return (base_range[0] * 3, base_range[1] * 5)
                elif tier == "tier2":
                    # Multiply by 1.5-2x for tier2
                    return (int(base_range[0] * 1.5), int(base_range[1] * 2))
        
        return base_range


# TODO: External call opportunity - Real reach data from analytics platforms
# Could use:
# - Google Analytics API for website traffic data
# - Social media APIs (Facebook Insights, Twitter Analytics) for follower counts
# - Ad platform APIs (Google Ads, Facebook Ads) for campaign reach metrics
# - Market research APIs (Nielsen, comScore)
# Function: fetch_real_reach_data(brand_name: str, campaign_id: str) -> Optional[int]

# TODO: External call opportunity - Historical reach benchmarks
# Could use:
# - Industry benchmarks database
# - Similar campaign performance data
# - Market research databases
# Function: get_benchmark_reach(category: str, campaign_type: str, region: str) -> Tuple[int, int]

# TODO: External call opportunity - Demographic-based reach estimation
# Could use:
# - Census data APIs
# - Demographic targeting APIs
# - Market research platforms
# Function: estimate_reach_by_demographics(campaign: Dict, demographics: Dict) -> int
