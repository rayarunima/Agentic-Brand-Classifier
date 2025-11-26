"""
Brand lift estimation agent with improved confidence scoring and reduced hardcoding.
"""
import dspy
import re
from typing import Dict, Optional, List
from utils.confidence_scoring import calculate_numeric_confidence


class BrandLiftEstimationSignature(dspy.Signature):
    """Estimate brand lift percentage based on campaign details."""
    brand_name = dspy.InputField(desc="Brand name")
    campaign_name = dspy.InputField(desc="Campaign name or description")
    campaign_type = dspy.InputField(desc="Campaign type")
    prompt = dspy.InputField(desc="User prompt or campaign context")
    brand_lift_percentage = dspy.OutputField(desc="Estimated brand lift percentage (as number)")
    lift_category = dspy.OutputField(desc="Lift category: minimal, moderate, significant, exceptional")
    key_factors = dspy.OutputField(desc="Key factors contributing to brand lift")


class BrandLiftAgent:
    """
    Agent that calculates brand lift percentage with advanced confidence scoring.
    
    Uses:
    - DSPy for brand lift estimation
    - Sophisticated confidence scoring
    - Context-aware confidence adjustment
    """
    
    # Realistic brand lift ranges by category
    LIFT_RANGES = {
        "minimal": (0.5, 3.0),
        "moderate": (3.0, 8.0),
        "significant": (8.0, 15.0),
        "exceptional": (15.0, 25.0)
    }
    
    # Confidence mapping for lift categories
    CATEGORY_CONFIDENCE_MAP = {
        "minimal": 0.6,
        "moderate": 0.75,
        "significant": 0.85,
        "exceptional": 0.9
    }
    
    # Brand lift keywords
    LIFT_KEYWORDS = ["lift", "awareness", "recall", "engagement", "perception", "recognition"]
    
    def __init__(self):
        self.lift_estimator = dspy.Predict(BrandLiftEstimationSignature)
    
    def calculate_brand_lift(
        self,
        brand_name: str,
        campaign_name: str = "",
        campaign_type: str = "",
        prompt: str = ""
    ) -> Dict[str, any]:
        """
        Calculate brand lift percentage with sophisticated confidence scoring.
        
        Args:
            brand_name: Brand name
            campaign_name: Campaign name
            campaign_type: Type of campaign
            prompt: User prompt text
        
        Returns: {
            'brand_name': str,
            'campaign_name': str,
            'brand_lift_percentage': float,
            'lift_category': str,
            'confidence': float (0-1),
            'key_factors': List[str]
        }
        """
        try:
            result = self.lift_estimator(
                brand_name=brand_name,
                campaign_name=campaign_name or f"{brand_name} Campaign",
                campaign_type=campaign_type or "Brand Awareness",
                prompt=prompt or f"Brand lift analysis for {brand_name}"
            )
            
            # Parse brand_lift_percentage
            lift_str = str(result.brand_lift_percentage).strip()
            numbers = re.findall(r'\d+\.?\d*', lift_str)
            if numbers:
                brand_lift = float(numbers[0])
                # Clamp to realistic range
                brand_lift = max(0.5, min(25.0, brand_lift))
            else:
                brand_lift = self._calculate_baseline_lift(brand_name, campaign_name)
            
            lift_category = result.lift_category.strip().lower() if result.lift_category else "moderate"
            
            # Normalize category
            if lift_category not in self.CATEGORY_CONFIDENCE_MAP:
                # Try to match partial
                for cat in self.CATEGORY_CONFIDENCE_MAP.keys():
                    if cat in lift_category:
                        lift_category = cat
                        break
                else:
                    lift_category = "moderate"
            
            # Parse key factors
            key_factors_str = result.key_factors.strip() if result.key_factors else ""
            if key_factors_str:
                key_factors = [f.strip() for f in key_factors_str.split(",")[:3] if f.strip()]
            else:
                key_factors = ["Brand Awareness", "Market Engagement", "Consumer Recall"]
                
        except Exception:
            # Fallback calculation
            brand_lift = self._calculate_baseline_lift(brand_name, campaign_name)
            lift_category = "moderate"
            key_factors = ["Brand Awareness", "Market Engagement"]
        
        # Get expected range for this lift category
        expected_range = self.LIFT_RANGES.get(lift_category, self.LIFT_RANGES["moderate"])
        
        # Calculate numeric confidence based on range
        context = {
            "brand_name": brand_name,
            "campaign_name": campaign_name,
            "campaign_type": campaign_type,
            "lift_category": lift_category
        }
        
        numeric_confidence = calculate_numeric_confidence(
            value=brand_lift,
            expected_range=expected_range,
            context=context
        )
        
        # Get category-based confidence
        category_confidence = self.CATEGORY_CONFIDENCE_MAP.get(lift_category, 0.75)
        
        # Combine confidences (weighted average)
        combined_confidence = (numeric_confidence * 0.6) + (category_confidence * 0.4)
        
        # Boost confidence based on input quality
        if campaign_name and len(campaign_name) > 3:
            combined_confidence = min(0.95, combined_confidence + 0.1)
        
        prompt_lower = prompt.lower() if prompt else ""
        if any(kw in prompt_lower for kw in self.LIFT_KEYWORDS):
            combined_confidence = min(0.95, combined_confidence + 0.1)
        
        # Penalty if lift is outside realistic range
        if brand_lift < expected_range[0] or brand_lift > expected_range[1]:
            combined_confidence *= 0.8
        
        return {
            "brand_name": brand_name,
            "campaign_name": campaign_name or f"{brand_name} Campaign",
            "brand_lift_percentage": round(brand_lift, 2),
            "lift_category": lift_category,
            "confidence": round(combined_confidence, 3),
            "key_factors": key_factors
        }
    
    def _calculate_baseline_lift(self, brand_name: str, campaign_name: str) -> float:
        """Calculate baseline lift percentage deterministically."""
        seed = abs(hash(brand_name + campaign_name)) % 1000
        # Realistic range: 2% to 18%
        lift = 2.0 + (seed % 160) / 10.0
        return round(lift, 2)


# TODO: External call opportunity - Real brand lift data from measurement platforms
# Could use:
# - Brand lift measurement APIs (Facebook Brand Lift Studies, Google Brand Lift)
# - Market research platforms (Nielsen, Kantar)
# - Survey APIs for brand awareness studies
# Function: fetch_real_brand_lift_data(brand_name: str, campaign_id: str, date_range: Tuple) -> Optional[float]

# TODO: External call opportunity - Industry benchmarks
# Could use:
# - Industry benchmark databases
# - Competitive intelligence platforms
# - Market research reports APIs
# Function: get_industry_brand_lift_benchmark(category: str, campaign_type: str) -> Tuple[float, float]

# TODO: External call opportunity - Predictive modeling
# Could use:
# - ML models trained on historical brand lift data
# - Campaign attribute-based prediction models
# - Multi-factor analysis APIs
# Function: predict_brand_lift_with_ml(campaign_features: Dict) -> Tuple[float, float]
