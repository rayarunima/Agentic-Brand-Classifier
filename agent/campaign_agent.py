"""
Campaign insight agent with improved confidence scoring and reduced hardcoding.
"""
import os
from typing import Dict, List, Optional
from pathlib import Path
import json
import dspy
from config import get_brand_tiers
from utils.confidence_scoring import calculate_campaign_confidence


class CampaignExtractionSignature(dspy.Signature):
    """Extract campaign information from user prompt or brand context."""
    prompt = dspy.InputField(desc="User prompt or brand context")
    brand_name = dspy.InputField(desc="Brand name to analyze")
    campaign_name = dspy.OutputField(desc="Campaign name if mentioned, otherwise generate realistic campaign name")
    campaign_type = dspy.OutputField(desc="Campaign type: product launch, brand awareness, seasonal, promotional, etc.")
    campaign_description = dspy.OutputField(desc="Brief description of the campaign")


class CampaignInsightAgent:
    """
    Campaign insight agent with advanced confidence scoring.
    
    Uses:
    - DSPy for campaign extraction
    - Configurable campaign templates
    - Advanced confidence scoring
    """
    
    def __init__(self, campaign_templates_path: Optional[str] = None):
        """
        Initialize campaign agent.
        
        Args:
            campaign_templates_path: Optional path to JSON file with campaign templates
        """
        self.api_key = os.getenv("SERPAPI_API_KEY")
        self.campaign_extractor = dspy.Predict(CampaignExtractionSignature)
        
        # Load campaign templates from file or use defaults
        self.campaign_templates = self._load_campaign_templates(campaign_templates_path)
    
    def _load_campaign_templates(self, templates_path: Optional[str] = None) -> Dict[str, List[str]]:
        """Load campaign templates from file or return defaults."""
        if templates_path and Path(templates_path).exists():
            try:
                with open(templates_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Default templates - could be moved to config file
        return {
            "Samsung": ["Galaxy Unpacked", "Sustainable Future", "Innovation for All"],
            "Apple": ["Shot on iPhone", "Think Different", "New Product Launch"],
            "LG": ["Life's Good", "OLED Experience", "Smart Home"],
            "Sony": ["Be Moved", "The Best of Entertainment", "Creator's Choice"],
            "Google": ["Made by Google", "Pixel Experience", "Android"],
            "Microsoft": ["Surface", "Microsoft 365", "Windows"]
        }
    
    def get_campaign_insights(self, brand_name: str, prompt: str = "") -> Dict[str, any]:
        """
        Get campaign insights for a brand with sophisticated confidence scoring.
        
        Args:
            brand_name: Brand name to analyze
            prompt: User prompt text
        
        Returns: {
            'brand': str,
            'campaigns': List[Dict],
            'confidence': float (0-1),
            'method_used': str
        }
        """
        insights = []
        method_used = "DSPy-based extraction"
        
        # Try to extract campaign info using DSPy
        try:
            context = prompt if prompt else f"Marketing campaign for {brand_name}"
            result = self.campaign_extractor(
                prompt=context,
                brand_name=brand_name
            )
            
            campaign_name = result.campaign_name.strip() if result.campaign_name else None
            if not campaign_name or len(campaign_name) < 3:
                # Generate from templates if available
                campaigns = self.campaign_templates.get(brand_name, [f"{brand_name} Campaign"])
                campaign_name = campaigns[0] if campaigns else f"{brand_name} Marketing Campaign"
            
            insights.append({
                "name": campaign_name,
                "type": result.campaign_type.strip() if result.campaign_type else "Brand Awareness",
                "description": result.campaign_description.strip() if result.campaign_description else f"Marketing campaign for {brand_name}",
                "status": "Active",
                "estimated_launch": "Recent"
            })
        except Exception as e:
            # Fallback to template-based approach
            campaigns = self.campaign_templates.get(brand_name, [])
            if campaigns:
                insights.append({
                    "name": campaigns[0],
                    "type": "Brand Awareness",
                    "description": f"Marketing campaign promoting {brand_name} products and brand values",
                    "status": "Active",
                    "estimated_launch": "Recent"
                })
            else:
                insights.append({
                    "name": f"{brand_name} Marketing Campaign",
                    "type": "General Marketing",
                    "description": f"Ongoing marketing activities for {brand_name}",
                    "status": "Estimated",
                    "estimated_launch": "Ongoing"
                })
        
        # Calculate sophisticated confidence score
        confidence = calculate_campaign_confidence(
            campaigns=insights,
            prompt=prompt,
            brand_name=brand_name,
            context={}
        )
        
        return {
            "brand": brand_name,
            "campaigns": insights,
            "confidence": confidence,
            "method_used": method_used
        }


# TODO: External call opportunity - Real campaign data from APIs
# Could use:
# - SerpAPI Google News search (already configured but not used)
# - BrandWatch API for campaign mentions
# - Social media APIs (Twitter, Facebook) for campaign hashtags
# - Ad libraries (Facebook Ad Library, Google Ads Transparency)
# Function: fetch_real_campaign_data(brand_name: str, date_range: Tuple) -> List[Dict]

# TODO: External call opportunity - Campaign database lookup
# Could use:
# - Industry campaign databases
# - Marketing intelligence platforms (SEMrush, Ahrefs)
# - PR wire services (PR Newswire, Business Wire)
# Function: lookup_campaign_in_database(brand_name: str, campaign_name: str) -> Optional[Dict]

# TODO: External call opportunity - Campaign performance metrics
# Could use:
# - Campaign analytics APIs
# - Ad performance data
# - Social media engagement metrics
# Function: get_campaign_performance_metrics(campaign_id: str) -> Dict
