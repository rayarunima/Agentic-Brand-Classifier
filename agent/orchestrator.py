"""
Agent Orchestrator - Coordinates multiple agents intelligently.
"""
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from agent.brand_agent import BrandAgent
from agent.category_agent import CategoryAgent
from agent.campaign_agent import CampaignInsightAgent
from agent.reach_agent import ReachEstimatorAgent
from agent.brand_lift_agent import BrandLiftAgent
from utils.performance import AgentMetrics


@dataclass
class AgentResult:
    """Result from a single agent execution."""
    agent_name: str
    result: Any
    confidence: float
    latency_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class OrchestrationContext:
    """Context shared between agents during orchestration."""
    prompt: str
    brands: Optional[List[str]] = None
    category: Optional[str] = None
    campaign_info: Optional[Dict] = None
    execution_order: List[str] = None
    metrics: Dict[str, AgentMetrics] = None
    
    def __post_init__(self):
        if self.execution_order is None:
            self.execution_order = []
        if self.metrics is None:
            self.metrics = {}


class AgentOrchestrator:
    """
    Orchestrates multiple agents intelligently, passing context between them.
    """
    
    def __init__(self):
        self.brand_agent = BrandAgent()
        self.category_agent = CategoryAgent()
        self.campaign_agent = CampaignInsightAgent()
        self.reach_agent = ReachEstimatorAgent()
        self.brand_lift_agent = BrandLiftAgent()
    
    def execute_pipeline(
        self,
        prompt: str,
        enable_brand: bool = True,
        enable_category: bool = True,
        enable_campaign: bool = False,
        enable_reach: bool = False,
        enable_brand_lift: bool = False
    ) -> Dict[str, Any]:
        """
        Execute agent pipeline with intelligent orchestration.
        
        Execution order:
        1. Brand & Category agents run in parallel (independent)
        2. Campaign agent uses brand/category context
        3. Reach agent uses campaign context
        4. Brand Lift agent uses campaign context
        """
        context = OrchestrationContext(prompt=prompt)
        results = {}
        total_start = time.time()
        
        # Phase 1: Independent agents (can run in parallel)
        if enable_brand:
            start = time.time()
            try:
                brand_result = self.brand_agent.extract_brand(prompt)
                if isinstance(brand_result, dict):
                    brands_data = brand_result.get("brands", [])
                    # Extract brand names for context (used by other agents)
                    # Handle both new format (list of dicts) and old format (list of strings)
                    if brands_data and isinstance(brands_data[0], dict):
                        context.brands = [b["name"] for b in brands_data]
                    else:
                        context.brands = brands_data if isinstance(brands_data, list) else []
                    results["brand"] = AgentResult(
                        agent_name="Brand",
                        result=brand_result,
                        confidence=brand_result.get("confidence", 0.5),
                        latency_ms=(time.time() - start) * 1000,
                        success=True
                    )
                else:
                    # Backward compatibility
                    context.brands = brand_result if isinstance(brand_result, list) else []
                    results["brand"] = AgentResult(
                        agent_name="Brand",
                        result={"brands": context.brands},
                        confidence=0.7,
                        latency_ms=(time.time() - start) * 1000,
                        success=True
                    )
                context.execution_order.append("Brand")
            except Exception as e:
                results["brand"] = AgentResult(
                    agent_name="Brand",
                    result=None,
                    confidence=0.0,
                    latency_ms=(time.time() - start) * 1000,
                    success=False,
                    error=str(e)
                )
        
        if enable_category:
            start = time.time()
            try:
                category_result = self.category_agent.extract_category(prompt)
                if isinstance(category_result, dict):
                    context.category = category_result.get("category", "")
                    results["category"] = AgentResult(
                        agent_name="Category",
                        result=category_result,
                        confidence=category_result.get("confidence", 0.5),
                        latency_ms=(time.time() - start) * 1000,
                        success=True
                    )
                else:
                    # Backward compatibility
                    context.category = category_result if isinstance(category_result, str) else ""
                    results["category"] = AgentResult(
                        agent_name="Category",
                        result={"category": context.category},
                        confidence=0.7,
                        latency_ms=(time.time() - start) * 1000,
                        success=True
                    )
                context.execution_order.append("Category")
            except Exception as e:
                results["category"] = AgentResult(
                    agent_name="Category",
                    result=None,
                    confidence=0.0,
                    latency_ms=(time.time() - start) * 1000,
                    success=False,
                    error=str(e)
                )
        
        # Phase 2: Context-dependent agents
        if enable_campaign and context.brands:
            start = time.time()
            try:
                # Use first brand for campaign insights
                primary_brand = context.brands[0] if context.brands else ""
                campaign_result = self.campaign_agent.get_campaign_insights(
                    brand_name=primary_brand,
                    prompt=prompt
                )
                context.campaign_info = campaign_result
                results["campaign"] = AgentResult(
                    agent_name="Campaign",
                    result=campaign_result,
                    confidence=campaign_result.get("confidence", 0.5),
                    latency_ms=(time.time() - start) * 1000,
                    success=True
                )
                context.execution_order.append("Campaign")
            except Exception as e:
                results["campaign"] = AgentResult(
                    agent_name="Campaign",
                    result=None,
                    confidence=0.0,
                    latency_ms=(time.time() - start) * 1000,
                    success=False,
                    error=str(e)
                )
        
        if enable_reach and context.brands:
            start = time.time()
            try:
                primary_brand = context.brands[0] if context.brands else ""
                category = context.category or ""
                campaign_type = ""
                if context.campaign_info and context.campaign_info.get("campaigns"):
                    campaign_type = context.campaign_info["campaigns"][0].get("type", "")
                
                reach_result = self.reach_agent.calculate_reach(
                    brand_name=primary_brand,
                    category=category,
                    campaign_type=campaign_type,
                    prompt=prompt
                )
                results["reach"] = AgentResult(
                    agent_name="Reach",
                    result=reach_result,
                    confidence=reach_result.get("confidence", 0.5),
                    latency_ms=(time.time() - start) * 1000,
                    success=True
                )
                context.execution_order.append("Reach")
            except Exception as e:
                results["reach"] = AgentResult(
                    agent_name="Reach",
                    result=None,
                    confidence=0.0,
                    latency_ms=(time.time() - start) * 1000,
                    success=False,
                    error=str(e)
                )
        
        if enable_brand_lift and context.brands:
            start = time.time()
            try:
                primary_brand = context.brands[0] if context.brands else ""
                campaign_name = ""
                campaign_type = ""
                if context.campaign_info and context.campaign_info.get("campaigns"):
                    campaign_name = context.campaign_info["campaigns"][0].get("name", "")
                    campaign_type = context.campaign_info["campaigns"][0].get("type", "")
                
                lift_result = self.brand_lift_agent.calculate_brand_lift(
                    brand_name=primary_brand,
                    campaign_name=campaign_name,
                    campaign_type=campaign_type,
                    prompt=prompt
                )
                results["brand_lift"] = AgentResult(
                    agent_name="BrandLift",
                    result=lift_result,
                    confidence=lift_result.get("confidence", 0.5),
                    latency_ms=(time.time() - start) * 1000,
                    success=True
                )
                context.execution_order.append("BrandLift")
            except Exception as e:
                results["brand_lift"] = AgentResult(
                    agent_name="BrandLift",
                    result=None,
                    confidence=0.0,
                    latency_ms=(time.time() - start) * 1000,
                    success=False,
                    error=str(e)
                )
        
        total_latency_ms = (time.time() - total_start) * 1000
        
        # Build comprehensive result
        return {
            "results": results,
            "context": context,
            "total_latency_ms": round(total_latency_ms, 2),
            "execution_order": context.execution_order,
            "success_count": sum(1 for r in results.values() if r.success),
            "total_agents": len(results)
        }
    
    def get_execution_graph(self, execution_order: List[str]) -> Dict[str, Any]:
        """Generate execution graph visualization data."""
        graph = {
            "nodes": [],
            "edges": []
        }
        
        # Define dependencies
        dependencies = {
            "Campaign": ["Brand"],
            "Reach": ["Brand", "Category", "Campaign"],
            "BrandLift": ["Brand", "Campaign"]
        }
        
        # Create nodes
        for i, agent in enumerate(execution_order):
            graph["nodes"].append({
                "id": agent,
                "label": agent,
                "position": i,
                "level": self._get_agent_level(agent, execution_order)
            })
            
            # Create edges for dependencies
            if agent in dependencies:
                for dep in dependencies[agent]:
                    if dep in execution_order and execution_order.index(dep) < i:
                        graph["edges"].append({
                            "from": dep,
                            "to": agent,
                            "type": "dependency"
                        })
        
        return graph
    
    def _get_agent_level(self, agent: str, execution_order: List[str]) -> int:
        """Determine the execution level (depth) of an agent."""
        level_map = {
            "Brand": 0,
            "Category": 0,
            "Campaign": 1,
            "Reach": 2,
            "BrandLift": 2
        }
        return level_map.get(agent, 0)

