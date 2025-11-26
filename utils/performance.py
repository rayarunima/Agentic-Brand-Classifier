"""
Performance tracking utilities for agent execution metrics.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict
import psutil
import os


@dataclass
class AgentMetrics:
    """Metrics for a single agent execution."""
    agent_name: str
    start_time: float
    end_time: Optional[float] = None
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    cache_hit: bool = False
    confidence_score: Optional[float] = None
    error: Optional[str] = None
    
    def finish(self, tokens_used: Optional[int] = None, cache_hit: bool = False):
        """Mark the execution as finished and calculate latency."""
        self.end_time = time.time()
        self.latency_ms = (self.end_time - self.start_time) * 1000
        if tokens_used is not None:
            self.tokens_used = tokens_used
        self.cache_hit = cache_hit


@dataclass
class PerformanceTracker:
    """Tracks performance metrics across multiple agent executions."""
    metrics_history: List[AgentMetrics] = field(default_factory=list)
    cache_stats: Dict[str, int] = field(default_factory=lambda: {"hits": 0, "misses": 0})
    total_requests: int = 0
    total_tokens: int = 0
    
    def start_agent(self, agent_name: str) -> AgentMetrics:
        """Start tracking an agent execution."""
        metric = AgentMetrics(agent_name=agent_name, start_time=time.time())
        self.metrics_history.append(metric)
        return metric
    
    def record_cache_hit(self, agent_name: str):
        """Record a cache hit."""
        self.cache_stats["hits"] += 1
    
    def record_cache_miss(self, agent_name: str):
        """Record a cache miss."""
        self.cache_stats["misses"] += 1
    
    def get_average_latency(self, agent_name: Optional[str] = None) -> float:
        """Get average latency in milliseconds."""
        metrics = [m for m in self.metrics_history if m.latency_ms is not None]
        if agent_name:
            metrics = [m for m in metrics if m.agent_name == agent_name]
        
        if not metrics:
            return 0.0
        return sum(m.latency_ms for m in metrics) / len(metrics)
    
    def get_throughput(self) -> float:
        """Get requests per second."""
        if not self.metrics_history:
            return 0.0
        
        total_time = sum(m.latency_ms for m in self.metrics_history if m.latency_ms) / 1000
        if total_time == 0:
            return 0.0
        return len(self.metrics_history) / total_time
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate as a percentage."""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total == 0:
            return 0.0
        return (self.cache_stats["hits"] / total) * 100
    
    def get_summary(self) -> Dict:
        """Get a summary of all metrics."""
        return {
            "total_requests": len(self.metrics_history),
            "average_latency_ms": self.get_average_latency(),
            "throughput_rps": self.get_throughput(),
            "cache_hit_rate": self.get_cache_hit_rate(),
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "total_tokens": self.total_tokens,
            "agents_summary": self._get_agents_summary()
        }
    
    def _get_agents_summary(self) -> Dict[str, Dict]:
        """Get summary per agent."""
        agent_stats = defaultdict(lambda: {"count": 0, "total_latency": 0, "errors": 0})
        
        for metric in self.metrics_history:
            agent_stats[metric.agent_name]["count"] += 1
            if metric.latency_ms:
                agent_stats[metric.agent_name]["total_latency"] += metric.latency_ms
            if metric.error:
                agent_stats[metric.agent_name]["errors"] += 1
        
        summary = {}
        for agent_name, stats in agent_stats.items():
            summary[agent_name] = {
                "execution_count": stats["count"],
                "average_latency_ms": stats["total_latency"] / stats["count"] if stats["count"] > 0 else 0,
                "error_count": stats["errors"]
            }
        
        return summary


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "rss_mb": mem_info.rss / 1024 / 1024,  # Resident Set Size
        "vms_mb": mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        "percent": process.memory_percent()
    }


def get_cpu_usage() -> float:
    """Get current CPU usage percentage."""
    return psutil.cpu_percent(interval=0.1)


def calculate_cost_savings(requests: int, tokens_per_request: int = 1000) -> Dict[str, float]:
    """
    Calculate cost savings of Ollama (local) vs cloud APIs.
    Returns cost breakdown in USD.
    """
    # Ollama costs (approximate electricity/hardware amortization)
    # Assuming $0.01 per 1000 requests (very low overhead)
    ollama_cost = (requests / 1000) * 0.01
    
    # Cloud API costs (approximate, as of 2024)
    # OpenAI GPT-4: $0.03 per 1K input tokens, $0.06 per 1K output tokens
    # Anthropic Claude: $0.008 per 1K input tokens, $0.024 per 1K output tokens
    # Using average of $0.02 per 1K tokens (input + output)
    
    total_tokens = requests * tokens_per_request
    cloud_cost_openai = (total_tokens / 1000) * 0.03
    cloud_cost_anthropic = (total_tokens / 1000) * 0.016  # average of input/output
    cloud_cost_avg = (cloud_cost_openai + cloud_cost_anthropic) / 2
    
    return {
        "ollama_cost_usd": round(ollama_cost, 2),
        "cloud_cost_openai_usd": round(cloud_cost_openai, 2),
        "cloud_cost_anthropic_usd": round(cloud_cost_anthropic, 2),
        "cloud_cost_avg_usd": round(cloud_cost_avg, 2),
        "savings_vs_openai": round(cloud_cost_openai - ollama_cost, 2),
        "savings_vs_anthropic": round(cloud_cost_anthropic - ollama_cost, 2),
        "savings_avg": round(cloud_cost_avg - ollama_cost, 2),
        "savings_percentage": round(((cloud_cost_avg - ollama_cost) / cloud_cost_avg * 100) if cloud_cost_avg > 0 else 0, 1)
    }

