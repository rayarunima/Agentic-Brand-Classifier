"""
Enhanced Streamlit UI with performance metrics, visualizations, and agent orchestration.
"""
import json
import os
import re
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

import dspy
import streamlit as st
from dotenv import load_dotenv

from agent.brand_agent import BrandAgent
from agent.category_agent import CategoryAgent
from agent.orchestrator import AgentOrchestrator
from brand_extraction.entity_extractor import extract_entities
from utils.performance import PerformanceTracker, get_memory_usage, get_cpu_usage, calculate_cost_savings


STREAMLIT_TITLE = "Agentic Brand Classifier - Enterprise Demo Purposes"
SAMPLES_PATH = Path("prompts") / "sample_prompts.json"
MORE_SAMPLES_PATH = Path("prompts") / "more_prompts.json"


# Initialize session state
if "performance_tracker" not in st.session_state:
    st.session_state.performance_tracker = PerformanceTracker()
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None


# Global lock for DSPy configuration (thread safety)
import threading
_dspy_lock = threading.Lock()
_dspy_configured = False

def configure_lm() -> None:
    """Configure DSPy once per process with thread safety."""
    global _dspy_configured
    
    # Fast path: already configured
    if _dspy_configured:
        return
    
    # Thread-safe configuration
    with _dspy_lock:
        # Double-check after acquiring lock
        if _dspy_configured:
            return
        
        try:
            # Check if DSPy is already configured
            if hasattr(dspy.settings, 'lm') and dspy.settings.lm is not None:
                _dspy_configured = True
                return
        except:
            pass
        
        model_name = os.getenv("DSPY_MODEL_NAME", "ollama/phi3")
        max_tokens = int(os.getenv("DSPY_MAX_TOKENS", "4096"))
        temperature = float(os.getenv("DSPY_TEMPERATURE", "0.2"))

        try:
            dspy.configure(
                lm=dspy.LM(
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            _dspy_configured = True
        except RuntimeError:
            # Already configured in another thread, mark as configured
            _dspy_configured = True
            pass


@st.cache_resource(show_spinner=False)
def load_orchestrator():
    """Load the agent orchestrator."""
    # Configure DSPy before creating orchestrator
    configure_lm()
    return AgentOrchestrator()


@st.cache_data(show_spinner=False)
def load_samples() -> List[str]:
    """Load sample prompts from JSON files."""
    samples = []
    if SAMPLES_PATH.exists():
        with open(SAMPLES_PATH, "r") as handle:
            samples.extend(json.load(handle))
    if MORE_SAMPLES_PATH.exists():
        with open(MORE_SAMPLES_PATH, "r") as handle:
            samples.extend(json.load(handle))
    return samples


def render_sidebar(samples: List[str]) -> Dict[str, bool]:
    """Render sidebar with controls and return agent configuration."""
    st.sidebar.header("⚙️ Configuration")
    
    # Prompt presets
    st.sidebar.subheader("📝 Prompt Presets")
    choice = st.sidebar.selectbox(
        "Pick a sample",
        options=[""] + samples[:20],  # Limit to first 20 for performance
        index=0,
        key="sample_prompt"
    )
    
    st.sidebar.divider()
    st.sidebar.subheader("🤖 Agents")
    
    agent_config = {
        "brand": st.sidebar.toggle(
            "Brand Agent",
            key="use_brand_agent",
            value=st.session_state.get("use_brand_agent", True),
            help="Extract brand names from prompts"
        ),
        "category": st.sidebar.toggle(
            "Category Agent",
            key="use_category_agent",
            value=st.session_state.get("use_category_agent", True),
            help="Extract product/service categories"
        ),
        "campaign": st.sidebar.toggle(
            "Campaign Agent",
            key="use_campaign_agent",
            value=st.session_state.get("use_campaign_agent", False),
            help="Extract campaign insights"
        ),
        "reach": st.sidebar.toggle(
            "Reach Agent",
            key="use_reach_agent",
            value=st.session_state.get("use_reach_agent", False),
            help="Estimate campaign reach"
        ),
        "brand_lift": st.sidebar.toggle(
            "Brand Lift Agent",
            key="use_brand_lift_agent",
            value=st.session_state.get("use_brand_lift_agent", False),
            help="Calculate brand lift percentage"
        )
    }
    
    st.sidebar.divider()
    st.sidebar.subheader("📊 View Options")
    show_metrics = st.sidebar.toggle("Show Performance Metrics", value=True)
    show_history = st.sidebar.toggle("Show Query History", value=False)
    
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Processing Mode")
    processing_mode = st.sidebar.radio(
        "Select mode",
        ["Single Query", "Batch Processing"],
        index=0
    )
    
    st.sidebar.divider()
    st.sidebar.subheader("🗑️ Cache Management")
    st.sidebar.caption("Clear cache to force fresh processing (useful for testing)")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Clear All Cache", use_container_width=True, help="Clear all cached data and resources"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            # Re-initialize essential session state
            st.session_state["performance_tracker"] = PerformanceTracker()
            st.session_state["query_history"] = []
            st.session_state["orchestrator"] = None
            st.success("✅ Cache cleared! Refresh the page or run a new query.")
            st.rerun()
    
    with col2:
        if st.button("Clear Query History", use_container_width=True, help="Clear only query history"):
            st.session_state["query_history"] = []
            st.success("✅ Query history cleared!")
    
    # Show cache status
    cache_info = st.sidebar.expander("ℹ️ Cache Info", expanded=False)
    with cache_info:
        st.caption("**Cached Functions:**")
        st.caption("• Orchestrator (resource)")
        st.caption("• Sample prompts (data)")
        st.caption("\n**Note:** Results are not cached. Each query runs fresh.")
        st.caption("Fast responses on repeated queries may be due to:")
        st.caption("• Model warm-up")
        st.caption("• System optimization")
    
    return choice, agent_config, show_metrics, show_history, processing_mode


def extract_brand_names(brands_data: Any) -> List[str]:
    """
    Extract brand names from brand data structure.
    Handles both new format (list of dicts with 'name' and 'match_score') 
    and old format (list of strings) for backward compatibility.
    
    Args:
        brands_data: Can be list of dicts, list of strings, or None
        
    Returns:
        List of brand name strings
    """
    if not brands_data:
        return []
    
    if isinstance(brands_data, list):
        if len(brands_data) > 0 and isinstance(brands_data[0], dict):
            # New format: list of dicts
            return [b.get("name", "") for b in brands_data if b.get("name")]
        else:
            # Old format: list of strings
            return [str(b) for b in brands_data if b]
    
    return []


# Invalid brand names to filter out
INVALID_BRAND_PATTERNS = [
    "related_brands", "competitors", "alternatives", "examples",
    "these are", "dict", "list", "brands", "category", "type",
    "parent", "competitor", "category_related", "dspy_inferred",
    "no brand", "no brand names", "no brand name", "n/a", "none",
    "not mentioned", "not found", "no brands", "brand names were not",
    "were mentioned", "in the input", "in the given", "context"
]

def is_valid_brand_name(name: str) -> bool:
    """Check if a brand name is valid (not a placeholder or invalid pattern)."""
    if not name or len(name) < 2:
        return False
    
    name_lower = name.lower().strip()
    
    # Skip if it's a sentence or explanation (contains multiple words that look like a sentence)
    words = name_lower.split()
    if len(words) > 5:  # Likely a sentence, not a brand name
        return False
    
    # Skip invalid patterns
    for pattern in INVALID_BRAND_PATTERNS:
        if pattern in name_lower or name_lower == pattern:
            return False
    
    # Skip if it looks like a Python structure
    if any(char in name for char in ["{", "}", "[", "]", "'", "dict", "list"]):
        return False
    
    # Skip if it starts with special characters
    if name.startswith(("{", "[", "'", '"', "(", "/", "//")):
        return False
    
    # Skip if it contains common explanation phrases
    explanation_phrases = ["were mentioned", "in the input", "in the given", "no brand", "not found"]
    if any(phrase in name_lower for phrase in explanation_phrases):
        return False
    
    return True


def format_brands_simple(brands_data: Any, max_brands: int = 5) -> str:
    """
    Format brands as simple comma-separated list for history table.
    
    Args:
        brands_data: Can be list of dicts with 'name' or list of strings
        max_brands: Maximum number of brands to display
        
    Returns:
        Simple comma-separated string of brand names
    """
    if not brands_data:
        return "—"
    
    if isinstance(brands_data, list):
        brand_names = []
        seen = set()
        
        for brand in brands_data:
            if isinstance(brand, dict):
                name = str(brand.get("name", "")).strip()
            elif isinstance(brand, str):
                name = brand.strip()
            else:
                continue
            
            if not name or not is_valid_brand_name(name):
                continue
            
            name = name.strip("'\"")
            name_lower = name.lower()
            
            # Skip duplicates
            if name_lower in seen:
                continue
            seen.add(name_lower)
            
            brand_names.append(name)
            
            if len(brand_names) >= max_brands:
                break
        
        if not brand_names:
            return "—"
        
        return ", ".join(brand_names)
    
    return "—"


def format_brands_display(brands_data: Any, max_brands: int = 10) -> str:
    """
    Format brands for display in the format: "Brand1 (score1) | Brand2 (score2) | ..."
    
    Args:
        brands_data: Can be list of dicts with 'name' and 'match_score', or list of strings
        max_brands: Maximum number of brands to display (to avoid clutter)
        
    Returns:
        Formatted string for display
    """
    if not brands_data:
        return "—"
    
    if isinstance(brands_data, list):
        if len(brands_data) > 0 and isinstance(brands_data[0], dict):
            # New format: list of dicts with scores
            # Remove duplicates and clean up
            seen = set()
            unique_brands = []
            
            for brand in brands_data:
                if not isinstance(brand, dict):
                    continue
                    
                name = str(brand.get("name", "")).strip()
                
                # Skip if not a valid brand name
                if not is_valid_brand_name(name):
                    continue
                
                # Normalize name (remove quotes if present)
                name = name.strip("'\"")
                
                # Skip duplicates (case-insensitive)
                name_lower = name.lower()
                if name_lower in seen:
                    continue
                seen.add(name_lower)
                
                score = brand.get("match_score", 0.0)
                # Ensure score is a number
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = 0.0
                
                unique_brands.append((name, score))
            
            # Sort by score (descending) and limit
            unique_brands.sort(key=lambda x: x[1], reverse=True)
            unique_brands = unique_brands[:max_brands]
            
            if not unique_brands:
                return "—"
            
            # Format as percentages with better spacing
            formatted = []
            for name, score in unique_brands:
                # Convert score (0.0-1.0) to percentage (0-100)
                percentage = int(round(score * 100))
                formatted.append(f"{name} ({percentage}%)")
            
            # Join with better spacing (extra spaces around separator)
            return "  |  ".join(formatted)
        else:
            # Old format: list of strings (backward compatibility)
            # Remove duplicates
            seen = set()
            unique = []
            for b in brands_data:
                b_str = str(b).strip().strip("'\"")
                if is_valid_brand_name(b_str) and b_str.lower() not in seen:
                    seen.add(b_str.lower())
                    unique.append(b_str)
            return ", ".join(unique[:max_brands])
    
    return "—"


def highlight_entities_in_text(text: str, entities: List[Dict]) -> str:
    """Highlight entities in text with HTML."""
    highlighted = text
    for entity in sorted(entities, key=lambda x: x.get("start", 0), reverse=True):
        start = entity.get("start", 0)
        end = entity.get("end", len(text))
        label = entity.get("label", "")
        entity_text = text[start:end]
        color = "#FF6B6B" if label == "ORG" else "#4ECDC4"
        highlighted = (
            highlighted[:start] +
            f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{entity_text}</span>' +
            highlighted[end:]
        )
    return highlighted


def render_performance_metrics(orchestration_result: Dict, tracker: PerformanceTracker):
    """Render comprehensive performance metrics dashboard."""
    st.subheader("⚡ Performance Metrics")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Latency",
            f"{orchestration_result.get('total_latency_ms', 0):.1f}ms",
            help="Total time for all agents to complete"
        )
    
    with col2:
        total_requests = len(tracker.metrics_history)
        st.metric(
            "Total Requests",
            total_requests,
            help="Number of agent executions tracked"
        )
    
    with col3:
        avg_latency = tracker.get_average_latency()
        st.metric(
            "Avg Latency",
            f"{avg_latency:.1f}ms",
            help="Average latency across all requests"
        )
    
    with col4:
        throughput = tracker.get_throughput()
        st.metric(
            "Throughput",
            f"{throughput:.2f} req/s",
            help="Requests per second"
        )
    
    # Cost comparison
    st.subheader("💰 Cost Efficiency Analysis")
    cost_data = calculate_cost_savings(total_requests, tokens_per_request=500)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Ollama Cost",
            f"${cost_data['ollama_cost_usd']}",
            help="Estimated cost using local Ollama"
        )
    with col2:
        st.metric(
            "Cloud API Cost (Avg)",
            f"${cost_data['cloud_cost_avg_usd']}",
            delta=f"-${cost_data['savings_avg']}",
            delta_color="inverse",
            help="Average cost using cloud APIs (OpenAI/Anthropic)"
        )
    with col3:
        st.metric(
            "Savings",
            f"{cost_data['savings_percentage']}%",
            help="Percentage saved vs cloud APIs"
        )
    
    # System resources
    st.subheader("🖥️ System Resources")
    mem_info = get_memory_usage()
    cpu_percent = get_cpu_usage()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Memory Usage", f"{mem_info['percent']:.1f}%")
        st.progress(mem_info['percent'] / 100)
    with col2:
        st.metric("CPU Usage", f"{cpu_percent:.1f}%")
        st.progress(cpu_percent / 100)
    
    # Agent-specific metrics
    st.subheader("📈 Agent Performance Breakdown")
    if orchestration_result.get("results"):
        agent_data = []
        for agent_name, result in orchestration_result["results"].items():
            if hasattr(result, 'latency_ms') and hasattr(result, 'confidence'):
                agent_data.append({
                    "Agent": agent_name,
                    "Latency (ms)": result.latency_ms,
                    "Confidence": result.confidence * 100,
                    "Status": "✓" if result.success else "✗"
                })
        
        if agent_data:
            df_agents = pd.DataFrame(agent_data)
            st.dataframe(df_agents, use_container_width=True, hide_index=True)
            
            # Visualization
            col1, col2 = st.columns(2)
            with col1:
                fig_latency = px.bar(
                    df_agents,
                    x="Agent",
                    y="Latency (ms)",
                    title="Agent Latency Comparison",
                    color="Agent"
                )
                st.plotly_chart(fig_latency, use_container_width=True)
            
            with col2:
                fig_confidence = px.bar(
                    df_agents,
                    x="Agent",
                    y="Confidence",
                    title="Agent Confidence Scores",
                    color="Agent",
                    range_y=[0, 100]
                )
                st.plotly_chart(fig_confidence, use_container_width=True)


def render_visualizations(results: Dict, prompt: str):
    """Render visualizations for results."""
    st.subheader("📊 Visualizations")
    
    # Extract data for visualization
    brands = []
    categories = []
    
    if "brand" in results and results["brand"].success:
        brand_result = results["brand"].result
        if isinstance(brand_result, dict):
            brands_data = brand_result.get("brands", [])
            brands = extract_brand_names(brands_data)
    
    if "category" in results and results["category"].success:
        category_result = results["category"].result
        if isinstance(category_result, dict):
            categories = [category_result.get("category", "")]
    
    col1, col2 = st.columns(2)
    
    # Brand distribution (if multiple queries in history)
    with col1:
        if st.session_state.query_history:
            brand_counts = defaultdict(int)
            for entry in st.session_state.query_history[-10:]:  # Last 10 queries
                if "brands" in entry and entry["brands"]:
                    # Extract brand names (handles both new and old format)
                    brand_names = extract_brand_names(entry["brands"])
                    for brand in brand_names:
                        brand_counts[brand] += 1
            
            if brand_counts:
                df_brands = pd.DataFrame({
                    "Brand": list(brand_counts.keys()),
                    "Count": list(brand_counts.values())
                })
                fig_brands = px.pie(
                    df_brands,
                    values="Count",
                    names="Brand",
                    title="Brand Distribution (Last 10 Queries)"
                )
                st.plotly_chart(fig_brands, use_container_width=True)
            else:
                st.info("Run more queries to see brand distribution")
        else:
            st.info("Run queries to see visualizations")
    
    # Category distribution
    with col2:
        if st.session_state.query_history:
            category_counts = defaultdict(int)
            for entry in st.session_state.query_history[-10:]:
                if "category" in entry and entry["category"]:
                    category_counts[entry["category"]] += 1
            
            if category_counts:
                df_categories = pd.DataFrame({
                    "Category": list(category_counts.keys()),
                    "Count": list(category_counts.values())
                })
                fig_categories = px.bar(
                    df_categories,
                    x="Category",
                    y="Count",
                    title="Category Distribution (Last 10 Queries)",
                    color="Category"
                )
                st.plotly_chart(fig_categories, use_container_width=True)
            else:
                st.info("Run more queries to see category distribution")


def render_brands_carousel(brands_data: Any, max_brands: int = 5):
    """
    Render top 5 brands in a clean grid layout without scrolling.
    Brands are color-coded: green (>= 80%), orange (50-80%), purple (< 50%).
    Scores are displayed as percentages.
    
    Args:
        brands_data: List of brand dicts with 'name' and 'match_score' (0-1 range)
        max_brands: Maximum number of brands to display (default: 5)
    """
    if not brands_data or not isinstance(brands_data, list):
        st.write("**Brands:** —")
        return
    
    if len(brands_data) == 0 or not isinstance(brands_data[0], dict):
        st.write("**Brands:** —")
        return
    
    # Filter and process brands, separating explicit brands and parent companies
    seen = set()
    unique_brands = []
    parent_companies = []
    parent_names_set = set()  # Track parent company names for labeling
    
    for brand in brands_data:
        if not isinstance(brand, dict):
            continue
            
        name = str(brand.get("name", "")).strip()
        
        # Skip if not a valid brand name
        if not is_valid_brand_name(name):
            continue
        
        name = name.strip("'\"")
        name_lower = name.lower()
        
        # Skip duplicates
        if name_lower in seen:
            continue
        seen.add(name_lower)
        
        score = brand.get("match_score", 0.0)
        brand_type = brand.get("type", "")
        try:
            score = float(score)
        except (ValueError, TypeError):
            score = 0.0
        
        # Separate parent companies to show them separately
        if brand_type == "parent":
            parent_companies.append((name, score))
            parent_names_set.add(name_lower)
        else:
            unique_brands.append((name, score, brand_type))  # Include type for reference
    
    # Sort by score (descending)
    unique_brands.sort(key=lambda x: x[1], reverse=True)
    parent_companies.sort(key=lambda x: x[1], reverse=True)
    
    # Prioritize showing parent companies if they exist (e.g., ByteDance for TikTok, Google for YouTube)
    # Strategy: Show top explicit brands, but always include parent companies even if it exceeds max_brands
    display_brands = [(name, score) for name, score, _ in unique_brands[:max_brands]]
    
    # Always add parent companies if they exist (they're important relationships)
    for parent_name, parent_score in parent_companies:
        # Check if parent is already in display list
        if not any(name.lower() == parent_name.lower() for name, _ in display_brands):
            display_brands.append((parent_name, parent_score))
    
    # Re-sort to maintain score order
    display_brands.sort(key=lambda x: x[1], reverse=True)
    unique_brands = display_brands
    
    if not unique_brands:
        st.write("**Brands:** —")
        return
    
    # Add custom CSS for compact brand cards
    st.markdown("""
    <style>
    .brand-card-container {
        padding: 8px 10px;
        border-radius: 6px;
        text-align: center;
        color: white;
        font-weight: 600;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        margin-bottom: 6px;
    }
    .brand-card-high {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    .brand-card-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }
    .brand-card-low {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**Brands:** ({len(unique_brands)} found)")
    
    # Use Streamlit columns for compact responsive grid layout (up to 5 columns)
    num_cols = min(5, len(unique_brands))
    if num_cols > 0:
        cols = st.columns(num_cols)
        
        for idx, (name, score) in enumerate(unique_brands):
            # Convert score to percentage
            percentage = int(round(score * 100))
            
            # Determine color class based on percentage
            if percentage >= 80:
                color_class = "brand-card-high"
            elif percentage >= 50:
                color_class = "brand-card-medium"
            else:
                color_class = "brand-card-low"
            
            # Render in column (use modulo to wrap if needed)
            col_idx = idx % num_cols
            with cols[col_idx]:
                # Check if this is a parent company to add a label
                is_parent = name.lower() in parent_names_set
                parent_label = " (Parent)" if is_parent else ""
                
                st.markdown(
                    f'<div class="brand-card-container {color_class}">'
                    f'<div style="font-size: 13px; margin-bottom: 2px; line-height: 1.2;">{name}{parent_label}</div>'
                    f'<div style="font-size: 11px; opacity: 0.95;">{percentage}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    
    # Show total count if there are more brands
    if len(brands_data) > max_brands:
        st.caption(f"*Showing top {max_brands} of {len(brands_data)} brands*")


def render_results(orchestration_result: Dict, prompt: str, show_metrics: bool):
    """Render comprehensive results with all agent outputs."""
    results = orchestration_result.get("results", {})
    
    st.subheader("🎯 Predictions")
    
    # Show processing timestamp for cache verification
    from datetime import datetime
    processing_time = datetime.now().strftime("%H:%M:%S")
    st.caption(f"⏱️ Processed at: {processing_time} | Latency: {orchestration_result.get('total_latency_ms', 0):.0f}ms")
    
    # Execution summary
    execution_order = orchestration_result.get("execution_order", [])
    if execution_order:
        st.caption(f"Execution order: {' → '.join(execution_order)}")
    
    # Brand results
    if "brand" in results and results["brand"].success:
        brand_result = results["brand"].result
        if isinstance(brand_result, dict):
            brands_data = brand_result.get("brands", [])
            confidence = brand_result.get("confidence", 0.0)
        else:
            brands_data = brand_result if isinstance(brand_result, list) else []
            confidence = results["brand"].confidence
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if brands_data:
                # Use modern grid display (top 5 brands, including parent companies)
                render_brands_carousel(brands_data, max_brands=5)
            else:
                st.write("**Brands:** —")
        with col2:
            st.metric("Confidence", f"{confidence * 100:.1f}%")
    
    # Category results
    if "category" in results and results["category"].success:
        category_result = results["category"].result
        if isinstance(category_result, dict):
            category = category_result.get("category", "")
            confidence = category_result.get("confidence", 0.0)
        else:
            category = category_result if isinstance(category_result, str) else ""
            confidence = results["category"].confidence
        
        # Clean category string - remove placeholders and invalid values
        if category:
            category = str(category).strip()
            # Remove common placeholders
            if category in ["{category}", "category", "None", "null", ""]:
                category = ""
            
            # Remove duplicates from category string (e.g., "Electronics, Electronics" -> "Electronics")
            if category:
                # Split by comma and clean each part
                category_parts = [cat.strip() for cat in category.split(",")]
                # Remove duplicates while preserving order (case-insensitive)
                seen = set()
                unique_parts = []
                for part in category_parts:
                    part_lower = part.lower()
                    if part_lower not in seen and part:  # Also skip empty parts
                        seen.add(part_lower)
                        unique_parts.append(part)
                category = ", ".join(unique_parts)
        
        if category:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("**Category:**", category)
            with col2:
                st.metric("Confidence", f"{confidence * 100:.1f}%")
        else:
            st.write("**Category:** —")
        
        # Confidence Score Evaluation (moved below Category)
        if "brand" in results and results["brand"].success:
            brand_result = results["brand"].result
            if isinstance(brand_result, dict):
                st.divider()
                
                # Center the confidence section
                st.subheader("📊 Confidence Score Evaluation")
                
                confidence = brand_result.get("confidence", 0.0)
                ner_entity_count = brand_result.get("ner_entity_count", 0)
                matched_entities = brand_result.get("matched_entities", 0)
                brands_list = brand_result.get("brands", [])
                
                # Centered layout with empty columns on sides
                col_empty1, col1, col2, col3, col4, col5, col_empty2 = st.columns([1, 1.5, 1.5, 1.5, 1.5, 1.5, 1])
                with col1:
                    st.metric("**Overall Confidence**", f"{confidence * 100:.1f}%")
                with col2:
                    st.metric("NER Entities", ner_entity_count)
                with col3:
                    st.metric("Matched Brands", matched_entities)
                with col4:
                    if brands_list:
                        scores = [b.get("match_score", 0.0) for b in brands_list if isinstance(b, dict)]
                        avg_score = sum(scores) / len(scores) if scores else 0.0
                        st.metric("Avg Match", f"{avg_score * 100:.1f}%")
                    else:
                        st.metric("Avg Match", "0%")
                with col5:
                    if confidence >= 0.8:
                        conf_level = "High 🟢"
                    elif confidence >= 0.5:
                        conf_level = "Medium 🟡"
                    else:
                        conf_level = "Low 🔴"
                    st.metric("Level", conf_level)
                
                # Detailed analysis in expander (collapsed by default)
                with st.expander("🔍 Detailed Confidence Analysis & Accuracy Factors", expanded=False):
                    st.write("**Confidence Factors:**")
                    
                    if ner_entity_count > 0:
                        ner_ratio = matched_entities / ner_entity_count if ner_entity_count > 0 else 0
                        st.write(f"• **NER Entity Match Rate:** {ner_ratio * 100:.1f}% ({matched_entities}/{ner_entity_count} entities matched)")
                    else:
                        st.write("• **NER Entity Match Rate:** No NER entities found (using DSPy inference)")
                    
                    if brands_list and prompt:
                        prompt_lower = prompt.lower()
                        explicit_brands = []
                        inferred_brands = []
                        
                        for b in brands_list:
                            if isinstance(b, dict):
                                brand_name = b.get("name", "")
                                brand_lower = brand_name.lower()
                                word_pattern = r'\b' + re.escape(brand_lower) + r'\b'
                                if re.search(word_pattern, prompt_lower):
                                    explicit_brands.append(brand_name)
                                elif b.get("is_inferred", False) or b.get("type") in ["parent", "competitor", "category_related"]:
                                    inferred_brands.append(brand_name)
                        
                        if explicit_brands:
                            st.write(f"• **Explicitly Mentioned Brands:** {', '.join(explicit_brands)} (100% match accuracy)")
                        if inferred_brands:
                            st.write(f"• **Inferred Brands:** {', '.join(inferred_brands)} (relationship-based)")
                        
                        if brands_list:
                            explicit_ratio = len(explicit_brands) / len(brands_list) if brands_list else 0
                            st.write(f"• **Explicit Mention Ratio:** {explicit_ratio * 100:.1f}% (higher = more accurate)")
                    
                    if len(brands_list) > 0:
                        high_confidence_brands = sum(1 for b in brands_list if isinstance(b, dict) and b.get("match_score", 0) >= 0.8)
                        perfect_matches = sum(1 for b in brands_list if isinstance(b, dict) and b.get("match_score", 0) >= 0.95)
                        st.write(f"• **High Confidence Brands:** {high_confidence_brands} out of {len(brands_list)} brands have ≥80% match")
                        st.write(f"• **Perfect Matches (100%):** {perfect_matches} brands explicitly mentioned in prompt")
                    
                    # Parent company detection
                    parent_brands = [b for b in brands_list if isinstance(b, dict) and b.get("type") == "parent"]
                    if parent_brands:
                        parent_names = [b.get("name") for b in parent_brands]
                        st.write(f"• **Parent Companies Detected:** {', '.join(parent_names)} (inferred relationships)")
                    
                    # Accuracy factors
                    st.write("\n**Accuracy Factors:**")
                    
                    # Calculate accuracy score
                    accuracy_factors = []
                    accuracy_score = 0.0
                    
                    if brands_list:
                        # Factor 1: Explicit mention ratio
                        if explicit_brands:
                            explicit_factor = len(explicit_brands) / len(brands_list)
                            accuracy_score += explicit_factor * 0.4
                            accuracy_factors.append(f"Explicit mentions: +{explicit_factor * 40:.1f}%")
                         
                        # Factor 2: Perfect match ratio
                        if perfect_matches > 0:
                            perfect_factor = perfect_matches / len(brands_list)
                            accuracy_score += perfect_factor * 0.3
                            accuracy_factors.append(f"Perfect matches: +{perfect_factor * 30:.1f}%")
                        
                        # Factor 3: High confidence ratio
                        if high_confidence_brands > 0:
                            high_conf_factor = high_confidence_brands / len(brands_list)
                            accuracy_score += high_conf_factor * 0.2
                            accuracy_factors.append(f"High confidence: +{high_conf_factor * 20:.1f}%")
                        
                        # Factor 4: NER entity support
                        if ner_entity_count > 0 and matched_entities > 0:
                            ner_factor = matched_entities / ner_entity_count
                            accuracy_score += ner_factor * 0.1
                            accuracy_factors.append(f"NER support: +{ner_factor * 10:.1f}%")
                    
                    for factor in accuracy_factors:
                        st.write(f"  - {factor}")
                    
                    # Display accuracy score
                    accuracy_percentage = min(100, accuracy_score * 100)
                    st.metric("**Overall Accuracy Score**", f"{accuracy_percentage:.1f}%")
                    
                    # Overall assessment
                    st.write("\n**Overall Assessment:**")
                    if confidence >= 0.8 and accuracy_score >= 0.7:
                        st.success("✅ **High confidence & High accuracy** - Strong brand signals with explicit mentions and reliable entity matching.")
                    elif confidence >= 0.8:
                        st.success("✅ **High confidence** - Strong brand signals detected, but some may be inferred relationships.")
                    elif confidence >= 0.5 and accuracy_score >= 0.5:
                        st.warning("⚠️ **Medium confidence & Medium accuracy** - Some brand signals detected with mixed explicit/inferred sources.")
                    elif confidence >= 0.5:
                        st.warning("⚠️ **Medium confidence** - Some brand signals detected, but mostly inferred relationships.")
                    else:
                        st.error("❌ **Low confidence** - Limited brand signals, mostly inferred from context. Results may be less reliable.")
    
    # Campaign results
    if "campaign" in results and results["campaign"].success:
        st.divider()
        st.subheader("📢 Campaign Insights")
        campaign_result = results["campaign"].result
        campaigns = campaign_result.get("campaigns", [])
        
        if campaigns:
            for i, campaign in enumerate(campaigns[:3], 1):
                with st.expander(f"Campaign {i}: {campaign.get('name', 'Unknown')}"):
                    st.write("**Type:**", campaign.get("type", "—"))
                    st.write("**Description:**", campaign.get("description", "—"))
                    st.write("**Status:**", campaign.get("status", "—"))
        
        st.metric("Confidence", f"{campaign_result.get('confidence', 0.0) * 100:.1f}%")
    
    # Reach results
    if "reach" in results and results["reach"].success:
        st.divider()
        st.subheader("👥 Reach Estimation")
        reach_result = results["reach"].result
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Estimated Reach",
                f"{reach_result.get('estimated_reach', 0):,}",
                help="Estimated number of users reached"
            )
        with col2:
            st.metric(
                "Reach Category",
                reach_result.get("reach_category", "—").title()
            )
        with col3:
            st.metric(
                "Confidence",
                f"{reach_result.get('confidence', 0.0) * 100:.1f}%"
            )
    
    # Brand Lift results
    if "brand_lift" in results and results["brand_lift"].success:
        st.divider()
        st.subheader("📈 Brand Lift Analysis")
        lift_result = results["brand_lift"].result
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Brand Lift",
                f"{lift_result.get('brand_lift_percentage', 0):.2f}%",
                help="Estimated brand lift percentage"
            )
        with col2:
            st.metric(
                "Lift Category",
                lift_result.get("lift_category", "—").title()
            )
        with col3:
            st.metric(
                "Confidence",
                f"{lift_result.get('confidence', 0.0) * 100:.1f}%"
            )
        
        key_factors = lift_result.get("key_factors", [])
        if key_factors:
            st.write("**Key Factors:**")
            for factor in key_factors:
                st.write(f"• {factor}")
    
    # Entity highlighting
    st.divider()
    st.subheader("🔍 Entity Analysis")
    ner_result = extract_entities(prompt)
    entities = ner_result.get("entities", [])
    
    if entities:
        highlighted_text = highlight_entities_in_text(prompt, entities)
        st.markdown(highlighted_text, unsafe_allow_html=True)
        
        with st.expander("Detailed Entity Information"):
            st.json(ner_result["entities"])
    else:
        st.info("No entities detected")
    
    # Performance metrics
    if show_metrics:
        st.divider()
        render_performance_metrics(orchestration_result, st.session_state.performance_tracker)


def process_batch(
    prompts: List[str],
    orchestrator: AgentOrchestrator,
    agent_config: Dict[str, bool],
    max_workers: int = 5
) -> List[Dict[str, Any]]:
    """
    Process multiple prompts in parallel using ThreadPoolExecutor.
    Optimized for performance with proper thread safety.
    """
    results = []
    
    # Ensure DSPy is configured before threading (critical for performance)
    configure_lm()
    
    def process_single(prompt: str) -> Dict[str, Any]:
        """Process a single prompt with error handling."""
        try:
            # Use the shared orchestrator instance (thread-safe for read operations)
            # Each agent internally handles thread safety for DSPy calls
            result = orchestrator.execute_pipeline(
                prompt=prompt,
                enable_brand=agent_config["brand"],
                enable_category=agent_config["category"],
                enable_campaign=agent_config["campaign"],
                enable_reach=agent_config["reach"],
                enable_brand_lift=agent_config["brand_lift"]
            )
            return {"prompt": prompt, "result": result, "success": True, "error": None}
        except Exception as e:
            return {"prompt": prompt, "result": None, "success": False, "error": str(e)}
    
    # Optimize max_workers: don't exceed prompt count or CPU cores
    import os
    cpu_count = os.cpu_count() or 4
    optimal_workers = min(max_workers, len(prompts), cpu_count)
    
    # Process in parallel with optimized worker count
    with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
        # Submit all tasks at once for better scheduling
        futures = {executor.submit(process_single, prompt): prompt for prompt in prompts}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        completed = 0
        total = len(prompts)
        
        # Process results as they complete (non-blocking)
        for future in as_completed(futures):
            try:
                result = future.result(timeout=300)  # 5 min timeout per prompt
                results.append(result)
                completed += 1
                progress_bar.progress(completed / total)
                status_text.text(f"Processed {completed}/{total} prompts...")
            except Exception as e:
                # Handle timeout or other errors
                prompt = futures[future]
                results.append({
                    "prompt": prompt,
                    "result": None,
                    "success": False,
                    "error": f"Timeout or error: {str(e)}"
                })
                completed += 1
                progress_bar.progress(completed / total)
                status_text.text(f"Processed {completed}/{total} prompts... (some failed)")
    
    return results


def render_batch_processing(orchestrator: AgentOrchestrator, agent_config: Dict[str, bool], samples: List[str]):
    """Render batch processing interface."""
    st.subheader("📦 Batch Processing")
    st.caption("Process multiple prompts simultaneously for faster throughput")
    
    # Batch input options
    batch_mode = st.radio(
        "Input method",
        ["Text Input (one per line)", "Select from Samples"],
        horizontal=True
    )
    
    prompts = []
    
    if batch_mode == "Text Input (one per line)":
        batch_input = st.text_area(
            "Enter prompts (one per line)",
            height=200,
            placeholder="Samsung Galaxy phone\nApple iPhone promotion\nLG TV comparison"
        )
        if batch_input:
            prompts = [p.strip() for p in batch_input.split("\n") if p.strip()]
    else:
        selected_samples = st.multiselect(
            "Select prompts to process",
            samples[:50],  # Limit for performance
            help="Select multiple prompts to process in batch"
        )
        prompts = selected_samples
    
    if not prompts:
        st.info("Enter or select prompts to begin batch processing")
        return
    
    st.info(f"Ready to process {len(prompts)} prompts")
    
    import os
    cpu_count = os.cpu_count() or 4
    max_workers = st.slider(
        "Parallel workers", 
        1, 
        min(10, cpu_count), 
        min(5, cpu_count),
        help=f"Number of concurrent requests (CPU cores: {cpu_count})"
    )
    
    if st.button("🚀 Process Batch", use_container_width=True):
        if len(prompts) > 20:
            st.warning(f"⚠️ Processing {len(prompts)} prompts may take a while. Consider reducing the batch size.")
        
        # Performance tip: Disable optional agents for faster batch processing
        optional_agents_enabled = agent_config.get("campaign", False) or agent_config.get("reach", False) or agent_config.get("brand_lift", False)
        if optional_agents_enabled and len(prompts) > 3:
            st.info("💡 **Tip**: Disable Campaign/Reach/Brand Lift agents in sidebar for faster batch processing")
        
        start_time = time.time()
        batch_results = process_batch(prompts, orchestrator, agent_config, max_workers)
        total_time = time.time() - start_time
        
        # Display results
        st.success(f"✅ Processed {len(batch_results)} prompts in {total_time:.2f} seconds")
        
        # Summary statistics
        successful = sum(1 for r in batch_results if r["success"])
        failed = len(batch_results) - successful
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", len(batch_results))
        with col2:
            st.metric("Successful", successful, delta=f"{successful/len(batch_results)*100:.1f}%")
        with col3:
            st.metric("Failed", failed)
        with col4:
            st.metric("Avg Time/Prompt", f"{total_time/len(batch_results):.2f}s")
        
        # Results table
        st.subheader("📊 Results Summary")
        results_data = []
        for i, result in enumerate(batch_results):
            prompt_preview = result["prompt"][:50] + "..." if len(result["prompt"]) > 50 else result["prompt"]
            if result["success"]:
                orchestration = result["result"]
                brand_result = orchestration.get("results", {}).get("brand", None)
                category_result = orchestration.get("results", {}).get("category", None)
                
                brands_display = "—"
                if brand_result and brand_result.success:
                    brand_data = brand_result.result
                    brands_data = brand_data.get("brands", []) if isinstance(brand_data, dict) else []
                    brands_display = format_brands_display(brands_data)
                
                category = ""
                if category_result and category_result.success:
                    category_data = category_result.result
                    category = category_data.get("category", "") if isinstance(category_data, dict) else ""
                
                results_data.append({
                    "Prompt": prompt_preview,
                    "Brands": brands_display,
                    "Category": category or "—",
                    "Latency (ms)": f"{orchestration.get('total_latency_ms', 0):.1f}",
                    "Status": "✓"
                })
            else:
                results_data.append({
                    "Prompt": prompt_preview,
                    "Brands": "—",
                    "Category": "—",
                    "Latency (ms)": "—",
                    "Status": f"✗ {result['error'][:30]}"
                })
        
        if results_data:
            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, use_container_width=True, hide_index=True)
            
            # Export option
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"batch_results_{int(time.time())}.csv",
                mime="text/csv"
            )
        
        # Add to history
        for result in batch_results:
            if result["success"]:
                orchestration = result["result"]
                history_entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prompt": result["prompt"][:100] + "..." if len(result["prompt"]) > 100 else result["prompt"],
                    "brands": [],
                    "category": "",
                    "latency_ms": orchestration.get("total_latency_ms", 0)
                }
                
                brand_result = orchestration.get("results", {}).get("brand", None)
                if brand_result and brand_result.success:
                    brand_data = brand_result.result
                    if isinstance(brand_data, dict):
                        history_entry["brands"] = brand_data.get("brands", [])
                
                category_result = orchestration.get("results", {}).get("category", None)
                if category_result and category_result.success:
                    category_data = category_result.result
                    if isinstance(category_data, dict):
                        history_entry["category"] = category_data.get("category", "")
                
                st.session_state.query_history.append(history_entry)


def render_query_history():
    """Render query history and analytics."""
    st.subheader("📜 Query History")
    
    if not st.session_state.query_history:
        st.info("No query history yet. Run some queries to see analytics.")
        return
    
    # Prepare history data for display
    history_data = []
    for entry in st.session_state.query_history[-20:]:  # Last 20 queries
        display_entry = entry.copy()
        
        # Format brands for display (simple format for history table)
        if "brands" in display_entry and display_entry["brands"]:
            brands_data = display_entry["brands"]
            # Use simple format for cleaner history table display
            display_entry["brands"] = format_brands_simple(brands_data, max_brands=5)
        else:
            display_entry["brands"] = "—"
        
        # Ensure category is a string
        if "category" in display_entry:
            category = display_entry.get("category", "")
            if not category or category == "":
                display_entry["category"] = "—"
        else:
            display_entry["category"] = "—"
        
        history_data.append(display_entry)
    
    # Show recent queries
    df_history = pd.DataFrame(history_data)
    
    # Reorder columns for better readability (timestamp, prompt, brands, category, latency)
    if not df_history.empty:
        preferred_order = ["timestamp", "prompt", "brands", "category", "latency_ms"]
        # Only include columns that exist
        columns_order = [col for col in preferred_order if col in df_history.columns]
        # Add any remaining columns
        remaining_cols = [col for col in df_history.columns if col not in columns_order]
        df_history = df_history[columns_order + remaining_cols]
        
        # Rename latency_ms for better display
        if "latency_ms" in df_history.columns:
            df_history = df_history.rename(columns={"latency_ms": "Latency (ms)"})
    
    st.dataframe(df_history, use_container_width=True, hide_index=True)
    
    # Analytics
    st.subheader("📊 Analytics Dashboard")
    
    # Most common brands
    if "brands" in df_history.columns:
        all_brands = []
        for brands_list in df_history["brands"].dropna():
            if isinstance(brands_list, list):
                for brand in brands_list:
                    # Extract brand name from dict or use string directly
                    if isinstance(brand, dict):
                        brand_name = brand.get("name", "")
                        if brand_name:
                            all_brands.append(brand_name)
                    elif isinstance(brand, str):
                        all_brands.append(brand)
        
        if all_brands:
            brand_counts = pd.Series(all_brands).value_counts().head(10)
            if len(brand_counts) > 0:
                fig = px.bar(
                    x=brand_counts.index.tolist(),
                    y=brand_counts.values.tolist(),
                    title="Top 10 Brands (All Time)",
                    labels={"x": "Brand", "y": "Count"}
                )
                st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application entry point."""
    load_dotenv()
    
    # Configure DSPy first (outside of cached function)
    configure_lm()
    
    # Load orchestrator
    if st.session_state.orchestrator is None:
        st.session_state.orchestrator = load_orchestrator()
    
    orchestrator = st.session_state.orchestrator
    samples = load_samples()
    preset_prompt, agent_config, show_metrics, show_history, processing_mode = render_sidebar(samples)
    
    # Main title
    st.title(STREAMLIT_TITLE)
    st.caption(
        "🤖 Multi-Agent System powered by Ollama & DSPy | "
        "⚡ Real-time Performance Metrics | 💰 Cost-Efficient Local AI"
    )
    
    # Route to appropriate processing mode
    if processing_mode == "Batch Processing":
        render_batch_processing(orchestrator, agent_config, samples)
    else:
        # Single query mode
        default_value = preset_prompt or st.session_state.get("last_prompt", "")
        with st.form(key="brand-form"):
            prompt = st.text_area(
                "Enter your prompt",
                value=default_value,
                height=150,
                placeholder="Describe a campaign, product, or marketing query...",
                help="Try queries like: 'Samsung Galaxy S24 launch campaign' or 'Apple iPhone promotion'"
            )
            submitted = st.form_submit_button("🚀 Classify Prompt", use_container_width=True)
    
        # Process query
        if submitted:
            if not prompt.strip():
                st.warning("⚠️ Please enter a prompt.")
                return
            
            st.session_state["last_prompt"] = prompt
            
            # Execute orchestration
            with st.spinner("🔄 Running agents..."):
                start_time = time.time()
                
                orchestration_result = orchestrator.execute_pipeline(
                    prompt=prompt,
                    enable_brand=agent_config["brand"],
                    enable_category=agent_config["category"],
                    enable_campaign=agent_config["campaign"],
                    enable_reach=agent_config["reach"],
                    enable_brand_lift=agent_config["brand_lift"]
                )
                
                # Track performance
                for agent_name, result in orchestration_result.get("results", {}).items():
                    if hasattr(result, 'latency_ms'):
                        metric = st.session_state.performance_tracker.start_agent(agent_name)
                        metric.finish(tokens_used=None, cache_hit=False)
                        metric.latency_ms = result.latency_ms
                        metric.confidence_score = result.confidence if hasattr(result, 'confidence') else None
                
                # Save to history
                history_entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "brands": [],
                    "category": "",
                    "latency_ms": orchestration_result.get("total_latency_ms", 0)
                }
                
                if "brand" in orchestration_result["results"] and orchestration_result["results"]["brand"].success:
                    brand_result = orchestration_result["results"]["brand"].result
                    if isinstance(brand_result, dict):
                        history_entry["brands"] = brand_result.get("brands", [])
                    else:
                        history_entry["brands"] = brand_result if isinstance(brand_result, list) else []
                
                if "category" in orchestration_result["results"] and orchestration_result["results"]["category"].success:
                    category_result = orchestration_result["results"]["category"].result
                    if isinstance(category_result, dict):
                        history_entry["category"] = category_result.get("category", "")
                    else:
                        history_entry["category"] = category_result if isinstance(category_result, str) else ""
                
                st.session_state.query_history.append(history_entry)
                # Keep only last 100 entries
                if len(st.session_state.query_history) > 100:
                    st.session_state.query_history = st.session_state.query_history[-100:]
            
            # Render results
            render_results(orchestration_result, prompt, show_metrics)
            
            # Visualizations
            if orchestration_result.get("results"):
                render_visualizations(orchestration_result["results"], prompt)
    
    # Query history tab
    if show_history:
        st.divider()
        render_query_history()
    
    # Footer
    st.divider()
    st.caption(
        "💡 **Tip:** Enable multiple agents to see orchestration in action. "
        "Campaign, Reach, and Brand Lift agents use context from Brand and Category agents."
    )


if __name__ == "__main__":
    main()
