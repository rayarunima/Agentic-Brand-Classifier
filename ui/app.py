import json
import os
from pathlib import Path
from typing import List

import dspy
import streamlit as st
from dotenv import load_dotenv

from agent.brand_agent import BrandAgent
from agent.category_agent import CategoryAgent
from brand_extraction.entity_extractor import extract_entities


STREAMLIT_TITLE = "Agentic Brand Classifier"
SAMPLES_PATH = Path("prompts") / "sample_prompts.json"


def configure_lm() -> None:
    """Configure DSPy once per process."""
    model_name = os.getenv("DSPY_MODEL_NAME", "ollama/phi3")
    max_tokens = int(os.getenv("DSPY_MAX_TOKENS", "4096"))
    temperature = float(os.getenv("DSPY_TEMPERATURE", "0.2"))

    dspy.configure(
        lm=dspy.LM(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    )


@st.cache_resource(show_spinner=False)
def load_agents():
    configure_lm()
    return BrandAgent(), CategoryAgent()


@st.cache_data(show_spinner=False)
def load_samples() -> List[str]:
    if not SAMPLES_PATH.exists():
        return []
    with open(SAMPLES_PATH, "r") as handle:
        return json.load(handle)


def render_sidebar(samples: List[str]) -> str:
    st.sidebar.header("Prompt presets")
    if not samples:
        st.sidebar.info("No sample prompts found in prompts/sample_prompts.json")
        return ""

    choice = st.sidebar.selectbox("Pick a sample", options=[""] + samples, index=0)
    if choice:
        st.sidebar.success("Loaded sample prompt.")

    # Agent toggles
    st.sidebar.divider()
    st.sidebar.subheader("Agents")

    # Primary agents (fully wired)
    st.sidebar.toggle(
        "Brand agent",
        key="use_brand_agent",
        value=st.session_state.get("use_brand_agent", True),
        help="Enable or disable the Brand extraction agent.",
    )
    st.sidebar.toggle(
        "Category agent",
        key="use_category_agent",
        value=st.session_state.get("use_category_agent", True),
        help="Enable or disable the Category extraction agent.",
    )

    # WIP agents (UI only, no backend calls yet)
    campaign_on = st.sidebar.toggle(
        "Campaign agent (WIP)",
        key="use_campaign_agent",
        value=st.session_state.get("use_campaign_agent", False),
    )
    reach_on = st.sidebar.toggle(
        "Reach agent (WIP)",
        key="use_reach_agent",
        value=st.session_state.get("use_reach_agent", False),
    )
    brand_lift_on = st.sidebar.toggle(
        "Brand lift agent (WIP)",
        key="use_brand_lift_agent",
        value=st.session_state.get("use_brand_lift_agent", False),
    )

    # If any WIP toggle is turned on, immediately show info and revert to OFF
    if campaign_on or reach_on or brand_lift_on:
        st.sidebar.info("Work in Progress – have patience")
        if campaign_on:
            st.session_state["use_campaign_agent"] = False
        if reach_on:
            st.session_state["use_reach_agent"] = False
        if brand_lift_on:
            st.session_state["use_brand_lift_agent"] = False

    return choice


def show_results(
    prompt: str, brands: List[str], category: str, active_agents: List[str] | None = None
) -> None:
    st.subheader("Predictions")
    active_agents = active_agents or []
    agent_count = len(active_agents)
    agent_label = ", ".join(active_agents) if active_agents else "None"

    # Simple modern badge-style summary of which agents ran
    badge = f"{agent_count} agent{'s' if agent_count != 1 else ''}"
    st.markdown(
        f"**Active agents:** {agent_label} &nbsp;&nbsp; "
        f"<span style='padding:2px 8px;border-radius:999px;border:1px solid rgba(255,255,255,0.15);"
        f"font-size:0.8rem;opacity:0.8;'>{badge}</span>",
        unsafe_allow_html=True,
    )

    st.write("**Brands:**", ", ".join(brands) if brands else "—")
    st.write("**Category:**", category or "—")

    st.divider()
    st.subheader("Intermediate signals")
    ner = extract_entities(prompt)
    with st.expander("Named entities"):
        st.json(ner["entities"])

    with st.expander("Tokens / POS"):
        st.json(ner["tokens"][:50])

    st.info(
        "Reach & brand-lift estimators are not wired yet, but the layout leaves room "
        "to plug them in for the demo."
    )


def main():
    load_dotenv()
    brand_agent, category_agent = load_agents()
    samples = load_samples()
    preset_prompt = render_sidebar(samples)

    use_brand_agent = st.session_state.get("use_brand_agent", True)
    use_category_agent = st.session_state.get("use_category_agent", True)

    st.title(STREAMLIT_TITLE)
    st.caption(
        "Explore how spaCy NER + DSPy validators collaborate to detect brands and "
        "product categories from marketing prompts."
    )

    default_value = preset_prompt or st.session_state.get("last_prompt", "")
    with st.form(key="brand-form"):
        prompt = st.text_area(
            "Enter your prompt",
            value=default_value,
            height=200,
            placeholder="Describe a campaign or product positioning...",
        )
        submitted = st.form_submit_button("Classify prompt")

    if submitted:
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        st.session_state["last_prompt"] = prompt
        brands: List[str] = []
        category: str = ""
        active_agents: List[str] = []

        # Only invoke agents that are currently enabled
        with st.spinner("Running agents..."):
            if use_brand_agent:
                brands = brand_agent.extract_brand(prompt)
                active_agents.append("Brand")
            if use_category_agent:
                category = category_agent.extract_category(prompt)
                active_agents.append("Category")

        show_results(prompt, brands, category, active_agents)


if __name__ == "__main__":
    main()

