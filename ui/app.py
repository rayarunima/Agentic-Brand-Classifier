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
    return choice


def show_results(prompt: str, brands: List[str], category: str) -> None:
    st.subheader("Predictions")
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
        with st.spinner("Running agents..."):
            brands = brand_agent.extract_brand(prompt)
            category = category_agent.extract_category(prompt)

        show_results(prompt, brands, category)


if __name__ == "__main__":
    main()

