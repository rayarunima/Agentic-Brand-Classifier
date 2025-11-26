# Prompt Classifier Agent

An AI-powered agent for classifying prompts using DSPy and ollama.

## System overview

The project wires multiple lightweight DSPy agents around a transformer-backed
spaCy NER model to classify free-form marketing prompts:

- `brand_extraction/entity_extractor.py` keeps a singleton `en_core_web_trf`
  pipeline (downloaded on first use) for token, POS, and entity signals.
- `agent/brand_agent.py` converts the spaCy entities into DSPy inputs so the
  LLM acts as a validator/normalizer for real brand names.
- `agent/category_agent.py` is a pure DSPy signature that infers the category
  string directly from the prompt.
- `main.py` loads sample prompts, configures DSPy to use `ollama/phi3`, and
  prints each prompt with its predicted brands and category.

External dependencies:
- **DSPy** for structured prompting (`dspy.Predict`, signatures).
- **Ollama** to serve the target model locally (`ollama serve` + `phi3`).
- **spaCy transformers stack** (`spacy`, `spacy-curated-transformers`,
  `en_core_web_trf`) for NER grounding.

## One-time setup

1. **Clone the repository**
2. **Install Python 3.9 with pyenv**
   ```bash
   pyenv install 3.9.23    # skip if already present
   cd Agentic-Brand-Classifier
   pyenv local 3.9.23      # creates .python-version
   ```
3. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```
4. **Install dependencies**
   ```bash
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   The *first* `python main.py` execution will automatically download the `en_core_web_trf` spaCy model (~500 MB). This only happens once because the model is cached locally afterwards.

5. **Configure Ollama locally**
   ```bash
   brew install ollama
   ollama pull phi3          # run once to cache the model
   ```

6. **Environment variables**
   - Create a `.env` file in the project root:
     ```dotenv
     DSPY_MODEL=ollama
     DSPY_MODEL_NAME=ollama/phi3
     SERPAPI_API_KEY=your-serpapi-key
     ```
   - `SERPAPI_API_KEY` is optional unless you enable evidence gathering inside the agents.

## Streamlit demo UI

We ship a Streamlit front end (`ui/app.py`) to make the demo interactive:

```bash
source .venv/bin/activate
ollama serve               # separate terminal
streamlit run ui/app.py
```

Features:
- Enter an ad prompt or pick from bundled samples.
- View predicted brands + category with intermediate NER entities.
- Inspect entity tokens, DSPy outputs, and future placeholders (reach/brand lift).

## One-command launcher

Use the helper script to run everything (CLI + optional Streamlit) without any SerpAPI dependency:

```bash
chmod +x scripts/run_local.sh          # one-time
./scripts/run_local.sh cli             # or ui | both
```

The script:
- Loads `.env` if present and forces DSPy to `ollama/phi3`.
- Checks for `ollama serve`; if it is not already running, the script starts it locally and tears it down on exit.
- Runs `python main.py`, launches Streamlit, or both depending on the mode you pass.

## Recurring runs (CLI)

1. **Activate the virtual environment**
   ```bash
   source .venv/bin/activate
   ```
2. **Start (or confirm) the Ollama service**
   ```bash
   ollama serve              # keep this running in a separate terminal
   ```
3. **Run the agent**
   ```bash
   python main.py
   ```
   You should see each sample prompt printed with the predicted brands & category. Make sure `ollama serve` is already running or the script will raise `OllamaException - [Errno 61] Connection refused`.

4. **VS Code debugging (optional)**
   - Use the provided `.vscode/launch.json` target **Run brand classifier** to execute `main.py` under the debugger. The configuration automatically loads the virtualenv interpreter and `.env` values.

## Enhancement roadmap

- **SpaCy caching/latency**: keep the `en_core_web_trf` pipeline warm via a simple
  service object so repeated `extract_entities` calls in Streamlit stay under
  50 ms.
- **Richer brand prompts**: inject brand/product context (specs, geography) into
  the DSPy signature to reduce hallucinated labels.
- **Evidence scoring toggle**: re-enable SerpAPI lookups when the API key is
  available and surface hit counts in the UI.
- **Telemetry**: log prompt + predictions (scrubbed) for demo QA and export
  samples quickly.
- **Additional agents**: wire `ReachEstimatorAgent` / `BrandLiftAgent` into the
  Streamlit layout once their implementations stabilize.

## Verification checklist

- `streamlit run ui/app.py` — Manual test the interactive flow (custom prompt +
  sample prompt, inspect intermediate entities).
- `python main.py` — Ensure CLI output still lists 5 prompts with brand/category.
- Optional: enable `SERPAPI_API_KEY` and confirm evidence toggles once wired.

## Open questions for the next sync

1. Should the demo capture telemetry (e.g., S3 or local JSON) for later review?
2. Which additional agents (reach/brand lift) should be prioritized for wiring
   into the UI before the executive demo date?
