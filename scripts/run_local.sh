#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-cli}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [cli|ui|both]

Runs the Agentic Brand Classifier end-to-end with a local Ollama model.
- cli  : execute python main.py against bundled prompts (default)
- ui   : launch the Streamlit app
- both : run the CLI first, then start the Streamlit app

The script never calls SerpAPI and configures DSPy to use ollama/phi3 locally.
EOF
}

case "${MODE}" in
  cli|ui|both) ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Invalid mode: ${MODE}" >&2; usage; exit 1 ;;
esac

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing dependency '$1'. Please install it first." >&2
    exit 1
  fi
}

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Missing dependency 'python3' (or 'python'). Please install it first." >&2
  exit 1
fi

require_cmd streamlit
require_cmd ollama
require_cmd curl

if ! command -v jq >/dev/null 2>&1; then
  echo "[run_local] Tip: install 'jq' for easier Ollama debugging (optional)." >&2
fi

if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  set -a
  source ".env"
  set +a
fi

export DSPY_MODEL="${DSPY_MODEL:-ollama}"
export DSPY_MODEL_NAME="${DSPY_MODEL_NAME:-ollama/phi3}"
export DSPY_MAX_TOKENS="${DSPY_MAX_TOKENS:-4096}"
export DSPY_TEMPERATURE="${DSPY_TEMPERATURE:-0.2}"
unset SERPAPI_API_KEY

OLLAMA_PID=""
cleanup() {
  if [[ -n "${OLLAMA_PID}" ]] && ps -p "${OLLAMA_PID}" >/dev/null 2>&1; then
    kill "${OLLAMA_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

wait_for_ollama() {
  for _ in {1..30}; do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

ensure_ollama() {
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "[run_local] Ollama already running."
    return
  fi

  if pgrep -f "ollama serve" >/dev/null 2>&1; then
    echo "[run_local] Found an existing 'ollama serve' process; waiting for it."
    if ! wait_for_ollama; then
      echo "[run_local] Existing Ollama process never responded on port 11434." >&2
      exit 1
    fi
    return
  fi

  echo "[run_local] Starting local Ollama server..."
  ollama serve >"${PROJECT_ROOT}/.ollama.log" 2>&1 &
  OLLAMA_PID=$!
  if ! wait_for_ollama; then
    echo "[run_local] Ollama failed to become ready. See .ollama.log for details." >&2
    exit 1
  fi
  echo "[run_local] Ollama is ready (pid ${OLLAMA_PID})."
}

run_cli() {
  echo "[run_local] Running CLI via python main.py"
  "${PYTHON_BIN}" main.py
}

run_ui() {
  echo "[run_local] Launching Streamlit UI (Ctrl+C to exit)"
  streamlit run ui/app.py
}

ensure_ollama

case "${MODE}" in
  cli) run_cli ;;
  ui) run_ui ;;
  both)
    run_cli
    echo "[run_local] CLI finished. Starting Streamlit next..."
    run_ui
    ;;
esac

