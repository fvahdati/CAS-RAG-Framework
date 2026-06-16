# CAS-RAG Research Platform

**CAS-RAG** (Context-Aware Semantic Retrieval-Augmented Generation) is a research-grade NLP platform for **non-clinical emotional reaction analysis** on user comments related to social-experiment videos.

Aligned with the thesis methodology (Chapter 4), the platform implements:

| Pipeline | Description |
|----------|-------------|
| **Algorithm 1** | LLM–LLM (P1 → P2 / P3 → P4 + human verification) |
| **Algorithm 2** | spaCy–LLM (linguistic gate → P2 / P3 → P4 + human verification) |
| **Algorithm 3** | spaCy + Semantic Similarity (P5/embedding + P2 / P7 / P6 + human verification) |
| **Algorithm 4** | LLM + Semantic Similarity (σ → P6 / ROUTE_TO_RAG → P2 / P7) |
| **CAS-RAG** | Three-way semantic routing (Explicit / Ambiguous / Context-Dependent) + selective RAG |

## Architecture

```
CAS_RAG_Project/
├── api/                 # FastAPI REST backend
├── dashboard/           # Streamlit analytics dashboard
├── config/              # YAML configuration
├── resources/           # Prompts, prototypes, datasets, scenarios
├── data/                # Runtime outputs (evaluations, analytics)
├── src/cas_rag/         # Core library (modular subsystems)
├── scripts/             # CLI experiment runners
└── tests/               # Unit & integration tests
```

## Quick start

```bash
cd CAS_RAG_Project
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dashboard,api]"
cp .env.example .env
python -m spacy download en_core_web_sm
python scripts/run_experiment.py --algorithm cas_rag --input resources/datasets/sample/comments.csv
uvicorn api.main:app --reload --port 8000
streamlit run dashboard/app.py
```

## LLM providers (real models by default)

1. Copy `.env.example` to `.env` and set **`OPENAI_API_KEY`** (or Gemini / Ollama — see below).
2. Default provider is **`openai`** (ChatGPT). Alternatives: `gemini`, `ollama` (local Llama/Meta via [Ollama](https://ollama.com)).

```bash
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-...
pip install -e ".[openai]"   # or [gemini] for Google
python scripts/run_experiment.py --algorithm cas_rag --input resources/datasets/Experiments-data/comments.csv
```

| Provider | Env vars | Install extra |
|----------|----------|---------------|
| `openai` | `OPENAI_API_KEY`, `OPENAI_MODEL` | `pip install -e ".[openai]"` |
| `gemini` | `GEMINI_API_KEY`, `GEMINI_MODEL` | `pip install -e ".[gemini]"` |
| `ollama` | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Ollama daemon + `ollama pull llama3.2` |

**Mock LLM** is disabled for experiments. It is only available when `CAS_RAG_ALLOW_MOCK=1` (used automatically in `pytest`).

## Human verification

Ambiguous comments are **always** routed to the human verification queue (`data/human_verification_queue.json`). Processing pauses until a reviewer submits a decision via API or CLI.

## License

Research / academic use — see LICENSE.
