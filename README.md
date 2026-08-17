# CAS-RAG Research Platform

**CAS-RAG** (Context-Aware Selective RAG / RAG-GATE) is the reference implementation of the thesis methodology: **non-clinical emotional reaction analysis** on YouTube comments under videos related to adolescent topics, with **selective retrieval**.

## Study design (must match the thesis)

| Evaluation layer | Coverage |
|------------------|----------|
| **Routing (Layer 1)** | Six annotated datasets (~100 comments each; 600 total) |
| **Emotion classification** | Tutoring/Diner and Bullying/Bus Stop only (\(N{=}100\) each); Algorithms 1--4 vs CAS-RAG |
| **Not emotion-evaluated** | Mental health, teen safety, teen pregnancy, drug prevention (routing / cost audit only) |

## Pipelines

| Pipeline | Gate | Emotion prompts |
|----------|------|-----------------|
| **Algorithm 1** | LLM P1 (unknown-reference) | P2 if RAG; else P3 then P4 + human verification |
| **Algorithm 2** | spaCy linguistic gate | P2 if RAG; else P3 then P4 + human verification |
| **Algorithm 3** | spaCy, then two-way semantic (Explicit / Ambiguous) | P2 / P7 / P6 + human verification |
| **Algorithm 4** | P1, then two-way semantic (Explicit / Ambiguous) | P2 / P7 / P6 + human verification |
| **CAS-RAG** | Seven-metric three-way semantic routing (no P1/P5 at the gate); Layer 2 spaCy on Context-Dependent only | P7 Explicit; P6 Ambiguous; P2 if RAG-resolvable |

Layer 1 defaults: SentenceTransformer `all-MiniLM-L6-v2`, seven-metric weights in `config/default.yaml`, margins \(m{=}0.12\), \(b{=}0.08\). No dataset-specific phrase lists on the production routing path.

## Architecture

```
CAS_RAG_Project/
├── api/                 # FastAPI REST backend
├── dashboard/           # Streamlit analytics dashboard
├── config/              # YAML configuration (margins, embeddings, top_k)
├── resources/           # Prompts, prototypes, six video packages
├── data/                # Runtime outputs (evaluations, analytics)
├── src/cas_rag/         # Core library
├── scripts/             # CLI experiment runners
└── tests/               # Unit & integration tests
```

Six video folders live under `resources/datasets/Experiments-data/video_01_...` through `video_06_...`. Each contains `scenario_full.txt`, `script_corpus.txt`, `prototypes_semantic.json`, and prompts P1--P7.

## Quick start

```bash
cd CAS_RAG_Project
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,openai,dashboard,api]"
cp .env.example .env
python -m spacy download en_core_web_sm
python scripts/run_experiment.py --algorithm cas_rag \
  --input resources/datasets/Experiments-data/comments.csv \
  --video-slug video_02_education_tutoring_diner
python scripts/run_five_algorithm_comparison.py \
  --input resources/datasets/Experiments-data/comments_education_tutoring_diner.xlsx \
  --video-slug video_02_education_tutoring_diner --kpi
```

Thesis runs require `CAS_RAG_OFFLINE_MODE=0` (or unset) so embeddings use the trained encoder.

## LLM providers

Default: **`openai`** (`gpt-4o-mini`). Alternatives: `gemini`, `ollama`. Mock LLM is tests-only (`CAS_RAG_ALLOW_MOCK=1` / pytest).

## RQ1 offline consensus audit

Model 1 / RQ1 is an **offline** layer (does not change CAS-RAG runtime):

```bash
# 1. Dual-provider comparison export
python scripts/run_five_algorithm_comparison.py --llms openai gemini --kpi \
  --input <gold_workbook.xlsx> --video-slug video_02_education_tutoring_diner

# 2. Consensus audit → Consensus_Audit + RQ1_Summary sheets
python scripts/run_rq1_consensus_audit.py \
  --input data/evaluation_outputs/<comparison>.xlsx \
  --output Results/rq1_consensus_tutoring.xlsx --dataset tutoring
```

See `THESIS_REPRODUCIBILITY.md` for full table/figure traceability.

## Human verification

Ambiguous P4/P6 outcomes and Context-Dependent **RAG-infeasible** (video mismatch) comments are queued in `data/human_verification_queue.json`.

## License

Research / academic use — see LICENSE.
