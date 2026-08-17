# Thesis Reproducibility Package

This document maps every reported thesis table and figure to its source data, scripts, and reproduction steps. It supports examiner verification without changing reported numbers.

**Repository:** `CAS_RAG_Project`  
**Thesis:** `Thesis_Template_UTS_PHD`  
**Primary emotion evaluations:** Tutoring + Bullying only (`N=100` each)  
**Routing / cost audit:** six datasets (`N=600` pooled)

---

## Quick start

```bash
cd CAS_RAG_Project
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,openai,gemini]"
python -m spacy download en_core_web_sm
cp .env.example .env   # set OPENAI_API_KEY, optional GEMINI_API_KEY, CAS_RAG_OFFLINE_MODE=0
```

---

## Traceability matrix

| ID | Thesis location | Metric / content | Source workbook / data | Gold labels | Script / code | Reproduce |
|----|-----------------|------------------|------------------------|-------------|---------------|-----------|
| **tab:tutoring_100_post_refine** | Ch6 §Tutoring | Post-refinement routing + emotion (5 algos) | `Result_after_Prompt_refinments.xlsx` (repo root) — **canonical post-refinement source for table values**; routing columns unchanged from pre-refinement pass | `Expected_category`, `final label` (exclude `?` for emotion n=98) | Manual scoring contract aligned with Ch6 tables; five-pipeline outputs embedded in workbook | Open workbook; verify routing mistakes and emotion counts per algorithm. **Do not re-run prompts** if preserving approved numbers. |
| **tab:tutoring_100_pre_refine** | Ch6 | Pre-refinement baseline | Same family as tutoring workbook; pre-refinement emotion columns | `final label` (n=97) | Same | Compare pre/post columns in archived workbook |
| **tab:tutoring_100_improvement** | Ch6 | Emotion Δ after prompt refinement | Derived from pre vs post tables | Same | Derived | Subtract pre from post table values |
| **tab:bullying_100_emotion** | Ch6 §Bullying | Three-way routing + emotion (n=88) | `Results/comments_bullying_Models_Run_results.xlsx` | `Expected category` (routing), `Emotion label` (emotion; 12 blank excluded) | Scoring: strip parenthetical suffixes on gold (e.g. `Frustration(Borderline Case)`); case-insensitive match | Open workbook; recompute routing/emotion per Ch6 contract |
| **tab:bullying_100_binary** | Ch6 | Binary CD vs not + workbook emotion % | Same bullying workbook | Same | Binary: gold CD vs predicted CD | Count binary matches on 100 rows |
| **tab:cross_dataset_comparison_framework** | Ch6 | Tutoring vs Bullying synthesis | Derived from above tables | Same | Derived | No separate source file |
| **tab:casrag_cost** | Ch6 §eval_results | Pooled cost / RAG audit (600 comments) | Six-dataset gold exports (see **Missing artifacts**) + CAS-RAG Layer-1 predictions | Layer-1 `expected_category` per video | Manual cost model in Ch6 (`retrieval=1`, LLM+ctx=2`, LLM=1`) | Apply formula: `n_rag=199`, `cost_gate=998`, `cost_no_gate=1800`, saving 44.6% |
| **Pooled Layer-1 accuracies** | Ch6, abstract, Ch7 | 69.5%, macro-F1 69.8%, per-topic % | Six-dataset CAS-RAG Layer-1 labels vs gold | Per-video xlsx in `VERY_IMPORTANT_RESULTS/` (**external to repo**) | `SemanticSimilarityEngine` or archived predictions | See **Missing artifacts** |
| **fig1–fig8, fig6_dataset_overview** | Ch6 | Cost, confusion matrix, dashboard, by-topic | Generated from pooled six-dataset audit | Same as cost table | `generate_figures.py` (**external**; see below) | Regenerate when gold + predictions available |
| **fig4_seven_metrics** | Ch5 | Seven-metric walkthrough | `figures/fig4_seven_metrics.png` (thesis); computed from `similarity.py` | Walkthrough comments in Ch5 | `src/cas_rag/semantic/similarity.py` | Run walkthrough examples in Ch5 |
| **Ch4 pipeline diagrams** | Ch4 | Architecture | `figures/ch4/*.drawio.png` | N/A | Static assets | Visual only |
| **sec:eval_rq1** | Ch6 | RQ1 consensus audit | Dual-provider comparison xlsx | Same gold emotion columns as above | `scripts/run_rq1_consensus_audit.py` | See **RQ1 audit** below |
| **Appendix C summary table** | Appendix C | Numeric index | Compiled from Ch6 tables | Same | Derived | Cross-check against Ch6 |

---

## RQ1 consensus audit (offline)

**Purpose:** Operationalise Model 1 / RQ1 without changing CAS-RAG runtime or primary emotion tables.

### Step 1 — Dual-provider comparison export (if not already present)

```bash
python scripts/run_five_algorithm_comparison.py \
  --input <gold_labelled_input.xlsx> \
  --output data/evaluation_outputs/<dataset>_comparison_openai_gemini.xlsx \
  --video-slug video_02_education_tutoring_diner \
  --llms openai gemini --kpi
```

For Bullying: `--video-slug video_01_bullying_bystander_bus_stop`.

**Note:** Existing `Results/comments_bullying_Models_Run_results.xlsx` contains primarily OpenAI (`ChatGpt`) columns. For full RQ1 audit, regenerate with `--llms openai gemini` or supply a workbook with both `*_emotion_label_ChatGpt` and `*_emotion_label_Gemini` columns.

### Step 2 — Run consensus audit

```bash
python scripts/run_rq1_consensus_audit.py \
  --input data/evaluation_outputs/<dataset>_comparison_openai_gemini.xlsx \
  --output Results/rq1_consensus_<dataset>.xlsx \
  --dataset <tutoring|bullying> \
  --algorithms CAS-RAG Alg1 Alg2 Alg3 Alg4
```

**Outputs:**

- `Consensus_Audit` — per-comment provider labels, consensus label, agreement ratio, reliability tier, disagreement flag, hallucination proxy
- `RQ1_Summary` — dataset-level unanimous rate, disagreement rate, provider vs consensus accuracy

**Code:**

- `src/cas_rag/evaluation/consensus.py`
- `src/cas_rag/experiments/consensus_audit.py`

---

## Tutoring post-refinement reproduction

| Item | Detail |
|------|--------|
| **Table** | `tab:tutoring_100_post_refine` |
| **Canonical workbook in repo** | `Result_after_Prompt_refinments.xlsx` |
| **Gold routing** | `Expected_category` |
| **Gold emotion** | `final label` (98 evaluable after excluding `?`) |
| **Reported CAS-RAG** | Routing 72.0% (28/100); emotion 77.6% (22/98) |
| **Prompt refinement** | Documented in Ch6 §`sec:eval_prompt_refine`; prompts in `resources/datasets/Experiments-data/video_02_.../prompts/` |
| **Important** | Table values are **approved frozen results**. Re-running LLM APIs may differ slightly; use archived workbook to verify reported numbers. |

---

## Bullying reproduction

| Item | Detail |
|------|--------|
| **Table** | `tab:bullying_100_emotion`, `tab:bullying_100_binary` |
| **Workbook** | `Results/comments_bullying_Models_Run_results.xlsx` |
| **Gold routing** | `Expected category` |
| **Gold emotion** | `Emotion label` (88 evaluable; 12 blank) |
| **Scoring contract** | Three-way routing on 100 rows; emotion with parenthetical normalisation on gold |

---

## Six-dataset Layer-1 / cost reproduction

| Item | Detail |
|------|--------|
| **Tables / figures** | `tab:casrag_cost`, fig1–fig3, fig5, fig7, fig8 |
| **Gold data** | ~100 comments × 6 videos; filenames listed in `resources/datasets/Experiments-data/README.md` |
| **Repo status** | Per-video **prompts, prototypes, scenarios** are in repo; **gold-labelled xlsx per video are not** in repo (referenced as `VERY_IMPORTANT_RESULTS/*.xlsx`) |
| **Cost formula** | Ch6 §Metrics: without gate `600×3=1800`; with gate `199×3+401×1=998`; saving **44.6%**; wrong retrieval **51/199** |
| **Action for submission** | Bundle `VERY_IMPORTANT_RESULTS/` (or equivalent) with thesis submission **or** document exact file hashes/paths used for the 600-comment audit |

---

## Missing artifacts (examiner-facing)

| Artifact | Referenced in | Status in repo | Recommendation |
|----------|---------------|----------------|----------------|
| `VERY_IMPORTANT_RESULTS/*.xlsx` | Ch6, Appendix A | **Missing** | Add to submission package (read-only) |
| `generate_figures.py` | Ch6 index, Appendix A | **Missing** | Add script or archive PDF figures with generation notes |
| `comparison_dataset.py` (26-comment pilot) | Ch6 | **Missing** | Archive pilot CSV/xlsx or remove script reference |
| Tutoring dual-provider xlsx for RQ1 | Ch6 §RQ1 | **Partial** | Regenerate with `--llms openai gemini` |
| Six per-video gold xlsx in `Experiments-data/` | README | **Missing** | Copy from `VERY_IMPORTANT_RESULTS` or document external path |
| Thesis figure PDFs (`fig1`–`fig8`) | Ch6 | **Not in thesis repo tree** | Ensure figures ship with thesis PDF build |

---

## Submission checklist

- [ ] `CAS_RAG_Project` tagged commit includes `consensus.py`, `consensus_audit.py`, `run_rq1_consensus_audit.py`, tests
- [ ] `Result_after_Prompt_refinments.xlsx` (Tutoring tables) in repo or submission bundle
- [ ] `Results/comments_bullying_Models_Run_results.xlsx` (Bullying tables)
- [ ] Six-dataset gold workbooks available to examiner (bundle or documented path)
- [ ] `THESIS_REPRODUCIBILITY.md` (this file) in repo root
- [ ] Thesis Ch6 §`sec:eval_rq1` describes offline audit scope
- [ ] Ch1/Ch3 RQ1 and Model 1 wording aligned to offline audit (not in-pipeline ensemble)
- [ ] Ch6 Model~1 naming collision fixed (linguistic-first **prototype**)
- [ ] RQ1 audit outputs (`Results/rq1_consensus_*.xlsx`) generated for Tutoring + Bullying when dual-provider columns exist
- [ ] `.env.example` documents `OPENAI_API_KEY`, `GEMINI_API_KEY`, `CAS_RAG_LLM_PROVIDER`
- [ ] Thesis PDF compiles with updated Ch6 `\input{evaluation_rq1}`

---

## RQ1 claim → implementation classification

| Claim | Status |
|-------|--------|
| Multi-provider emotion comparison (OpenAI/Gemini) | **B — Partial** (`comparison_runner.py` + audit) |
| Majority-vote consensus | **A — Implemented offline** (`consensus.py`) |
| Agreement scoring | **A — Implemented offline** |
| Disagreement detection | **A — Implemented offline** |
| Reliability tiers (HIGH/SPLIT/etc.) | **A — Implemented offline** |
| Gold-linked hallucination proxy | **B — Partial** (disagreement proxy, not full detector) |
| In-pipeline filtering | **D — Future work** |
| Weighted / confidence-weighted consensus | **D — Future work** |
| Three-provider GPT-4+Gemini+LLaMA ensemble | **B — Partial** (OpenAI+Gemini; Ollama optional) |
| Model 1 → CAS-RAG gating integration | **D — Conceptual** |

---

## Contact points for examiners

1. **Primary system evaluated:** CAS-RAG + Algorithms 1–4 (Ch6 Tutoring/Bullying tables).  
2. **RQ1 evaluated separately:** Offline consensus audit (Ch6 §RQ1); does not change those tables.  
3. **Frozen results:** Post-refinement tutoring and bullying numbers come from archived workbooks listed above.
