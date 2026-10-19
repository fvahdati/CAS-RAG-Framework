# Thesis reproducibility notes

This note maps the reported thesis tables to files in [CAS-RAG-Framework](https://github.com/fvahdati/CAS-RAG-Framework). It is a guide for examiners. The submitted PhD thesis is the authoritative source for reported results. Re-running the language-model APIs can change individual labels, so the archived workbooks below should be used when checking the published numbers.

**Emotion evaluation:** Tutoring/Diner and Bullying/Bus Stop only (`N=100` each).  
**Routing and cost audit:** six topic datasets (`N=600` pooled).

---

## Public release notes

### Included

- The CAS-RAG package, Algorithms 1–4, tests, and the offline RQ1 audit script.
- Per-video scenarios, transcript corpora, prototypes, and prompts P1–P7 under `resources/datasets/Experiments-data/`.
- Gold-labelled comment workbooks for all six videos (100 comments each). Each file contains the comment text and the gold routing label `expected_category`. In these files the third routing class is stored as `vague` (Ambiguous in the thesis) and context-dependent comments are stored as `context_dependent`.
- The Tutoring post-refinement workbook and the Bullying five-pipeline workbook, including the model outputs used for those two in-depth tables.
- `ETHICS.md` and `CITATION.cff`.

### Intentionally excluded

- The prediction workbooks that contain the CAS-RAG Layer-1 outputs used for the pooled six-dataset routing accuracies, confusion matrix, retrieval counts, and cost figures. Those workbooks are not part of the public release. The gold comment files in this repository do not, by themselves, reproduce those pooled figures: their model-output columns are empty.
- Thesis figure PDFs (fig1–fig8) and the script that drew them.
- The large background comment pool (152,594 comments) and the early 26-comment development pilot.
- Stored RQ1 consensus workbooks. The audit script is included. The thesis does not treat a saved consensus spreadsheet as a reported result.
- API keys. Copy `.env.example` to `.env` and supply keys locally.

Reported numbers in the thesis remain the record of the study. This repository is the public code and data release that supports that record.

---

## Gold-labelled comment workbooks

These files are in the repository. They are the annotated comment sets. They are not the pooled prediction archive.

| Topic | File | Rows | Gold routing column |
|-------|------|------|---------------------|
| Bullying | `resources/datasets/Experiments-data/comments_bullying.xlsx` | 100 | `expected_category` |
| Education (Tutoring/Diner) | `resources/datasets/Experiments-data/comments_education_tutoring_diner.xlsx` | 100 | `expected_category` |
| Suicide | `resources/datasets/Experiments-data/comments_Suicide.xlsx` | 100 | `expected_category` |
| TeenSafety | `resources/datasets/Experiments-data/comments-TeenSafety.xlsx` | 100 | `expected_category` |
| TeenPregnancy | `resources/datasets/Experiments-data/comments_teenPregnancy.xlsx` | 100 | `expected_category` |
| Teenagers-Drugs | `resources/datasets/Experiments-data/comments_teenagers drugs.xlsx` | 100 | `expected_category` |

`resources/datasets/Experiments-data/comments.csv` is a six-row example for a smoke test. It is not one of the six annotated datasets.

Each video folder (`video_01_` through `video_06_`) also contains `scenario_full.txt`, `script_corpus.txt`, `prototypes_semantic.json`, and prompts P1–P7.

---

## Traceability

| Thesis item | What to open in this repository | How to check |
|-------------|-------------------------------|--------------|
| Tutoring post-refinement routing and emotion | `Result_after_Prompt_refinments.xlsx` | Gold routing: `Expected_category`. Gold emotion: `final label` (98 evaluable rows after excluding `?`). CAS-RAG routing 72.0% (28/100 mistakes). CAS-RAG emotion 77.6% (22/98 mistakes). Use the archived columns. Do not re-run the prompts to recreate the table. |
| Tutoring pre-refinement emotion | Same workbook, pre-refinement emotion columns | Emotion denominator `n=97`. |
| Tutoring improvement after prompt refinement | Same workbook | Difference between the pre- and post-refinement emotion counts. |
| Bullying routing and emotion | `Results/comments_bullying_Models_Run_results.xlsx` | Gold routing: `Expected category` (100 rows). Gold emotion: `Emotion label` (88 rows; 12 blank). Strip a parenthetical suffix on gold before matching, for example `Frustration(Borderline Case)`. |
| Bullying binary retrieval decision | Same Bullying workbook | Context-dependent versus not, on 100 rows. |
| Tutoring versus Bullying synthesis | The two workbooks above | No separate file. |
| Pooled Layer-1 accuracy, macro-F1, per-topic rates, cost, and retrieval audit | Thesis Chapter 6 | Gold comments are in the six workbooks above. The prediction workbooks used to score those 600 comments are not in the public release. Thesis figures: pooled accuracy 69.5% (417/600); cost without a gate `600×3=1800`; cost with the gate `199×3+401×1=998` (44.6% relative reduction); wrong retrieval 51/199. |
| Chapter 6 figures fig1–fig8 | Submitted thesis | Drawn from the excluded prediction workbooks. Not shipped here. |
| Chapter 5 seven-metric walkthrough | `src/cas_rag/semantic/similarity.py` and `config/default.yaml` | Weights and margins (`m=0.12`, `b=0.08`) match the thesis. Walkthrough comments are in Chapter 5. |
| RQ1 offline consensus audit | `scripts/run_rq1_consensus_audit.py`, `src/cas_rag/evaluation/consensus.py` | The script is included. No consensus output workbook is archived here. Primary Tutoring and Bullying tables do not depend on that script. |

---

## In-depth workbooks that include model output

| Evaluation | File | What it contains |
|------------|------|------------------|
| Tutoring, post-refinement | `Result_after_Prompt_refinments.xlsx` | Comment text, gold routing, gold emotion, and Algorithms 1–4 plus CAS-RAG. Prompts for this video are under `resources/datasets/Experiments-data/video_02_education_tutoring_diner/prompts/`. |
| Bullying | `Results/comments_bullying_Models_Run_results.xlsx` | Comment text, gold routing, gold emotion, and the five-pipeline outputs used for the Bullying tables. The emotion columns are predominantly the OpenAI run. |

These two files are the source for the in-depth emotion tables. They are separate from the excluded six-dataset prediction workbooks.

---

## Running the code

From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,openai,gemini]"
python -m spacy download en_core_web_sm
cp .env.example .env   # set OPENAI_API_KEY; set GEMINI_API_KEY if running the dual-provider audit
```

Set `CAS_RAG_OFFLINE_MODE=0` for a thesis-style run so embeddings are the trained encoder. Leave `CAS_RAG_ALLOW_MOCK` unset or `0`. The mock provider is for tests.

A five-pipeline run on the Tutoring gold file:

```bash
python scripts/run_five_algorithm_comparison.py \
  --input resources/datasets/Experiments-data/comments_education_tutoring_diner.xlsx \
  --video-slug video_02_education_tutoring_diner \
  --llms openai gemini --kpi
```

Bullying uses `--video-slug video_01_bullying_bystander_bus_stop` and `resources/datasets/Experiments-data/comments_bullying.xlsx`.

A fresh run writes under `data/evaluation_outputs/`. It is a new execution, not a replay of the frozen tables.

### RQ1 audit

The audit reads a comparison workbook that already has both OpenAI and Gemini emotion columns. It does not change CAS-RAG routing.

```bash
python scripts/run_rq1_consensus_audit.py \
  --input data/evaluation_outputs/<dataset>_comparison_openai_gemini.xlsx \
  --output Results/rq1_consensus_<dataset>.xlsx \
  --dataset <tutoring|bullying> \
  --algorithms CAS-RAG Alg1 Alg2 Alg3 Alg4
```

The Bullying workbook already in `Results/` is mainly an OpenAI export. A dual-provider file has to be produced with `--llms openai gemini` before the audit can score both providers. That output is not part of the public release.

---

## What an examiner should treat as the result

1. Tutoring and Bullying numbers come from the two workbooks named above.
2. Pooled six-dataset routing and cost numbers are those printed in the thesis. The prediction files behind them are not in this public repository.
3. The RQ1 code is an offline audit. It is not an in-pipeline ensemble, and it does not alter the Tutoring or Bullying tables.
