# Thesis Build Validation — Final Submission

**Date:** 2026-08-17  
**Build command:** `latexmk -pdf -interaction=nonstopmode -file-line-error main.tex`  
**Thesis source root:** `Thesis_Template_UTS_PHD/` (sibling to `CAS_RAG_Project`; not version-controlled in this repository)  
**Output PDF (repo copy):** `archive/thesis/Vahdati_PhD_Thesis_Final_2026-08-17.pdf`

---

## Build status: PASS

| Check | Result |
|-------|--------|
| PDF generated | Yes — 196 pages, ~1.94 MB |
| LaTeX errors | None |
| Undefined references | None |
| Undefined citations | None |
| Bibliography (biber/apa) | 88 entries in `main.bbl` |
| Unit tests (`test_consensus.py`) | 6 passed |

**Non-blocking warnings:** glossaries package (no glossary printed); typical overfull/underfull boxes.

---

## Content verification (Chapter 6 & 7)

| Section / artefact | Label | Present in build |
|--------------------|-------|------------------|
| Bullying evaluation | `sec:eval_bullying_100` (§6.5) | Yes |
| Cross-dataset synthesis | `sec:eval_cross_dataset` (§6.6) | Yes |
| RQ1 offline audit | `sec:eval_rq1` (§6.7) | Yes |
| Bullying emotion table | `tab:bullying_100_emotion` | Yes |
| Bullying binary table | `tab:bullying_100_binary` | Yes |
| Cross-dataset table | `tab:cross_dataset_comparison_framework` | Yes |
| Tutoring post-refine table | `tab:tutoring_100_post_refine` | Yes |
| Chapter 7 discussion | `ch07` | Yes — refs to bullying, cross-dataset, RQ1 resolve |

---

## Thesis sources modified (RQ1 remediation)

See `VALIDATION_REPORT.md` and `THESIS_REPRODUCIBILITY.md` for the full list of 18 updated `.tex` files plus new `pages/chapters/06/evaluation_rq1.tex`.

---

## Remaining high-risk items (documented, not build blockers)

1. RQ1 audit outputs (`Results/rq1_consensus_*.xlsx`) not archived — Ch6 §6.7 describes protocol only.  
2. Six-dataset gold (`VERY_IMPORTANT_RESULTS/`) external to repo — Layer-1/cost figures not fully reproducible from repo alone.  
3. `generate_figures.py` missing — Ch6 fig1–fig8 provenance external.  
4. Thesis `.tex` sources live outside this git repository; PDF in `archive/thesis/` is the submission artefact.
