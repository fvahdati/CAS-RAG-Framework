# Final Validation Report — RQ1 Alignment & Reproducibility Package

**Date:** 2026-08-17  
**Scope:** Part A (thesis RQ1/Model 1 alignment), Part B (consensus audit code), Part C (reproducibility package). No CAS-RAG architecture changes; no modification of reported Ch6 numbers.

---

## 1. RQ1 issues resolved

| Audit finding | Resolution |
|---------------|------------|
| No consensus / disagreement / reliability code | **Resolved:** `src/cas_rag/evaluation/consensus.py`, `src/cas_rag/experiments/consensus_audit.py` |
| No RQ1 evaluation artefact | **Resolved:** `scripts/run_rq1_consensus_audit.py` → `Consensus_Audit` + `RQ1_Summary` sheets |
| Thesis over-claims full Model 1 implementation | **Resolved:** Ch1, Ch2, Ch3, Ch5, Ch6, Ch7, abstract, appendix reworded to offline audit scope |
| Model 1 naming collision (Ch6 vs Ch3) | **Resolved:** Ch6 “linguistic-first **prototype**” (not Model 1) |
| Ch7 future work contradicts RQ1 audit | **Resolved:** future work now targets weighted / in-pipeline extensions |
| No reproducibility traceability | **Resolved:** `THESIS_REPRODUCIBILITY.md` + updated Appendix A |

---

## 2. Thesis files modified

Path prefix: `Thesis_Template_UTS_PHD/`

| File | Change |
|------|--------|
| `pages/_preliminary/abstract.tex` | RQ1 offline audit mention |
| `pages/chapters/01/sections/section1_3.tex` | RQ1, Objective 1, component list |
| `pages/chapters/01/sections/section1_4.tex` | Implementation + hallucination proxy wording |
| `pages/chapters/01/sections/section1_5.tex` | Theoretical contribution (offline consensus) |
| `pages/chapters/02/sections/section2_1.tex` | Research Aim 1 |
| `pages/chapters/02/sections/section2_5.tex` | Aim 1 + transition |
| `pages/chapters/03/sections/section3_3.tex` | Component 1 scope |
| `pages/chapters/03/sections/section3_4.tex` | Model 1 implementation-status tags |
| `pages/chapters/03/sections/section3_7.tex` | Integration = conceptual/future |
| `pages/chapters/03/sections/section3_8.tex` | Evaluation dimension wording |
| `pages/chapters/03/sections/section3_9.tex` | Chapter roadmap includes §RQ1 |
| `pages/chapters/05/casrag_implementation.tex` | RQ1 audit subsection |
| `pages/chapters/06/index.tex` | `\input{evaluation_rq1}` |
| `pages/chapters/06/evaluation_results.tex` | Model 1 naming fix |
| `pages/chapters/06/evaluation_rq1.tex` | **New** RQ1 section |
| `pages/chapters/07/index.tex` | Contribution + future work |
| `pages/chapters/07/limitations_expanded.tex` | RQ1 evaluated via audit |
| `pages/appendices/a/index.tex` | Workbook paths + RQ1 steps |

**Note:** Thesis directory is not a git repository in this workspace; changes are local only.

---

## 3. Code files added / modified

| File | Action |
|------|--------|
| `src/cas_rag/evaluation/consensus.py` | **Added** |
| `src/cas_rag/experiments/consensus_audit.py` | **Added** |
| `scripts/run_rq1_consensus_audit.py` | **Added** |
| `tests/unit/test_consensus.py` | **Added** |
| `THESIS_REPRODUCIBILITY.md` | **Added** |
| `VALIDATION_REPORT.md` | **Added** |
| `README.md` | **Updated** (RQ1 section) |

**Not changed:** `PipelineEngine`, `cas_rag.py`, routing, reported evaluation numbers.

---

## 4. Reproducibility gaps resolved vs remaining

### Resolved

- Traceability matrix for Ch6 tables and figures documented
- Canonical workbook paths for Tutoring and Bullying in repo
- RQ1 reproduction steps in README, Appendix A, `THESIS_REPRODUCIBILITY.md`
- Submission checklist included

### Remaining (require your decision / bundling)

| Gap | Action needed |
|-----|----------------|
| `VERY_IMPORTANT_RESULTS/*.xlsx` (six-dataset gold) | Bundle with submission or copy into repo `Results/gold/` |
| `generate_figures.py` | Add to repo or ship figure PDFs only with provenance note |
| Tutoring **dual-provider** xlsx for full RQ1 | Run `run_five_algorithm_comparison.py --llms openai gemini` once APIs available |
| Bullying workbook currently single-provider dominant | Same dual-provider regeneration for full RQ1_Summary |
| Thesis not in git | Commit thesis separately if version-controlled elsewhere |

---

## 5. Remaining examiner risks

| Risk | Severity | Mitigation status |
|------|----------|-------------------|
| RQ1 “which mechanism is most effective” (plural) | Medium | Thesis now specifies majority vote + agreement scoring only; comparison of multiple mechanisms = future work |
| GPT-4 vs gpt-4o-mini naming | Low | Ch3 §3.4 notes production model; acceptable if consistent |
| RQ1 audit not yet run on dual-provider Tutoring/Bullying | Medium | **Your action:** run comparison + audit; archive `Results/rq1_consensus_*.xlsx` |
| Six-dataset figures without gold bundle | Medium | Bundle gold or examiner cannot recompute 69.5% |
| Consensus does not improve accuracy unless data shows it | Low | Report RQ1_Summary without claiming improvement unless metrics support it |

---

## 6. Decisions still required from you

1. **Bundle `VERY_IMPORTANT_RESULTS/`** into submission package?  
2. **Regenerate dual-provider comparisons** (API cost) for Tutoring + Bullying before submission?  
3. **Add thesis to git** or export panel PDF with these tex changes?  
4. **Remove duplicate** `comments_bullying_Models_Run_results.xlsx` at repo root (keep `Results/` copy only)?

---

## 7. Tests

`tests/unit/test_consensus.py` — 6 unit tests (majority, split, None label class, parenthetical normalisation, non-votable exclusion, gold match).

Run: `PYTHONPATH=src pytest tests/unit/test_consensus.py`

---

## 8. Claim classification summary (post-alignment)

| Component | Status |
|-----------|--------|
| Multi-provider comparison export | **B — Partial** (existing runner) |
| Majority consensus + agreement | **A — Implemented offline** |
| Disagreement + reliability tiers | **A — Implemented offline** |
| Hallucination proxy | **B — Partial** |
| Weighted / confidence consensus | **D — Future** |
| In-pipeline filtering | **D — Future** |
| Model 1 → CAS-RAG integration | **D — Conceptual** |
| CAS-RAG primary evaluation | **A — Unchanged** |
