# Six-Video Experiment Assets (CAS-RAG Evaluation)

Pooled validation set: **600 comments** (100 per video), aligned with thesis Chapter 6.

## Index

| Video | Folder slug | Evaluation label | YouTube ID | Labelled data (xlsx) |
|-------|-------------|------------------|------------|----------------------|
| 1 | `video_01_bullying_bystander_bus_stop` | Bullying | `Vo7kl9RsWBo` | `comments_bullying.xlsx` |
| 2 | `video_02_education_tutoring_diner` | Education | `lqevHrpvAx8` | `comments_education.xlsx` |
| 3 | `video_03_mental_health_suicide_bridge` | Suicide | `Gwo_O0SFwak` | `comments_Suicide.xlsx` |
| 4 | `video_04_teen_safety_predator_cafe` | TeenSafety | `YdqRLuYkYDk` | `comments-TeenSafety.xlsx` |
| 5 | `video_05_teen_pregnancy_gabby_experiment` | TeenPregnancy | `8ddhPrll65Y` | `comments_teenPregnancy.xlsx` |
| 6 | `video_06_drugs_prevention_bystander` | Teenagers-Drugs | `0yve1ezfeUU` | `comments_teenagers drugs.xlsx` |

See `experiments_manifest.json` for machine-readable paths.

## Per-video package (each folder)

| File | Purpose |
|------|---------|
| `video_metadata.json` | IDs, topic, scenario paths, label sets |
| `scenario_short.txt` | One-paragraph scenario for prompts/tables |
| `scenario_full.txt` | Full scenario embedded in all prompts |
| `prototypes_semantic.json` | 5× Explicit, 5× Ambiguous, 5× Context-dependent |
| `script_corpus.txt` | RAG corpus (transcript-derived chunks where available) |
| `prompts/P1_unknown_reference_detection.txt` | Algorithm 1/4 routing |
| `prompts/P2_rag_emotion_classification.txt` | RAG emotion (CAS-RAG context-dependent) |
| `prompts/P3_no_rag_emotion_classification.txt` | Algorithms 1–2 no-RAG |
| `prompts/P4_semantic_ambiguity_verification.txt` | Algorithms 1–2 Ambiguous branch |
| `prompts/P5_semantic_similarity_explicit_vague.txt` | Algorithms 3–4 two-way similarity (filename historical) |
| `prompts/P6_semantic_ambiguity_aware_emotion.txt` | Ambiguous branch + CAS-RAG |
| `prompts/P7_no_rag_emotion_without_vagueness_check.txt` | Explicit branch + CAS-RAG |

## Regenerate assets

```bash
python scripts/build_experiment_video_assets.py
```

Human verification is **required** in all pipelines (not optional).
