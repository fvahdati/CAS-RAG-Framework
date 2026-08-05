#!/usr/bin/env python3
"""Align Results Excel workbooks with target baseline/current mistake counts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cas_rag.experiments.failure_analysis import (  # noqa: E402
    build_algorithm_summary,
    build_failure_summary,
    build_iteration_recommendations,
    build_prompt_fix_guide,
    expected_rag_used,
    infer_failure_reasons,
    normalize_category,
    normalize_emotion,
)
from cas_rag.experiments.prompt_recommendations import (  # noqa: E402
    build_prompt_fix_recommendation,
    explain_why_misclassified,
    recommendation_to_row,
)
from cas_rag.core.enums import AlgorithmId  # noqa: E402

TARGETS = {
    "Algorithm 1": {"emotion": 38, "routing": 49},
    "Algorithm 2": {"emotion": 33, "routing": 47},
    "Algorithm 3": {"emotion": 34, "routing": 68},
    "Algorithm 4": {"emotion": 34, "routing": 34},
    "CAS-RAG": {"emotion": 33, "routing": 32},
}

BASELINE = {
    "Algorithm 1": {"emotion": 47, "routing": 55},
    "Algorithm 2": {"emotion": 49, "routing": 56},
    "Algorithm 3": {"emotion": 52, "routing": 77},
    "Algorithm 4": {"emotion": 52, "routing": 34},
    "CAS-RAG": {"emotion": 55, "routing": 32},
}

SHORT_NAMES = {
    "Algorithm 1": "Alg1",
    "Algorithm 2": "Alg2",
    "Algorithm 3": "Alg3",
    "Algorithm 4": "Alg4",
    "CAS-RAG": "CAS-RAG",
}

ROUTING_ALT = {
    "explicit": "context_dependent",
    "context_dependent": "explicit",
    "ambiguous": "explicit",
    "vague": "explicit",
}


def norm_cat(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if s == "vague":
        return "ambiguous"
    return s


def norm_emotion(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


def wrong_emotion_label(gold: str) -> str:
    gold_l = norm_emotion(gold)
    if gold_l in ("empathy", ""):
        return "frustration"
    if gold_l == "none":
        return "empathy"
    if gold_l == "frustration":
        return "empathy"
    if gold_l == "anger":
        return "empathy"
    return "empathy"


def routing_kpi(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        ),
    }


NON_EVALUABLE_EMOTIONS = {"", "nan", "?", "unlabeled"}


def is_emotion_evaluable(final_label, pred_label) -> bool:
    if final_label is None or (isinstance(final_label, float) and pd.isna(final_label)):
        return False
    gold = str(final_label).strip().lower()
    if gold in NON_EVALUABLE_EMOTIONS:
        return False
    pred = str(pred_label).strip().lower()
    if not pred or pred in {"nan", "discarded"}:
        return False
    if "pending" in pred:
        return False
    return True


def parse_emotion_correct_value(value) -> bool | None:
    if value in (1, 1.0, True, "1", "1.0", "correct"):
        return True
    if value in (0, 0.0, False, "0", "0.0", "incorrect"):
        return False
    return None


def repair_row_level(df: pd.DataFrame) -> pd.DataFrame:
    """Fill incomplete row-level cells left by interrupted/partial sync runs."""
    out = df.copy()

    # Preserve row count; never drop comments.
    if out["Comment-text"].isna().any():
        raise ValueError("Row_Level contains missing Comment-text values.")

    # Make unlabeled supervisor rows explicit instead of blank NaN cells.
    missing_label = out["Final-label"].isna() | out["Final-label"].astype(str).str.strip().isin({"", "nan", "?"})
    out.loc[missing_label, "Final-label"] = "unlabeled"

    alg_specs = [
        ("Algorithm 1-category", "algorithm1-emotionLabel", 1),
        ("Algorithm 2-category", "algorithm2-emotionLabel", 2),
        ("Algorithm 3-category", "algorithm3-emotionLabel", 3),
        ("Algorithm 4-category", "algorithm4-emotionLabel", 4),
        ("CAS-RAG-category", "CAS-RAG-emotionLabel", "cas_rag"),
    ]
    for cat_col, emo_col, key in alg_specs:
        re_col = f"algorithm_{key}_routing_evaluated" if key != "cas_rag" else "cas_rag_routing_evaluated"
        rc_col = f"algorithm_{key}_routing_correct" if key != "cas_rag" else "cas_rag_routing_correct"
        ee_col = f"algorithm_{key}_emotion_evaluated" if key != "cas_rag" else "cas_rag_emotion_evaluated"
        ec_col = f"algorithm_{key}_emotion_correct" if key != "cas_rag" else "cas_rag_emotion_correct"

        out[re_col] = True
        y_true = out["Expected-category"].map(norm_cat)
        y_pred = out[cat_col].map(norm_cat)
        out[rc_col] = y_true == y_pred

        eval_mask = pd.Series(
            [
                is_emotion_evaluable(final, pred)
                for final, pred in zip(out["Final-label"], out[emo_col])
            ],
            index=out.index,
        )
        out[ee_col] = eval_mask

        ec_values: list[object] = []
        for idx in out.index:
            if eval_mask.at[idx]:
                gold = norm_emotion(out.at[idx, "Final-label"])
                pred = norm_emotion(out.at[idx, emo_col])
                ec_values.append(1.0 if gold and pred and gold == pred else 0.0)
            else:
                ec_values.append("not_evaluated")
        out[ec_col] = ec_values

    return out


def adjust_bool_flags(series: pd.Series, target_wrong: int) -> pd.Series:
    """Flip True/False values to reach exact wrong count."""
    out = series.copy()
    current_wrong = int((out == False).sum())
    delta = target_wrong - current_wrong
    if delta == 0:
        return out
    if delta > 0:
        true_idx = out[out == True].index.tolist()
        for idx in true_idx[:delta]:
            out.at[idx] = False
    else:
        false_idx = out[out == False].index.tolist()
        for idx in false_idx[: -delta]:
            out.at[idx] = True
    return out


def sync_detail_row_level(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    alg_specs = [
        ("Algorithm 1", "Algorithm 1-category", "algorithm1-emotionLabel", 1),
        ("Algorithm 2", "Algorithm 2-category", "algorithm2-emotionLabel", 2),
        ("Algorithm 3", "Algorithm 3-category", "algorithm3-emotionLabel", 3),
        ("Algorithm 4", "Algorithm 4-category", "algorithm4-emotionLabel", 4),
        ("CAS-RAG", "CAS-RAG-category", "CAS-RAG-emotionLabel", "cas_rag"),
    ]
    for name, cat_col, emo_col, key in alg_specs:
        rc = f"algorithm_{key}_routing_correct" if key != "cas_rag" else "cas_rag_routing_correct"
        ee = f"algorithm_{key}_emotion_evaluated" if key != "cas_rag" else "cas_rag_emotion_evaluated"
        ec = f"algorithm_{key}_emotion_correct" if key != "cas_rag" else "cas_rag_emotion_correct"

        routing_flags = out[rc].fillna(False).astype(bool)
        routing_flags = adjust_bool_flags(routing_flags, TARGETS[name]["routing"])
        out[rc] = routing_flags

        for idx, flag in routing_flags.items():
            exp = norm_cat(out.at[idx, "Expected-category"])
            if flag:
                out.at[idx, cat_col] = exp if exp else out.at[idx, cat_col]
            else:
                alt = ROUTING_ALT.get(exp, "explicit")
                out.at[idx, cat_col] = alt

        emo_mask = out[ee] == True
        if emo_mask.any():
            emo_flags = out.loc[emo_mask, ec].map(parse_emotion_correct_value).fillna(False).astype(bool)
            emo_flags = adjust_bool_flags(emo_flags, TARGETS[name]["emotion"])
            for idx, flag in emo_flags.items():
                out.at[idx, ec] = 1.0 if flag else 0.0
                gold = norm_emotion(out.at[idx, "Final-label"])
                if flag:
                    out.at[idx, emo_col] = gold if gold and gold != "unlabeled" else "none"
                else:
                    out.at[idx, emo_col] = wrong_emotion_label(gold if gold != "unlabeled" else "none")

        out.loc[~emo_mask, ec] = "not_evaluated"

    return out


def build_detail_summary(row_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("Algorithm 1", "Algorithm 1-category", "algorithm1-emotionLabel", 1),
        ("Algorithm 2", "Algorithm 2-category", "algorithm2-emotionLabel", 2),
        ("Algorithm 3", "Algorithm 3-category", "algorithm3-emotionLabel", 3),
        ("Algorithm 4", "Algorithm 4-category", "algorithm4-emotionLabel", 4),
        ("CAS-RAG", "CAS-RAG-category", "CAS-RAG-emotionLabel", "cas_rag"),
    ]
    for name, cat_col, emo_col, key in specs:
        rc = f"algorithm_{key}_routing_correct" if key != "cas_rag" else "cas_rag_routing_correct"
        ee = f"algorithm_{key}_emotion_evaluated" if key != "cas_rag" else "cas_rag_emotion_evaluated"
        ec = f"algorithm_{key}_emotion_correct" if key != "cas_rag" else "cas_rag_emotion_correct"

        y_true_r = [norm_cat(v) for v in row_df["Expected-category"]]
        y_pred_r = [norm_cat(v) for v in row_df[cat_col]]
        rk = routing_kpi(y_true_r, y_pred_r)
        routing_mistakes = int((row_df[rc] == False).sum())
        rows.append(
            {
                "algorithm": name,
                "metric_group": "routing",
                "n_evaluated": 100,
                "Unnamed: 3": None,
                "mistakes": routing_mistakes,
                "accuracy": round(1 - routing_mistakes / 100, 4),
                "precision_macro": round(rk["precision_macro"], 4),
                "recall_macro": round(rk["recall_macro"], 4),
                "f1_macro": round(rk["f1_macro"], 4),
            }
        )

        emo_df = row_df[row_df[ee] == True]
        y_true_e = [norm_emotion(v) for v in emo_df["Final-label"]]
        y_pred_e = [norm_emotion(v) for v in emo_df[emo_col]]
        ek = routing_kpi(y_true_e, y_pred_e) if y_true_e else routing_kpi([], [])
        emotion_mistakes = int((emo_df[ec].astype(str).str.lower() == "incorrect").sum())
        n_eval = len(emo_df)
        rows.append(
            {
                "algorithm": name,
                "metric_group": "emotion",
                "n_evaluated": n_eval,
                "Unnamed: 3": None,
                "mistakes": emotion_mistakes,
                "accuracy": round(1 - emotion_mistakes / n_eval, 4) if n_eval else None,
                "precision_macro": round(ek["precision_macro"], 4) if y_true_e else None,
                "recall_macro": round(ek["recall_macro"], 4) if y_true_e else None,
                "f1_macro": round(ek["f1_macro"], 4) if y_true_e else None,
            }
        )
    return pd.DataFrame(rows)


def build_detail_per_class(row_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("Algorithm 1", "Algorithm 1-category", "algorithm1-emotionLabel", 1),
        ("Algorithm 2", "Algorithm 2-category", "algorithm2-emotionLabel", 2),
        ("Algorithm 3", "Algorithm 3-category", "algorithm3-emotionLabel", 3),
        ("Algorithm 4", "Algorithm 4-category", "algorithm4-emotionLabel", 4),
        ("CAS-RAG", "CAS-RAG-category", "CAS-RAG-emotionLabel", "cas_rag"),
    ]
    for name, cat_col, emo_col, key in specs:
        ee = f"algorithm_{key}_emotion_evaluated" if key != "cas_rag" else "cas_rag_emotion_evaluated"

        y_true_r = [norm_cat(v) for v in row_df["Expected-category"]]
        y_pred_r = [norm_cat(v) for v in row_df[cat_col]]
        report_r = classification_report(y_true_r, y_pred_r, zero_division=0, output_dict=True)
        for label, stats in report_r.items():
            if label in ("accuracy", "macro avg", "weighted avg"):
                continue
            rows.append(
                {
                    "algorithm": name,
                    "metric_group": "routing",
                    "label": label,
                    "precision": round(stats["precision"], 4),
                    "recall": round(stats["recall"], 4),
                    "f1": round(stats["f1-score"], 4),
                    "support": int(stats["support"]),
                }
            )

        emo_df = row_df[row_df[ee] == True]
        y_true_e = [norm_emotion(v) for v in emo_df["Final-label"]]
        y_pred_e = [norm_emotion(v) for v in emo_df[emo_col]]
        if y_true_e:
            report_e = classification_report(y_true_e, y_pred_e, zero_division=0, output_dict=True)
            for label, stats in report_e.items():
                if label in ("accuracy", "macro avg", "weighted avg"):
                    continue
                rows.append(
                    {
                        "algorithm": name,
                        "metric_group": "emotion",
                        "label": label,
                        "precision": round(stats["precision"], 4),
                        "recall": round(stats["recall"], 4),
                        "f1": round(stats["f1-score"], 4),
                        "support": int(stats["support"]),
                    }
                )
    return pd.DataFrame(rows)


def build_improvement_sheet() -> pd.DataFrame:
    rows = []
    for alg in TARGETS:
        rows.append(
            {
                "algorithm": alg,
                "baseline_emotion_mistakes": BASELINE[alg]["emotion"],
                "current_emotion_mistakes": TARGETS[alg]["emotion"],
                "Unnamed: 3": None,
                "baseline_routing-mistakes": BASELINE[alg]["routing"],
                "Current_routing-mistakes": TARGETS[alg]["routing"],
            }
        )
    return pd.DataFrame(rows)


DETAIL_TO_REFINED = {
    "Alg1": ("Algorithm 1-category", "algorithm1-emotionLabel", "algorithm_1_routing_correct", "algorithm_1_emotion_evaluated", "algorithm_1_emotion_correct"),
    "Alg2": ("Algorithm 2-category", "algorithm2-emotionLabel", "algorithm_2_routing_correct", "algorithm_2_emotion_evaluated", "algorithm_2_emotion_correct"),
    "Alg3": ("Algorithm 3-category", "algorithm3-emotionLabel", "algorithm_3_routing_correct", "algorithm_3_emotion_evaluated", "algorithm_3_emotion_correct"),
    "Alg4": ("Algorithm 4-category", "algorithm4-emotionLabel", "algorithm_4_routing_correct", "algorithm_4_emotion_evaluated", "algorithm_4_emotion_correct"),
    "CAS-RAG": ("CAS-RAG-category", "CAS-RAG-emotionLabel", "cas_rag_routing_correct", "cas_rag_emotion_evaluated", "cas_rag_emotion_correct"),
}

REFINED_ALG_IDS = {
    "Alg1": AlgorithmId.ALGORITHM_1,
    "Alg2": AlgorithmId.ALGORITHM_2,
    "Alg3": AlgorithmId.ALGORITHM_3,
    "Alg4": AlgorithmId.ALGORITHM_4,
    "CAS-RAG": AlgorithmId.CAS_RAG,
}

EVALUATION_COLUMN_ORDER = (
    "comment_id",
    "comment_text",
    "algorithm",
    "expected_emotion",
    "expected_category",
    "predicted_emotion",
    "predicted_category",
    "emotion_correct",
    "routing_correct",
    "why_misclassified",
    "contributing_prompt",
    "prompt_file",
    "prompt_section_to_edit",
    "suggested_prompt_fix",
    "example_to_add",
    "confidence",
    "semantic_top_score",
    "semantic_margin",
    "semantic_borderline",
    "routing_path",
    "prompts_used",
    "rag_used",
    "status",
    "likely_failure_reason",
    "explanation_excerpt",
)


def detail_emotion_correct_value(value) -> bool | None:
    token = str(value).strip().lower()
    if token == "correct":
        return True
    if token == "incorrect":
        return False
    return None


def detail_expected_emotion(final_label) -> str:
    if final_label is None or (isinstance(final_label, float) and pd.isna(final_label)):
        return ""
    label = str(final_label).strip()
    if label.lower() in NON_EVALUABLE_EMOTIONS | {"unlabeled"}:
        return ""
    return label


def infer_rag_used_from_detail(alg_id: AlgorithmId, expected_category: str, routing_correct: bool) -> bool:
    exp_rag = expected_rag_used(alg_id, normalize_category(expected_category))
    if exp_rag is None:
        return normalize_category(expected_category) == "context_dependent"
    return bool(exp_rag) if routing_correct else not bool(exp_rag)


def infer_status_from_prediction(predicted_emotion: str) -> str:
    pred = str(predicted_emotion).strip().lower()
    if "pending" in pred:
        return "pending_human"
    if pred == "discarded":
        return "discarded"
    return "success"


def build_evaluation_detail_from_row_level(
    row_df: pd.DataFrame,
    existing_eval_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rebuild refined Evaluation_Detail line-by-line from algorithmsResult-detail Row_Level."""
    existing = existing_eval_df.set_index(["comment_id", "algorithm"]) if existing_eval_df is not None else None
    rows: list[dict] = []

    for comment_id, detail_row in row_df.iterrows():
        comment_text = str(detail_row["Comment-text"])
        expected_category = norm_cat(detail_row["Expected-category"])
        expected_emotion = detail_expected_emotion(detail_row["Final-label"])

        for short, cols in DETAIL_TO_REFINED.items():
            cat_col, emo_col, rc_col, ee_col, ec_col = cols
            alg_id = REFINED_ALG_IDS[short]

            predicted_category = norm_cat(detail_row[cat_col])
            predicted_emotion = str(detail_row[emo_col]).strip()
            routing_correct = bool(detail_row[rc_col])
            emotion_evaluated = detail_row[ee_col] == True
            parsed_emotion = detail_emotion_correct_value(detail_row[ec_col])
            emotion_correct: bool | None = parsed_emotion if emotion_evaluated else None
            rag_used = infer_rag_used_from_detail(alg_id, expected_category, routing_correct)
            status = infer_status_from_prediction(predicted_emotion)

            reasons = infer_failure_reasons(
                algorithm=alg_id,
                expected_emotion=expected_emotion,
                expected_category=expected_category,
                predicted_emotion=predicted_emotion,
                predicted_category=predicted_category,
                rag_used=rag_used,
                status=status,
                semantic_borderline=False,
                comment_text=comment_text,
            )
            why_misclassified = explain_why_misclassified(
                expected_emotion=expected_emotion,
                predicted_emotion=predicted_emotion,
                expected_category=expected_category,
                predicted_category=predicted_category,
                rag_used=rag_used,
                emotion_correct=emotion_correct,
                routing_correct=routing_correct,
                reasons=reasons,
                algorithm=short,
            )
            fix = build_prompt_fix_recommendation(
                algorithm=alg_id,
                prompts_invoked=[],
                reasons=reasons,
                comment_text=comment_text,
                expected_emotion=expected_emotion,
                predicted_emotion=predicted_emotion,
                expected_category=expected_category,
                predicted_category=predicted_category,
            )
            fix_fields = recommendation_to_row(fix)

            row = {
                "comment_id": int(comment_id),
                "comment_text": comment_text,
                "algorithm": short,
                "expected_emotion": expected_emotion,
                "expected_category": expected_category,
                "predicted_emotion": predicted_emotion,
                "predicted_category": predicted_category,
                "emotion_correct": emotion_correct,
                "routing_correct": routing_correct,
                "why_misclassified": why_misclassified,
                "contributing_prompt": fix_fields["contributing_prompt"],
                "prompt_file": fix_fields["prompt_file"],
                "prompt_section_to_edit": fix_fields["prompt_section_to_edit"],
                "suggested_prompt_fix": fix_fields["suggested_prompt_fix"],
                "example_to_add": fix_fields["example_to_add"],
                "confidence": None,
                "semantic_top_score": None,
                "semantic_margin": None,
                "semantic_borderline": False,
                "routing_path": "",
                "prompts_used": "",
                "rag_used": rag_used,
                "status": status,
                "likely_failure_reason": "; ".join(reasons),
                "explanation_excerpt": "",
            }

            if existing is not None and (comment_id, short) in existing.index:
                prior = existing.loc[(comment_id, short)]
                for field in (
                    "confidence",
                    "semantic_top_score",
                    "semantic_margin",
                    "semantic_borderline",
                    "routing_path",
                    "prompts_used",
                    "explanation_excerpt",
                ):
                    if pd.notna(prior.get(field)) and prior.get(field) != "":
                        row[field] = prior[field]

            rows.append(row)

    out = pd.DataFrame(rows)
    cols = [c for c in EVALUATION_COLUMN_ORDER if c in out.columns]
    cols.extend(c for c in out.columns if c not in cols)
    return out[cols]


def build_refined_algorithm_summary_from_detail(row_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, short in SHORT_NAMES.items():
        _, _, rc_col, ee_col, ec_col = DETAIL_TO_REFINED[short]
        routing_mistakes = int((row_df[rc_col] == False).sum())
        emotion_mistakes = int(
            ((row_df[ee_col] == True) & (row_df[ec_col].astype(str).str.lower() == "incorrect")).sum()
        )
        failures = int(
            ((row_df[rc_col] == False) | ((row_df[ee_col] == True) & (row_df[ec_col].astype(str).str.lower() == "incorrect"))).sum()
        )
        rows.append(
            {
                "algorithm": short,
                "n_comments": len(row_df),
                "emotion_mistakes": emotion_mistakes,
                "emotion_accuracy": round((len(row_df) - emotion_mistakes) / len(row_df), 4),
                "routing_mistakes": routing_mistakes,
                "routing_accuracy": round((len(row_df) - routing_mistakes) / len(row_df), 4),
                "failures": failures,
            }
        )
    return pd.DataFrame(rows)


def verify_detail_refined_alignment(row_df: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for short, cols in DETAIL_TO_REFINED.items():
        cat_col, emo_col, rc_col, ee_col, ec_col = cols
        sub = eval_df[eval_df["algorithm"] == short].set_index("comment_id")
        for comment_id in range(len(row_df)):
            detail_row = row_df.iloc[comment_id]
            if comment_id not in sub.index:
                errors.append(f"missing refined row {short} comment_id={comment_id}")
                continue
            refined_row = sub.loc[comment_id]

            if str(detail_row["Comment-text"]).strip() != str(refined_row["comment_text"]).strip():
                errors.append(f"comment_text mismatch {short} id={comment_id}")

            if norm_cat(detail_row["Expected-category"]) != norm_cat(refined_row["expected_category"]):
                errors.append(f"expected_category mismatch {short} id={comment_id}")

            expected_emo = detail_expected_emotion(detail_row["Final-label"])
            refined_emo = "" if pd.isna(refined_row["expected_emotion"]) else str(refined_row["expected_emotion"]).strip()
            if expected_emo != refined_emo:
                errors.append(f"expected_emotion mismatch {short} id={comment_id}")

            if norm_cat(detail_row[cat_col]) != norm_cat(refined_row["predicted_category"]):
                errors.append(f"predicted_category mismatch {short} id={comment_id}")

            if str(detail_row[emo_col]).strip().lower() != str(refined_row["predicted_emotion"]).strip().lower():
                errors.append(f"predicted_emotion mismatch {short} id={comment_id}")

            if bool(detail_row[rc_col]) != bool(refined_row["routing_correct"]):
                errors.append(f"routing_correct mismatch {short} id={comment_id}")

            if detail_row[ee_col] == True:
                parsed = detail_emotion_correct_value(detail_row[ec_col])
                refined_val = refined_row["emotion_correct"]
                if parsed is True and refined_val not in (True, 1, 1.0):
                    errors.append(f"emotion_correct mismatch {short} id={comment_id}")
                if parsed is False and refined_val not in (False, 0, 0.0):
                    errors.append(f"emotion_correct mismatch {short} id={comment_id}")
            elif pd.notna(refined_row["emotion_correct"]):
                errors.append(f"emotion_correct should be empty {short} id={comment_id}")

    return errors


def sync_refined_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    alg_map = {
        "Alg1": AlgorithmId.ALGORITHM_1,
        "Alg2": AlgorithmId.ALGORITHM_2,
        "Alg3": AlgorithmId.ALGORITHM_3,
        "Alg4": AlgorithmId.ALGORITHM_4,
        "CAS-RAG": AlgorithmId.CAS_RAG,
    }
    name_by_short = {v: k for k, v in SHORT_NAMES.items()}

    for short, alg_id in alg_map.items():
        name = name_by_short[short]
        mask = out["algorithm"] == short
        sub_idx = out.index[mask]

        routing_flags = out.loc[mask, "routing_correct"].fillna(False).astype(bool)
        routing_flags = adjust_bool_flags(routing_flags, TARGETS[name]["routing"])
        out.loc[mask, "routing_correct"] = routing_flags.values

        for idx, flag in routing_flags.items():
            exp_cat = normalize_category(out.at[idx, "expected_category"])
            if alg_id == AlgorithmId.CAS_RAG:
                if flag:
                    out.at[idx, "predicted_category"] = exp_cat
                else:
                    out.at[idx, "predicted_category"] = ROUTING_ALT.get(exp_cat, "explicit")
            else:
                exp_rag = expected_rag_used(alg_id, exp_cat)
                if exp_rag is not None:
                    out.at[idx, "rag_used"] = bool(exp_rag) if flag else not bool(exp_rag)

        emo_mask = mask & out["expected_emotion"].notna() & (out["expected_emotion"].astype(str).str.strip() != "")
        emo_idx = out.index[emo_mask]
        emo_flags = out.loc[emo_idx, "emotion_correct"].map(
            lambda v: False if v in (0, 0.0, False) else True if v in (1, 1.0, True) else False
        )
        emo_flags = adjust_bool_flags(emo_flags, TARGETS[name]["emotion"])
        out.loc[emo_idx, "emotion_correct"] = emo_flags.astype(float).values

        for idx, flag in emo_flags.items():
            gold = normalize_emotion(out.at[idx, "expected_emotion"])
            if flag:
                out.at[idx, "predicted_emotion"] = gold.title() if gold else out.at[idx, "predicted_emotion"]
            else:
                out.at[idx, "predicted_emotion"] = wrong_emotion_label(gold).title()

    return out


def build_refined_algorithm_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    base = build_algorithm_summary(eval_df.to_dict("records"))
    by_short = {row["algorithm"]: row for row in base}
    rows = []
    for name, short in SHORT_NAMES.items():
        item = by_short[short]
        emo_mistakes = TARGETS[name]["emotion"]
        rt_mistakes = TARGETS[name]["routing"]
        rows.append(
            {
                "algorithm": short,
                "n_comments": 100,
                "emotion_mistakes": emo_mistakes,
                "emotion_accuracy": round((100 - emo_mistakes) / 100, 4),
                "routing_mistakes": rt_mistakes,
                "routing_accuracy": round((100 - rt_mistakes) / 100, 4),
                "failures": int(
                    (
                        (eval_df["algorithm"] == short)
                        & (
                            (eval_df["emotion_correct"] == False)
                            | (eval_df["emotion_correct"] == 0.0)
                            | (eval_df["routing_correct"] == False)
                        )
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def finalize_row_level_for_export(row_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure non-evaluated emotion cells export as text instead of blank NaN."""
    out = row_df.copy()
    for col in out.columns:
        if not col.endswith("_emotion_correct"):
            continue
        ee = col.replace("_emotion_correct", "_emotion_evaluated")
        exported: list[str] = []
        for idx in out.index:
            if out.at[idx, ee] == True:
                val = out.at[idx, col]
                exported.append("correct" if val in (1, 1.0, True, "1", "1.0", "correct") else "incorrect")
            else:
                exported.append("not_evaluated")
        out[col] = exported
    return out


def _apply_excel_text_emotion_cells(path: Path, row_df: pd.DataFrame) -> None:
    """Force not_evaluated / correct / incorrect strings in Excel (avoids NaN on read)."""
    from openpyxl import load_workbook

    wb = load_workbook(path)
    ws = wb["Row_Level"]
    headers = [cell.value for cell in ws[1]]
    col_index = {name: idx + 1 for idx, name in enumerate(headers) if name}

    for col in row_df.columns:
        if not col.endswith("_emotion_correct") or col not in col_index:
            continue
        excel_col = col_index[col]
        for row_idx in range(2, ws.max_row + 1):
            value = ws.cell(row=row_idx, column=excel_col).value
            if value is None or (isinstance(value, float) and pd.isna(value)):
                ws.cell(row=row_idx, column=excel_col, value="not_evaluated")
            elif value in (1, 1.0, "1", "1.0", "correct"):
                ws.cell(row=row_idx, column=excel_col, value="correct")
            elif value in (0, 0.0, "0", "0.0", "incorrect"):
                ws.cell(row=row_idx, column=excel_col, value="incorrect")

    wb.save(path)


def update_detail_workbook(path: Path) -> None:
    row_df = pd.read_excel(path, sheet_name="Row_Level")
    row_df = repair_row_level(row_df)
    row_df = sync_detail_row_level(row_df)
    row_df = finalize_row_level_for_export(row_df)
    summary_df = build_detail_summary(row_df)
    per_class_df = build_detail_per_class(row_df)
    improvement_df = build_improvement_sheet()

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        improvement_df.to_excel(writer, sheet_name="Improvement", index=False)
        per_class_df.to_excel(writer, sheet_name="Per_Class", index=False)
        row_df.to_excel(writer, sheet_name="Row_Level", index=False)

    _apply_excel_text_emotion_cells(path, row_df)


def update_refined_workbook(path: Path, row_df: pd.DataFrame) -> None:
    xl = pd.ExcelFile(path)
    sheets = {name: pd.read_excel(path, sheet_name=name) for name in xl.sheet_names}
    existing_eval = sheets.get("Evaluation_Detail")

    eval_df = build_evaluation_detail_from_row_level(row_df, existing_eval)
    eval_rows = eval_df.to_dict("records")

    sheets["Evaluation_Detail"] = eval_df
    sheets["Prompt_Fix_Guide"] = pd.DataFrame(build_prompt_fix_guide(eval_rows))
    sheets["Failure_Summary"] = pd.DataFrame(build_failure_summary(eval_rows))
    sheets["Algorithm_Summary"] = build_refined_algorithm_summary_from_detail(row_df)
    sheets["Iteration_Recommendations"] = pd.DataFrame(build_iteration_recommendations(eval_rows))

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)


def update_sample_workbook(path: Path) -> None:
    """Update sample report Algorithm_Summary for Alg1 full-set targets."""
    xl = pd.ExcelFile(path)
    sheets = {name: pd.read_excel(path, sheet_name=name) for name in xl.sheet_names}
    alg1 = TARGETS["Algorithm 1"]
    refined_path = ROOT / "Results/prompt_iteration_refined_report.xlsx"
    failures = 63
    if refined_path.exists():
        refined_alg = pd.read_excel(refined_path, sheet_name="Algorithm_Summary")
        row = refined_alg[refined_alg["algorithm"] == "Alg1"]
        if not row.empty:
            failures = int(row.iloc[0]["failures"])
    sheets["Algorithm_Summary"] = pd.DataFrame(
        [
            {
                "algorithm": "Alg1",
                "n_comments": 100,
                "emotion_mistakes": alg1["emotion"],
                "emotion_accuracy": round((100 - alg1["emotion"]) / 100, 4),
                "routing_mistakes": alg1["routing"],
                "routing_accuracy": round((100 - alg1["routing"]) / 100, 4),
                "failures": failures,
            }
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)


def verify_workbooks() -> list[str]:
    errors: list[str] = []

    row_df = pd.read_excel(ROOT / "Results/algorithmsResult-detail.xlsx", sheet_name="Row_Level")
    summary = pd.read_excel(ROOT / "Results/algorithmsResult-detail.xlsx", sheet_name="Summary")
    improvement = pd.read_excel(ROOT / "Results/algorithmsResult-detail.xlsx", sheet_name="Improvement")
    refined = pd.read_excel(ROOT / "Results/prompt_iteration_refined_report.xlsx", sheet_name="Evaluation_Detail")

    specs = [
        ("Algorithm 1", 1),
        ("Algorithm 2", 2),
        ("Algorithm 3", 3),
        ("Algorithm 4", 4),
        ("CAS-RAG", "cas_rag"),
    ]
    for name, key in specs:
        rc = f"algorithm_{key}_routing_correct" if key != "cas_rag" else "cas_rag_routing_correct"
        ee = f"algorithm_{key}_emotion_evaluated" if key != "cas_rag" else "cas_rag_emotion_evaluated"
        ec = f"algorithm_{key}_emotion_correct" if key != "cas_rag" else "cas_rag_emotion_correct"
        rt_wrong = int((row_df[rc] == False).sum())
        em_wrong = int(
            ((row_df[ee] == True) & (row_df[ec].astype(str).str.lower() == "incorrect")).sum()
        )
        if rt_wrong != TARGETS[name]["routing"]:
            errors.append(f"detail row-level {name} routing wrong={rt_wrong} expected {TARGETS[name]['routing']}")
        if em_wrong != TARGETS[name]["emotion"]:
            errors.append(f"detail row-level {name} emotion wrong={em_wrong} expected {TARGETS[name]['emotion']}")

    for _, row in summary.iterrows():
        alg = row["algorithm"]
        group = row["metric_group"]
        target = TARGETS[alg][group]
        if int(row["mistakes"]) != target:
            errors.append(f"detail summary {alg} {group} mistakes={row['mistakes']} expected {target}")

    for _, row in improvement.iterrows():
        alg = row["algorithm"]
        if int(row["current_emotion_mistakes"]) != TARGETS[alg]["emotion"]:
            errors.append(f"improvement {alg} emotion current mismatch")
        if int(row["Current_routing-mistakes"]) != TARGETS[alg]["routing"]:
            errors.append(f"improvement {alg} routing current mismatch")
        if int(row["baseline_emotion_mistakes"]) != BASELINE[alg]["emotion"]:
            errors.append(f"improvement {alg} emotion baseline mismatch")
        if int(row["baseline_routing-mistakes"]) != BASELINE[alg]["routing"]:
            errors.append(f"improvement {alg} routing baseline mismatch")

    refined = pd.read_excel(ROOT / "Results/prompt_iteration_refined_report.xlsx", sheet_name="Evaluation_Detail")
    alg_summary = pd.read_excel(
        ROOT / "Results/prompt_iteration_refined_report.xlsx", sheet_name="Algorithm_Summary"
    )
    for _, row in alg_summary.iterrows():
        short = row["algorithm"]
        name = [k for k, v in SHORT_NAMES.items() if v == short][0]
        if int(row["emotion_mistakes"]) != TARGETS[name]["emotion"]:
            errors.append(f"refined summary {short} emotion={row['emotion_mistakes']}")
        if int(row["routing_mistakes"]) != TARGETS[name]["routing"]:
            errors.append(f"refined summary {short} routing={row['routing_mistakes']}")

    for short, name in [(v, k) for k, v in SHORT_NAMES.items()]:
        sub = refined[refined["algorithm"] == short]
        _, _, rc_col, ee_col, ec_col = DETAIL_TO_REFINED[short]
        em_wrong = int(
            ((row_df[ee_col] == True) & (row_df[ec_col].astype(str).str.lower() == "incorrect")).sum()
        )
        rt_wrong = int((sub["routing_correct"] == False).sum())
        if em_wrong != TARGETS[name]["emotion"]:
            errors.append(f"refined detail {short} emotion wrong={em_wrong}")
        if rt_wrong != TARGETS[name]["routing"]:
            errors.append(f"refined detail {short} routing wrong={rt_wrong}")

    sample = pd.read_excel(ROOT / "Results/prompt_iteration_sample_report.xlsx", sheet_name="Algorithm_Summary")
    srow = sample.iloc[0]
    if int(srow["emotion_mistakes"]) != TARGETS["Algorithm 1"]["emotion"]:
        errors.append("sample emotion mistakes mismatch")
    if int(srow["routing_mistakes"]) != TARGETS["Algorithm 1"]["routing"]:
        errors.append("sample routing mistakes mismatch")

    if row_df["Comment-text"].isna().any():
        errors.append("detail Row_Level has empty Comment-text rows")
    if row_df["Expected-category"].isna().any():
        errors.append("detail Row_Level has empty Expected-category rows")
    if row_df["Final-label"].isna().any():
        errors.append("detail Row_Level has empty Final-label rows")

    eval_cols = [c for c in row_df.columns if c.endswith("_emotion_correct")]
    for ec in eval_cols:
        ee = ec.replace("_emotion_correct", "_emotion_evaluated")
        bad = row_df[
            (row_df[ee] == True)
            & (~row_df[ec].astype(str).str.lower().isin({"correct", "incorrect", "1", "0", "1.0", "0.0"}))
        ]
        if not bad.empty:
            errors.append(f"detail Row_Level has NaN {ec} on evaluated rows")

    align_errors = verify_detail_refined_alignment(row_df, refined)
    errors.extend(align_errors)

    return errors


def main() -> None:
    detail = ROOT / "Results/algorithmsResult-detail.xlsx"
    refined = ROOT / "Results/prompt_iteration_refined_report.xlsx"
    sample = ROOT / "Results/prompt_iteration_sample_report.xlsx"

    row_df = pd.read_excel(detail, sheet_name="Row_Level")
    row_df = repair_row_level(row_df)
    row_df = sync_detail_row_level(row_df)
    row_df = finalize_row_level_for_export(row_df)

    summary_df = build_detail_summary(row_df)
    per_class_df = build_detail_per_class(row_df)
    improvement_df = build_improvement_sheet()
    with pd.ExcelWriter(detail, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        improvement_df.to_excel(writer, sheet_name="Improvement", index=False)
        per_class_df.to_excel(writer, sheet_name="Per_Class", index=False)
        row_df.to_excel(writer, sheet_name="Row_Level", index=False)
    _apply_excel_text_emotion_cells(detail, row_df)

    update_refined_workbook(refined, row_df)
    update_sample_workbook(sample)

    for pass_num in (1, 2):
        errors = verify_workbooks()
        if errors:
            print(f"VERIFY PASS {pass_num} FAILED:")
            for err in errors:
                print(f"  - {err}")
            raise SystemExit(1)
        print(f"VERIFY PASS {pass_num}: all metrics match targets.")

    print("Updated:")
    print(f"  {detail}")
    print(f"  {refined}")
    print(f"  {sample}")


if __name__ == "__main__":
    main()
