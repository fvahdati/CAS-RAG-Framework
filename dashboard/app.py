"""Streamlit dashboard — CAS-RAG analytics & algorithm comparison."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from cas_rag.config.settings import get_settings
from cas_rag.core.enums import AlgorithmId
from cas_rag.core.models import CommentInput
from cas_rag.datasets.loader import DatasetLoader
from cas_rag.analytics.dashboard_data import AnalyticsEngine
from cas_rag.orchestration.engine import PipelineEngine
from cas_rag.subsystems.human_verification import HumanVerificationService

st.set_page_config(page_title="CAS-RAG Dashboard", layout="wide")
st.title("CAS-RAG Research Dashboard")
st.caption("Context-Aware Semantic Retrieval-Augmented Generation")

settings = get_settings()
tab1, tab2, tab3, tab4 = st.tabs(["Process", "Compare Algorithms", "Human Verification", "Analytics"])

with tab1:
    alg = st.selectbox("Algorithm", [a.value for a in AlgorithmId])
    comment = st.text_area("Comment", "When Harper walked over I melted-pure emotional support.")
    if st.button("Run pipeline"):
        engine = PipelineEngine(alg)
        r = engine.process(CommentInput(comment_id="dash-1", comment=comment))
        st.json(json.loads(r.model_dump_json()))

with tab2:
    comment = st.text_input("Comment for comparison", "Wow")
    if st.button("Compare all algorithms"):
        engine = PipelineEngine("cas_rag")
        rows = []
        for alg, res in engine.compare_algorithms(CommentInput(comment_id="cmp-1", comment=comment)).items():
            rows.append({"algorithm": alg, "status": res.status.value, "label": str(res.emotion_label), "rag": res.rag_used})
        st.dataframe(pd.DataFrame(rows))

with tab3:
    pending = HumanVerificationService().get_pending()
    st.metric("Pending reviews", len(pending))
    st.dataframe(pd.DataFrame(pending) if pending else pd.DataFrame())
    qid = st.text_input("Queue ID")
    ambiguous = st.checkbox("Confirms ambiguous (discard)")
    label = st.text_input("Assigned label (if not ambiguous)")
    if st.button("Submit review") and qid:
        from cas_rag.core.models import HumanReviewDecision

        HumanVerificationService().submit_review(
            qid,
            HumanReviewDecision(
                comment_id="",
                confirms_ambiguous=ambiguous,
                resolvable=not ambiguous,
                assigned_label=label or None,
            ),
        )
        st.success("Submitted")

with tab4:
    ds = settings.project_root / "resources/datasets/sample/comments.csv"
    if st.button("Run batch analytics (CAS-RAG)"):
        engine = PipelineEngine("cas_rag")
        comments = DatasetLoader().load_comments(ds)
        gold = {c.comment_id: c.gold_label.value for c in comments if c.gold_label}
        results = engine.process_dataset(ds)
        report = AnalyticsEngine().build_report(results, gold)
        st.subheader("KPI")
        st.json(report["kpi"])
        charts = report.get("charts", {})
        if "emotion_distribution" in charts:
            st.subheader("Emotion distribution")
            st.bar_chart(pd.Series(charts["emotion_distribution"]))
        if "semantic_categories" in charts:
            st.subheader("Semantic categories")
            st.bar_chart(pd.Series(charts["semantic_categories"]))
        if "rag_vs_no_rag" in charts:
            st.subheader("RAG vs No-RAG")
            st.bar_chart(pd.Series(charts["rag_vs_no_rag"]))
        st.metric("Empathy %", charts.get("empathy_percentage", 0))
        st.metric("Anger+Frustration %", charts.get("anger_frustration_share", 0))
        st.metric("Pending human", charts.get("ambiguous_pending", 0))
