"""
dashboard/page_metrics.py
-------------------------
Page 4 — Model Performance Report.

Reads the model_metrics table populated by src/train.py and displays:
  - Comparison table of all three trained models
  - Highlight of the winning model
  - Confusion matrix visualisation for the winner
  - Explanation of why recall is the primary metric
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.db import Tables, get_engine


def _load_metrics() -> pd.DataFrame:
    """Load the model_metrics table from SQLite."""
    engine = get_engine()
    return pd.read_sql(f"SELECT * FROM {Tables.MODEL_METRICS}", engine)


def render() -> None:
    st.title("📈 Model Performance Report")
    st.write(
        "Performance comparison across the three classifiers trained on the "
        "combined Mendeley + UCI dataset (1,303 patients, 80/20 stratified "
        "train/test split, random_state=42)."
    )

    # Load metrics
    try:
        metrics = _load_metrics()
    except Exception as exc:
        st.error(f"Could not load model metrics: {exc}")
        st.info("Run `python -m src.train` first to populate the metrics table.")
        return

    # -----------------------------------------------------------------------
    # Section 1: Why recall as primary metric
    # -----------------------------------------------------------------------
    with st.expander("ℹ️ Why is recall the primary metric?", expanded=False):
        st.markdown(
            """
            In medical screening, a **false negative** — predicting healthy
            when the patient has disease — has higher clinical and economic
            cost than a **false positive**. Missing a sick patient delays
            treatment and can be fatal; over-flagging a healthy patient
            results in additional (but generally low-cost) confirmatory tests.

            **Recall = True Positives / (True Positives + False Negatives)**

            captures exactly this trade-off. A recall of 0.96 means the model
            correctly identifies 96% of patients who actually have heart disease.

            We also report **AUC-ROC, accuracy, precision, and F1** for completeness,
            but recall is the metric we optimised for via balanced class weighting
            during training.
            """
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Section 2: Headline metrics — the winner
    # -----------------------------------------------------------------------
    winner = metrics[metrics["is_winner"] == 1].iloc[0]

    st.subheader(f"🏆 Selected model: {winner['model_name']}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recall", f"{winner['recall']:.1%}",
                help="Primary metric — true positive rate")
    col2.metric("AUC-ROC", f"{winner['auc_roc']:.3f}",
                help="Area under the ROC curve")
    col3.metric("Precision", f"{winner['precision']:.1%}",
                help="Of patients flagged, how many actually have disease")
    col4.metric("F1 Score", f"{winner['f1']:.3f}",
                help="Harmonic mean of precision and recall")

    st.divider()

    # -----------------------------------------------------------------------
    # Section 3: Full comparison table
    # -----------------------------------------------------------------------
    st.subheader("All models compared")

    display_cols = ["model_name", "recall", "auc_roc", "accuracy",
                    "precision", "f1", "cv_recall_mean", "cv_recall_std"]
    display = metrics[display_cols].copy()
    display.columns = ["Model", "Recall", "AUC-ROC", "Accuracy",
                       "Precision", "F1", "CV Recall (mean)", "CV Recall (std)"]

    # Format numbers nicely
    for col in ["Recall", "AUC-ROC", "Accuracy", "Precision", "F1",
                "CV Recall (mean)", "CV Recall (std)"]:
        display[col] = display[col].apply(lambda x: f"{x:.4f}")

    st.dataframe(display, use_container_width=True, hide_index=True)

    st.caption(
        "**CV** = 5-fold stratified cross-validation on the training set, "
        "reporting mean and standard deviation of recall across folds. "
        "Low standard deviation indicates the model performs consistently "
        "across different data splits."
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Section 4: Confusion matrix for the winner
    # -----------------------------------------------------------------------
    st.subheader(f"Confusion matrix — {winner['model_name']}")

    cm = json.loads(winner["confusion_matrix"])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No Disease", "Actual: Disease"],
        columns=["Predicted: No Disease", "Predicted: Disease"],
    )

    # Plotly heatmap for visual appeal
    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig.update_layout(
        height=400,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    tn, fp = cm[0]
    fn, tp = cm[1]
    total = tn + fp + fn + tp

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**True Negatives:** {tn}")
        st.caption("Patients correctly identified as healthy")
        st.markdown(f"**False Positives:** {fp}")
        st.caption("Healthy patients incorrectly flagged")
    with col_b:
        st.markdown(f"**True Positives:** {tp}")
        st.caption("Patients correctly flagged for disease")
        st.markdown(f"**False Negatives:** {fn}")
        st.caption("Sick patients missed (highest concern)")

    st.success(
        f"Out of **{total}** test patients, the model missed only **{fn}** "
        f"sick patients and incorrectly flagged **{fp}** healthy patients."
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Section 5: Training metadata
    # -----------------------------------------------------------------------
    st.caption(
        f"All models trained on {winner['trained_at']}. "
        "Pipeline reproducible via `python -m src.train`."
    )