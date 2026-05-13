"""
dashboard/page_analytics.py
---------------------------
Page 2 — Analytics & Insights.

Provides a population-level view of the training data:
  - Summary statistics for Mendeley (Indian) vs UCI (Western benchmark)
  - Age and risk distribution by cohort
  - Feature comparison: how do Indian patients differ from Western ones?
  - Global SHAP feature importance (the plots from src/explain.py)

This page serves two audiences:
  - The mentor / examiner who wants to verify the data is real and the
    comparative-analysis claim of the title is supported.
  - A clinical user who wants to understand the population the model was
    trained on, to judge whether predictions for their patient are likely
    to be reliable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.db import PROJECT_ROOT, Tables, get_engine


MODELS_DIR = PROJECT_ROOT / "models"
SHAP_BAR_PATH = MODELS_DIR / "shap_summary_bar.png"
SHAP_BEESWARM_PATH = MODELS_DIR / "shap_summary_beeswarm.png"


@st.cache_data
def _load_clean_data() -> pd.DataFrame:
    """Load patients_clean from SQLite. Cached for fast page reloads."""
    engine = get_engine()
    return pd.read_sql(f"SELECT * FROM {Tables.CLEAN}", engine)


def _cohort_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a side-by-side summary comparing the two cohorts."""
    summary = []
    for source, label in [("mendeley_india", "Mendeley (India)"),
                          ("uci_cleveland", "UCI (Cleveland)")]:
        sub = df[df["source"] == source]
        summary.append({
            "Cohort": label,
            "Patients (n)": len(sub),
            "Age (mean)": f"{sub['age'].mean():.1f}",
            "% Male": f"{(sub['sex'] == 1).mean() * 100:.1f}%",
            "Cholesterol (mean)": f"{sub['cholesterol'].mean():.0f}",
            "Resting BP (mean)": f"{sub['resting_bp'].mean():.0f}",
            "Max HR (mean)": f"{sub['max_heart_rate'].mean():.0f}",
            "% with disease": f"{(sub['target'] == 1).mean() * 100:.1f}%",
        })
    return pd.DataFrame(summary)


def render() -> None:
    st.title("📊 Analytics & Insights")
    st.write(
        "Population-level patterns in the training data, with comparative "
        "analysis between Indian and Western patient cohorts. This view "
        "supports the core academic argument that India-specific cardiac "
        "risk profiles differ from global benchmarks."
    )

    try:
        df = _load_clean_data()
    except Exception as exc:
        st.error(f"Could not load patients_clean: {exc}")
        st.info("Run the pipeline first: `python -m src.ingest`, then "
                "`python -m src.clean`, then `python -m src.features`.")
        return

    st.divider()

    # -----------------------------------------------------------------------
    # Section 1: Cohort summary
    # -----------------------------------------------------------------------
    st.subheader("Cohort summary")
    st.write(
        "Side-by-side summary of the two source cohorts. Notice that the "
        "Indian cohort skews older and shows substantially higher mean "
        "cholesterol — consistent with a tertiary-care referral population "
        "rather than community screening."
    )

    summary = _cohort_summary_table(df)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()

    # -----------------------------------------------------------------------
    # Section 2: Age distribution
    # -----------------------------------------------------------------------
    st.subheader("Age distribution by cohort")

    fig = px.histogram(
        df,
        x="age",
        color="source",
        nbins=30,
        barmode="overlay",
        opacity=0.6,
        labels={"age": "Age (years)", "source": "Cohort"},
        color_discrete_map={
            "mendeley_india": "#dc2626",
            "uci_cleveland": "#2563eb",
        },
    )
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Both cohorts cover roughly the same adult age range. The Indian "
        "cohort has a slightly heavier upper tail (more patients aged 60+)."
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Section 3: Disease prevalence by cohort
    # -----------------------------------------------------------------------
    st.subheader("Disease prevalence by cohort")

    prevalence = (
        df.groupby("source")["target"]
        .agg(["sum", "count"])
        .reset_index()
    )
    prevalence["pct_disease"] = prevalence["sum"] / prevalence["count"] * 100
    prevalence["cohort_label"] = prevalence["source"].map({
        "mendeley_india": "Mendeley (India)",
        "uci_cleveland": "UCI (Cleveland)",
    })

    fig = px.bar(
        prevalence,
        x="cohort_label",
        y="pct_disease",
        text=prevalence["pct_disease"].apply(lambda v: f"{v:.1f}%"),
        labels={"cohort_label": "Cohort", "pct_disease": "% with disease"},
        color="cohort_label",
        color_discrete_map={
            "Mendeley (India)": "#dc2626",
            "UCI (Cleveland)": "#2563eb",
        },
    )
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        yaxis_range=[0, 100],
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Higher prevalence in the Indian cohort reflects the multispecialty "
        "hospital workup nature of the source rather than population-level "
        "disease rates. The UCI cohort was likewise drawn from a "
        "cardiology referral context (Cleveland Clinic catheterisation)."
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Section 4: Key feature comparison
    # -----------------------------------------------------------------------
    st.subheader("Feature comparison: India vs UCI")
    st.write(
        "Distribution of key clinical features across the two cohorts. "
        "Use the dropdown to explore individual features."
    )

    feature_choice = st.selectbox(
        "Feature to compare",
        options=[
            ("cholesterol", "Cholesterol (mg/dL)"),
            ("resting_bp", "Resting Blood Pressure (mm Hg)"),
            ("max_heart_rate", "Max Heart Rate Achieved"),
            ("oldpeak", "ST Depression (oldpeak)"),
            ("age", "Age (years)"),
        ],
        format_func=lambda x: x[1],
    )
    feat_col = feature_choice[0]
    feat_label = feature_choice[1]

    fig = go.Figure()
    for source, label, color in [
        ("mendeley_india", "Mendeley (India)", "#dc2626"),
        ("uci_cleveland", "UCI (Cleveland)", "#2563eb"),
    ]:
        sub = df[df["source"] == source]
        fig.add_trace(go.Box(
            y=sub[feat_col],
            name=label,
            marker_color=color,
            boxmean=True,
        ))
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title=feat_label,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -----------------------------------------------------------------------
    # Section 5: Global SHAP feature importance
    # -----------------------------------------------------------------------
    st.subheader("🔍 What features drive the model's predictions?")
    st.write(
        "Global SHAP feature importance across all 1,303 training patients. "
        "These plots tell us which features the model relies on most when "
        "making its risk assessments. Generated by `python -m src.explain`."
    )

    tab1, tab2 = st.tabs(["Bar plot", "Beeswarm plot"])

    with tab1:
        if SHAP_BAR_PATH.exists():
            st.image(
                str(SHAP_BAR_PATH),
                caption="Top 20 features by mean absolute SHAP value. "
                        "ST slope categories, vessel involvement, and "
                        "resting blood pressure dominate.",
                use_container_width=True,
            )
        else:
            st.warning(
                f"SHAP bar plot not found at {SHAP_BAR_PATH}. "
                "Run `python -m src.explain` to generate it."
            )

    with tab2:
        if SHAP_BEESWARM_PATH.exists():
            st.image(
                str(SHAP_BEESWARM_PATH),
                caption="Beeswarm plot. Each dot is one patient; colour shows "
                        "feature value (red = high, blue = low); horizontal "
                        "position shows the SHAP contribution to that "
                        "patient's prediction.",
                use_container_width=True,
            )
        else:
            st.warning(
                f"SHAP beeswarm plot not found at {SHAP_BEESWARM_PATH}. "
                "Run `python -m src.explain` to generate it."
            )

    st.divider()

    # -----------------------------------------------------------------------
    # Section 6: Dataset audit note (important for viva defence)
    # -----------------------------------------------------------------------
    with st.expander("ℹ️ A note on dataset selection and exclusion", expanded=False):
        st.markdown(
            """
            **Datasets used:**
            - **Mendeley Indian Hospital CVD Dataset** — 1,000 patients from
              an Indian multispecialty hospital. Primary training source.
            - **UCI Cleveland Heart Disease Dataset** — 303 patients, the
              standard global benchmark. Used for comparative analysis.

            **Dataset excluded after audit:**
            A third dataset (Kaggle "Heart Attack Risk and Prediction Dataset
            in India", 10,000 records) was initially considered. Upon audit,
            it showed strong signals of synthetic generation:
            - Zero missing values across 260,000 cells
            - All numerical features uniformly distributed between bounded
              limits, with means at exact range midpoints
            - Lipid panel values (Total cholesterol, LDL, HDL, Triglycerides)
              biologically inconsistent — they fail the standard
              Total ≈ LDL + HDL + (Trig/5) constraint
            - Dataset description names no clinical source or institution

            The dataset was retained on disk for methodology reference only
            and excluded from model training. This audit is documented in
            `src/ingest.py` and discussed in the thesis methodology chapter.
            """
        )