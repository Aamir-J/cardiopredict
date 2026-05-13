"""
app.py
------
CardioPredict — Streamlit dashboard entry point.

This is the main file Streamlit runs to launch the dashboard.
It handles:
  - Page configuration (title, icon, layout)
  - Sidebar navigation between the four dashboard pages
  - Routing to the appropriate page module

Each page lives in its own file under dashboard/, so the routing logic
here stays simple and the pages stay independently testable.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioPredict",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar — branding + navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🫀 CardioPredict")
    st.caption("AI-Powered Heart Disease Risk Detection")
    # st.caption("for Indian Patients")
    st.divider()

    page = st.radio(
        "Navigate",
        options=[
            "🩺 Risk Calculator",
            "📊 Analytics & Insights",
            "💰 ROI Calculator",
            "📈 Model Performance",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(
        "⚠️ **Decision-support prototype.** "
        "This tool supports clinical reasoning. "
        "It does not replace medical judgement."
    )


# ---------------------------------------------------------------------------
# Page routing — call the appropriate render function based on selection
# ---------------------------------------------------------------------------
if page == "🩺 Risk Calculator":
    from dashboard.page_risk_calculator import render
    render()

elif page == "📊 Analytics & Insights":
    from dashboard.page_analytics import render
    render()

elif page == "💰 ROI Calculator":
    from dashboard.page_roi import render
    render()

elif page == "📈 Model Performance":
    from dashboard.page_metrics import render
    render()