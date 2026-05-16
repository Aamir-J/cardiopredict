"""
app.py
------
CardioPredict — Streamlit application entry point.

This module is the entrypoint Streamlit Cloud runs. It configures the page,
sets up the sidebar navigation, and dispatches to the correct page module
based on the selected option.

The four pages are implemented separately in dashboard/page_*.py modules
and exposed via a render() function each. Keeping app.py thin lets us add
or remove pages without touching the routing logic.
"""

from __future__ import annotations

import streamlit as st

from dashboard import (
    page_analytics,
    page_metrics,
    page_risk_calculator,
    page_roi,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
# initial_sidebar_state="collapsed" gives a cleaner first impression on
# mobile (no hamburger menu obscuring the content). Desktop users see a
# small toggle to open the sidebar; mobile users tap the hamburger icon.
st.set_page_config(
    page_title="CardioPredict",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Sidebar — title, navigation, disclaimer
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🩺 CardioPredict")
    st.caption("AI-Powered Heart Disease Risk Detection")
    st.caption("for Indian Patients")
    st.divider()

    page_choice = st.radio(
        "Navigation",
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
        "⚠️ **Decision-support prototype.** This tool supports "
        "clinical reasoning. It does not replace medical judgement."
    )


# ---------------------------------------------------------------------------
# Dispatch to the correct page
# ---------------------------------------------------------------------------
if page_choice == "🩺 Risk Calculator":
    page_risk_calculator.render()
elif page_choice == "📊 Analytics & Insights":
    page_analytics.render()
elif page_choice == "💰 ROI Calculator":
    page_roi.render()
elif page_choice == "📈 Model Performance":
    page_metrics.render()