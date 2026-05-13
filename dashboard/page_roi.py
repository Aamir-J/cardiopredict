"""
dashboard/page_roi.py
---------------------
Page 3 — Hospital ROI Calculator.

A pragmatic, inputs-driven cost-benefit calculator for hospital administrators
evaluating CardioPredict deployment. No machine learning involved — purely a
financial model with adjustable assumptions.

The calculator estimates the financial impact of deploying CardioPredict for
early cardiac screening, based on three core mechanisms of value:

  1. Early detection avoids high-cost acute events.
     Treating a heart attack in India costs ₹3-8 lakh on average. Catching
     elevated risk early and intervening costs a small fraction of that.

  2. Faster triage frees up cardiologist time.
     A 2-minute AI-assisted screen vs a 30-minute manual review per patient
     translates to higher patient throughput per cardiologist day.

  3. Insurance / corporate wellness programmes reward preventive care.
     Insurers increasingly offer premium rebates and bonuses for hospitals
     that demonstrate measurable preventive screening outcomes.

All assumptions are user-adjustable so different hospitals can model their
own scenarios. Default values are pegged to publicly available figures from
ICMR, NHA, and industry reports. Values are illustrative; the methodology
chapter notes these are scenario assumptions, not validated cost models.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _format_inr(value: float) -> str:
    """Format a number as INR (₹) with lakh/crore-aware short labels."""
    if value >= 1e7:
        return f"₹{value / 1e7:.2f} Cr"
    if value >= 1e5:
        return f"₹{value / 1e5:.2f} L"
    return f"₹{value:,.0f}"


def render() -> None:
    st.title("💰 Hospital ROI Calculator")
    st.write(
        "Estimate the annual financial impact of deploying CardioPredict at "
        "your hospital. Adjust the assumptions in the sidebar — the model "
        "recomputes instantly."
    )

    st.info(
        "📌 **This is a scenario model**, not a validated cost study. "
        "All defaults are based on publicly available Indian hospital "
        "and ICMR estimates and should be adjusted to reflect your "
        "institution's actual figures."
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Input panel — two columns of assumptions
    # -----------------------------------------------------------------------
    st.subheader("Scenario assumptions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Hospital scale**")
        patients_screened = st.number_input(
            "Annual patients screened",
            min_value=100, max_value=100000, value=5000, step=100,
            help="Total patients passing through preventive cardiac screening per year",
        )
        baseline_risk_pct = st.slider(
            "Baseline % at elevated cardiac risk",
            min_value=5, max_value=60, value=25,
            help="Estimated proportion of the screened population who actually "
                 "have elevated cardiovascular risk",
        )

        st.markdown("**Clinical effectiveness**")
        detection_uplift_pct = st.slider(
            "Detection uplift with CardioPredict (%)",
            min_value=5, max_value=50, value=20,
            help="Additional high-risk patients correctly identified above the "
                 "current standard-of-care detection rate",
        )
        intervention_acceptance_pct = st.slider(
            "% of detected patients who accept intervention",
            min_value=20, max_value=100, value=60,
            help="Proportion of flagged patients who actually engage in "
                 "preventive treatment (lifestyle, medication, monitoring)",
        )
        avoided_event_rate_pct = st.slider(
            "% of accepted interventions that prevent an acute event",
            min_value=5, max_value=50, value=15,
            help="Proportion of preventive interventions that successfully "
                 "avert what would otherwise have been a hospitalisation",
        )

    with col2:
        st.markdown("**Cost assumptions (INR)**")
        cost_acute_event = st.number_input(
            "Avg cost of acute cardiac event (₹)",
            min_value=100000, max_value=2000000, value=500000, step=50000,
            help="Mean cost of treating an acute MI in Indian hospitals "
                 "(angiography, stenting, ICU stay, follow-up)",
        )
        cost_preventive_per_patient = st.number_input(
            "Avg cost of preventive intervention per patient (₹)",
            min_value=1000, max_value=100000, value=15000, step=1000,
            help="Annual cost of preventive care: consultations, lipid panels, "
                 "medication, lifestyle counselling",
        )

        st.markdown("**Platform cost**")
        cardiopredict_cost_per_screen = st.number_input(
            "CardioPredict cost per screen (₹)",
            min_value=10, max_value=500, value=50, step=10,
            help="Per-patient cost of running CardioPredict (SaaS pricing)",
        )
        cardiopredict_setup_cost = st.number_input(
            "One-time setup / integration cost (₹)",
            min_value=0, max_value=2000000, value=200000, step=10000,
            help="One-time deployment, training, and EHR integration cost",
        )

    st.divider()

    # -----------------------------------------------------------------------
    # The model
    # -----------------------------------------------------------------------
    # Number of patients in each stage of the funnel
    n_at_risk = patients_screened * baseline_risk_pct / 100
    n_additional_detected = n_at_risk * detection_uplift_pct / 100
    n_engage = n_additional_detected * intervention_acceptance_pct / 100
    n_events_avoided = n_engage * avoided_event_rate_pct / 100

    # Financial impact
    gross_savings = n_events_avoided * cost_acute_event
    intervention_cost = n_engage * cost_preventive_per_patient
    platform_cost_year1 = (
        patients_screened * cardiopredict_cost_per_screen
        + cardiopredict_setup_cost
    )
    platform_cost_year2 = patients_screened * cardiopredict_cost_per_screen

    net_savings_year1 = gross_savings - intervention_cost - platform_cost_year1
    net_savings_year2 = gross_savings - intervention_cost - platform_cost_year2

    # ROI %
    roi_year1 = (
        (net_savings_year1 / platform_cost_year1 * 100)
        if platform_cost_year1 > 0 else 0
    )
    roi_year2 = (
        (net_savings_year2 / platform_cost_year2 * 100)
        if platform_cost_year2 > 0 else 0
    )

    # -----------------------------------------------------------------------
    # Headline metrics
    # -----------------------------------------------------------------------
    st.subheader("Projected outcomes")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Acute events avoided / year",
        f"{n_events_avoided:.0f}",
        help="Estimated annual hospitalisations prevented",
    )
    m2.metric(
        "Gross savings",
        _format_inr(gross_savings),
        help="Total avoided treatment cost",
    )
    m3.metric(
        "Net savings (Year 1)",
        _format_inr(net_savings_year1),
        help="Savings after intervention and platform costs, "
             "including setup",
    )
    m4.metric(
        "ROI (Year 2 onwards)",
        f"{roi_year2:.0f}%",
        help="Return on investment after the one-time setup cost is paid off",
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Funnel visualisation
    # -----------------------------------------------------------------------
    st.subheader("Patient funnel")

    funnel = pd.DataFrame({
        "Stage": [
            "Patients screened",
            "At elevated risk",
            "Additionally detected",
            "Engage in prevention",
            "Acute events avoided",
        ],
        "Count": [
            patients_screened,
            n_at_risk,
            n_additional_detected,
            n_engage,
            n_events_avoided,
        ],
    })

    fig = go.Figure(go.Funnel(
        y=funnel["Stage"],
        x=funnel["Count"],
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(color=["#1d4ed8", "#2563eb", "#3b82f6",
                           "#60a5fa", "#93c5fd"]),
    ))
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -----------------------------------------------------------------------
    # Detailed cost breakdown
    # -----------------------------------------------------------------------
    st.subheader("Financial breakdown")

    breakdown = pd.DataFrame({
        "Item": [
            "Gross savings from avoided events",
            "(−) Preventive intervention cost",
            "(−) CardioPredict platform cost (Year 1)",
            "(−) CardioPredict platform cost (Year 2+)",
            "Net savings — Year 1",
            "Net savings — Year 2 onwards",
        ],
        "Amount (INR)": [
            _format_inr(gross_savings),
            f"({_format_inr(intervention_cost)})",
            f"({_format_inr(platform_cost_year1)})",
            f"({_format_inr(platform_cost_year2)})",
            _format_inr(net_savings_year1),
            _format_inr(net_savings_year2),
        ],
    })
    st.dataframe(breakdown, use_container_width=True, hide_index=True)

    if net_savings_year1 > 0:
        st.success(
            f"💚 Under these assumptions, CardioPredict pays for itself in "
            f"Year 1 with net positive savings of {_format_inr(net_savings_year1)}. "
            f"Year-2-onward ROI is {roi_year2:.0f}%."
        )
    else:
        breakeven_events = (platform_cost_year1 + intervention_cost) / cost_acute_event
        st.warning(
            f"⚠️ Under these assumptions, the platform does not break even in "
            f"Year 1. Breakeven would require {breakeven_events:.0f} acute "
            f"events avoided per year (currently modelled at "
            f"{n_events_avoided:.0f}). Consider adjusting the screening "
            f"volume or intervention acceptance rate."
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Assumptions methodology note
    # -----------------------------------------------------------------------
    with st.expander("ℹ️ Methodology and assumption sources", expanded=False):
        st.markdown(
            """
            **Default assumption sources:**

            - **Avg cost of acute cardiac event (₹5L default):** Mean cost
              of acute MI treatment in Indian tertiary hospitals, including
              coronary angiography, stenting, ICU stay, and 90-day
              follow-up. Range across published studies: ₹3-8 lakh.

            - **Avg preventive cost (₹15k default):** Annual cost of
              guideline-based preventive cardiology — lipid management,
              quarterly consultations, basic monitoring. Drawn from
              private-sector pricing in Indian metros.

            - **Detection uplift (20% default):** Conservative estimate of
              additional high-risk patients identified by an ML-augmented
              workflow vs current standard of care, based on comparative
              AUC differences between Framingham (≈0.75) and India-trained
              models (≈0.85+).

            **Important caveats:**

            - This is a *scenario tool*, not a validated cost study. Real-world
              outcomes will vary substantially based on patient population,
              baseline detection rates, and intervention adherence.

            - The model assumes the platform delivers detection uplift
              *independently* of physician judgement. In practice, value
              comes from the human-AI combination, not the AI alone.

            - Avoided event probabilities are aggregate population estimates.
              Individual patient outcomes cannot be predicted from this model.
            """
        )