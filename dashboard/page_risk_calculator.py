"""
dashboard/page_risk_calculator.py
---------------------------------
Page 1 — Patient Risk Calculator.

The demo centrepiece. The user enters a patient's clinical details on the
left; on submit, the right-hand panel shows the model's risk prediction with
a SHAP-based explanation of which features drove the result.

The prediction and patient inputs are stored in st.session_state so they
survive Streamlit reruns. This is what allows the chatbot at the bottom of
the page to keep functioning after the user types — without it, each chat
turn would wipe out the prediction display.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.chatbot import render_chatbot, reset_chat
from dashboard.utils import (
    CHEST_PAIN_MAP,
    RESTING_ECG_MAP,
    SLOPE_MAP,
    build_feature_row,
    load_model_bundle,
    predict,
    pretty,
)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def _risk_color(category: str) -> str:
    """Map risk category to a colour for the metric badge."""
    return {
        "Low": "#16a34a",       # green
        "Moderate": "#ea580c",  # orange
        "High": "#dc2626",      # red
    }.get(category, "#6b7280")


def _label_with_value(feature_name: str, patient_value: float) -> str:
    """
    Format a feature label that includes the patient's actual value, so the
    SHAP bar reads like 'Exercise Angina = No' instead of just 'Exercise Angina'.
    """
    base = pretty(feature_name)

    if feature_name.startswith(("chest_pain_type_", "resting_ecg_",
                                 "slope_", "age_group_", "bp_category_",
                                 "cholesterol_band_")):
        if patient_value == 1:
            return f"{base} (yes)"
        else:
            return f"{base} (no)"

    if feature_name in ("sex", "fasting_blood_sugar", "exercise_angina",
                        "cholesterol_was_missing"):
        if feature_name == "sex":
            return f"{base} = {'Male' if patient_value == 1 else 'Female'}"
        return f"{base} = {'Yes' if patient_value == 1 else 'No'}"

    if feature_name in ("age", "resting_bp", "max_heart_rate", "num_vessels"):
        return f"{base} = {int(patient_value)}"
    if feature_name in ("cholesterol", "oldpeak"):
        return f"{base} = {patient_value:g}"

    return base


def _shap_waterfall(top_drivers, top_protectors, base_value, probability,
                    feature_row):
    """
    Build a horizontal bar chart showing which features pushed the prediction
    in each direction, with each bar labelled by the patient's actual value
    for that feature.
    """
    items = []
    for name, val in top_drivers:
        patient_value = feature_row[name].iloc[0]
        items.append({
            "feature": _label_with_value(name, patient_value),
            "shap": val,
            "direction": "Toward Disease",
        })
    for name, val in top_protectors:
        patient_value = feature_row[name].iloc[0]
        items.append({
            "feature": _label_with_value(name, patient_value),
            "shap": val,
            "direction": "Toward Healthy",
        })

    df = pd.DataFrame(items).sort_values("shap")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["feature"],
        x=df["shap"],
        orientation="h",
        marker_color=["#dc2626" if d == "Toward Disease" else "#16a34a"
                      for d in df["direction"]],
        text=[f"{v:+.2f}" for v in df["shap"]],
        textposition="auto",
    ))
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="SHAP contribution (+ = toward disease, − = toward healthy)",
        yaxis_title=None,
        showlegend=False,
    )
    return fig


def _list_to_text(names: list[str]) -> str:
    """Render a list as 'A', 'A and B', or 'A, B, and C'."""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _show_prediction(result: dict, feature_row, patient_inputs: dict) -> None:
    """
    Render the prediction output (badge, recommendation, SHAP chart, summary).
    Called both immediately after a new prediction AND on every rerun where
    a cached prediction exists in session_state.
    """
    st.divider()
    st.subheader("📊 Risk Assessment")

    prob_pct = result["probability"] * 100
    color = _risk_color(result["risk_category"])

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown(
            f"""
            <div style="
                background-color: {color};
                padding: 24px;
                border-radius: 12px;
                text-align: center;
                color: white;
            ">
                <div style="font-size: 14px; opacity: 0.9;">Risk Category</div>
                <div style="font-size: 36px; font-weight: bold; margin: 8px 0;">
                    {result['risk_category']}
                </div>
                <div style="font-size: 28px;">
                    {prob_pct:.1f}%
                </div>
                <div style="font-size: 12px; opacity: 0.85; margin-top: 8px;">
                    probability of heart disease
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        if result["risk_category"] == "High":
            st.error(
                "**Clinical recommendation:** Patient profile suggests "
                "elevated cardiovascular risk. Recommend prompt clinical "
                "evaluation, including 12-lead ECG, lipid panel, and "
                "consideration of stress testing or imaging."
            )
        elif result["risk_category"] == "Moderate":
            st.warning(
                "**Clinical recommendation:** Patient profile shows "
                "moderate risk factors. Recommend lifestyle counselling, "
                "lipid management as appropriate, and follow-up evaluation "
                "within 3–6 months."
            )
        else:
            st.success(
                "**Clinical recommendation:** Patient profile indicates "
                "low cardiovascular risk. Continue routine preventive care "
                "with annual screening as per guidelines."
            )

    st.divider()

    st.subheader("🔍 Why this prediction?")
    st.write(
        "The chart below shows which features pushed this patient's risk "
        "score in each direction. Red bars push toward higher risk; green "
        "bars push toward lower risk. Each bar shows the patient's actual "
        "input value for that feature."
    )

    fig = _shap_waterfall(
        result["top_drivers"],
        result["top_protectors"],
        result["shap_base_value"],
        result["probability"],
        feature_row,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Plain-English summary
    with st.expander("📝 Plain-English summary", expanded=True):

        driver_names = [pretty(n) for n, _ in result["top_drivers"][:3]]
        protector_names = [pretty(n) for n, _ in result["top_protectors"][:3]]

        if result["risk_category"] == "High":
            if driver_names:
                st.markdown(
                    f"The primary factors elevating this patient's risk "
                    f"are **{_list_to_text(driver_names)}**. These features "
                    f"contributed most strongly toward the model's prediction "
                    f"of cardiovascular disease."
                )
            if protector_names:
                st.markdown(
                    f"Partially offsetting this, **{_list_to_text(protector_names)}** "
                    f"pushed the assessment toward lower risk."
                )

        elif result["risk_category"] == "Moderate":
            if driver_names and protector_names:
                st.markdown(
                    f"This patient sits in the moderate-risk band. Factors "
                    f"pushing the assessment upward include "
                    f"**{_list_to_text(driver_names)}**, while "
                    f"**{_list_to_text(protector_names)}** push it downward. "
                    f"The model's overall prediction reflects the balance "
                    f"between these influences."
                )
            elif driver_names:
                st.markdown(
                    f"This patient sits in the moderate-risk band, driven "
                    f"primarily by **{_list_to_text(driver_names)}**."
                )

        else:  # Low
            if protector_names:
                st.markdown(
                    f"This patient's profile shows **low** cardiovascular risk. "
                    f"The strongest protective factors are "
                    f"**{_list_to_text(protector_names)}**, which pushed the "
                    f"model's assessment toward absence of disease."
                )
            if driver_names:
                st.markdown(
                    f"Modest contributions in the other direction came from "
                    f"**{_list_to_text(driver_names)}**, but they were "
                    f"outweighed by the protective factors above."
                )

        st.markdown(
            "---\n"
            "*This is an AI-generated risk assessment for decision support. "
            "Final clinical judgement rests with the treating physician.*"
        )

    st.caption(
        "CardioPredict is a decision-support prototype trained on the "
        "Mendeley Indian Hospital + UCI Cleveland datasets (1,303 patients). "
        "Predictions are intended to assist, not replace, clinical assessment."
    )

    # -------------------------------------------------------------------
    # HeartGuide chatbot — context-aware follow-up Q&A
    # -------------------------------------------------------------------
    st.divider()
    st.subheader("💬 Ask HeartGuide")
    st.write(
        "HeartGuide is a conversational decision-support assistant "
        "that knows this patient's full assessment. Ask follow-up "
        "questions about the prediction, request lifestyle advice, "
        "or explore next clinical steps."
    )

    col_chat_a, col_chat_b = st.columns([5, 1])
    with col_chat_b:
        if st.button("🔄 Reset chat", help="Clear the conversation",
                     key="reset_page1_chat"):
            reset_chat("page1_chat")
            st.rerun()

    render_chatbot(
        session_key="page1_chat",
        patient_inputs=patient_inputs,
        prediction=result,
        intro_message=(
            f"This patient was assessed as **{result['risk_category']} risk** "
            f"({result['probability']*100:.1f}%). Ask me anything about the "
            f"factors driving this assessment or recommended next steps."
        ),
    )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render() -> None:
    st.title("🩺 Patient Risk Calculator")
    st.write(
        "Enter a patient's clinical details below and click **Predict Risk**. "
        "The system returns a cardiovascular risk score and explains the "
        "key drivers of its assessment."
    )

    try:
        bundle = load_model_bundle()
    except Exception as exc:
        st.error(f"Model not available: {exc}")
        st.info("Run `python -m src.train` first to train and save the model.")
        return

    st.divider()

    # -----------------------------------------------------------------------
    # Input form
    # -----------------------------------------------------------------------
    with st.form("patient_form"):
        st.subheader("Patient details")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=18, max_value=100, value=55, step=1,
                help="Patient age in years",
            )
            sex = st.selectbox(
                "Sex",
                options=["Male", "Female"],
                help="Biological sex",
            )
            resting_bp = st.number_input(
                "Resting BP (mm Hg)",
                min_value=80, max_value=220, value=130, step=1,
                help="Systolic blood pressure at rest",
            )
            cholesterol = st.number_input(
                "Total cholesterol (mg/dL)",
                min_value=100, max_value=600, value=220, step=1,
                help="Serum total cholesterol",
            )

        with col2:
            chest_pain = st.selectbox(
                "Chest pain type",
                options=list(CHEST_PAIN_MAP.keys()),
                index=3,
                help="Patient's reported chest pain characteristics",
            )
            fasting_bs = st.selectbox(
                "Fasting blood sugar > 120 mg/dL?",
                options=["No", "Yes"],
                help="Indicator of diabetic / pre-diabetic glucose levels",
            )
            resting_ecg = st.selectbox(
                "Resting ECG result",
                options=list(RESTING_ECG_MAP.keys()),
                help="Findings from a resting electrocardiogram",
            )
            max_hr = st.number_input(
                "Max heart rate achieved",
                min_value=60, max_value=220, value=150, step=1,
                help="Peak heart rate during exercise stress test",
            )

        with col3:
            exercise_angina = st.selectbox(
                "Exercise-induced angina?",
                options=["No", "Yes"],
                help="Did the patient experience chest pain during exercise testing?",
            )
            oldpeak = st.number_input(
                "ST depression (oldpeak)",
                min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                help="ST segment depression induced by exercise (mm)",
            )
            slope = st.selectbox(
                "ST slope",
                options=list(SLOPE_MAP.keys()),
                index=2,
                help="Slope of the peak exercise ST segment (raw category)",
            )
            num_vessels = st.number_input(
                "Number of major vessels affected",
                min_value=0, max_value=3, value=0, step=1,
                help="Diseased coronary vessels visible on fluoroscopy (0-3)",
            )

        submitted = st.form_submit_button("🔮 Predict Risk", type="primary")

    # -----------------------------------------------------------------------
    # On submit: run prediction and cache it in session_state
    # -----------------------------------------------------------------------
    if submitted:
        feature_row = build_feature_row(
            age=age, sex=sex, chest_pain=chest_pain,
            resting_bp=resting_bp, cholesterol=cholesterol,
            fasting_blood_sugar_high=fasting_bs,
            resting_ecg=resting_ecg, max_heart_rate=max_hr,
            exercise_angina=exercise_angina, oldpeak=oldpeak,
            slope=slope, num_vessels=num_vessels,
            feature_columns=bundle["feature_columns"],
        )

        result = predict(feature_row)

        # Cache everything in session_state so the prediction survives reruns
        # triggered by chatbot input.
        st.session_state["page1_result"] = result
        st.session_state["page1_feature_row"] = feature_row
        st.session_state["page1_patient_inputs"] = {
            "age": age, "sex": sex, "resting_bp": resting_bp,
            "cholesterol": cholesterol, "chest_pain": chest_pain,
            "fasting_bs": fasting_bs, "resting_ecg": resting_ecg,
            "max_hr": max_hr, "exercise_angina": exercise_angina,
            "oldpeak": oldpeak, "slope": slope, "num_vessels": num_vessels,
        }
        # Clear any previous chat when a new prediction is made
        reset_chat("page1_chat")

    # -----------------------------------------------------------------------
    # Show the prediction if one exists in session_state (covers both:
    # — the rerun immediately after the form submit
    # — every subsequent rerun triggered by chatbot input)
    # -----------------------------------------------------------------------
    if "page1_result" in st.session_state:
        _show_prediction(
            st.session_state["page1_result"],
            st.session_state["page1_feature_row"],
            st.session_state["page1_patient_inputs"],
        )