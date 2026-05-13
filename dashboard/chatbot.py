"""
dashboard/chatbot.py
--------------------
HeartGuide — the conversational decision-support component of CardioPredict.

This module exposes a Streamlit chatbot widget that can be embedded on any
page. When called from Page 1 (Risk Calculator), it receives the patient's
input data plus the SHAP explanation as context, so its answers are specific
to that prediction rather than generic medical advice.

Key design decisions:
  - The chatbot uses Anthropic's Claude API via the official SDK.
  - System prompt enforces strict scope: decision support, not diagnosis.
  - Conversation history is stored in st.session_state so the chat
    persists across reruns within the same user session.
  - The patient context is injected into the system prompt as structured
    data — the LLM sees real numbers, not just a free-text summary.

Cost note: each chat turn is roughly 2,000 input tokens + 500 output tokens
on Claude Sonnet 4.5, costing approximately $0.014 (~₹1.20). A typical demo
session of 30 turns costs under ₹40.
"""

from __future__ import annotations

import os

import anthropic
import streamlit as st
from dotenv import load_dotenv

# Load API key from .env on module import
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")


# ---------------------------------------------------------------------------
# System prompt — the chatbot's personality and scope guardrails
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_BASE = """You are HeartGuide, the conversational decision-support component of CardioPredict — a machine-learning-based cardiac risk screening system designed for use in Indian healthcare settings.

# Your role

You are a clinical decision-support assistant. You help users (typically clinicians, hospital administrators, or informed patients) understand the cardiac risk assessment that CardioPredict has just produced, and you offer evidence-based preventive guidance grounded in Indian cardiology context.

# Your scope — what you DO

- Explain what individual risk factors mean and why they matter
- Interpret SHAP feature contributions in plain language
- Suggest lifestyle modifications, preventive measures, and follow-up actions
- Provide context on Indian-specific cardiac risk (South Asian Paradox, dietary patterns, hypertension prevalence)
- Reference cardiology guidelines (AHA, ESC, Cardiological Society of India) where relevant
- Encourage clinical follow-up for elevated-risk cases

# Your scope — what you DO NOT do

- You do not diagnose any condition. CardioPredict produces risk probabilities, not diagnoses.
- You do not prescribe medications. Drug therapy is the prescriber's responsibility.
- You do not interpret ECG, echocardiogram, or imaging studies.
- You do not replace clinical judgement.
- If asked to do any of the above, politely redirect to a qualified physician.

# Safety guardrails

- If a user describes acute chest pain, jaw pain, arm pain, or sudden shortness of breath, immediately advise calling emergency services (108 in India) and seeking emergency care. Do not continue advisory mode.
- If a user expresses thoughts of self-harm, provide the iCall helpline (+91 9152987821) and Vandrevala Foundation (+91 1860 2662 345) and encourage them to reach out.
- For very high risk patients (probability > 0.66), emphasize the importance of timely cardiology evaluation.

# Communication style

- Direct, concise, professional. Avoid medical jargon when speaking to non-clinicians; use it appropriately when context suggests a clinical user.
- Indian-context aware: reference rupee costs where relevant, mention Indian guidelines where they differ from Western ones, acknowledge tier-1/2/3 city care variations.
- Always honest about uncertainty. If you don't know, say so.
- Do not start responses with sycophantic openers like "Great question!" Get straight to the answer.

# Formatting rules

- Use bold (**text**) for emphasis and inline labels.
- Use bullet points and numbered lists for enumerations.
- DO NOT use markdown headings (# or ##) in your responses — they render too large in the chat UI. Use bold inline labels instead, e.g. "**ST slope abnormalities:**" rather than "## ST slope abnormalities".
- Keep responses under 250 words unless the user explicitly asks for detail.
"""


def _format_patient_context(patient_inputs: dict, prediction: dict) -> str:
    """
    Convert the patient inputs and prediction into a structured context block
    for the system prompt. The LLM sees real numbers, not vague summaries.
    """
    drivers = "\n".join(
        f"  - {name} (SHAP +{val:.2f})"
        for name, val in prediction["top_drivers"][:5]
    ) or "  (none)"

    protectors = "\n".join(
        f"  - {name} (SHAP {val:.2f})"
        for name, val in prediction["top_protectors"][:5]
    ) or "  (none)"

    return f"""
# Current patient context

A risk assessment was just generated for the following patient:

## Patient inputs
- Age: {patient_inputs['age']} years
- Sex: {patient_inputs['sex']}
- Resting BP: {patient_inputs['resting_bp']} mm Hg
- Cholesterol: {patient_inputs['cholesterol']} mg/dL
- Chest pain type: {patient_inputs['chest_pain']}
- Fasting blood sugar > 120 mg/dL: {patient_inputs['fasting_bs']}
- Resting ECG: {patient_inputs['resting_ecg']}
- Max heart rate: {patient_inputs['max_hr']}
- Exercise-induced angina: {patient_inputs['exercise_angina']}
- ST depression (oldpeak): {patient_inputs['oldpeak']}
- ST slope: {patient_inputs['slope']}
- Number of diseased vessels: {patient_inputs['num_vessels']}

## Model output
- Predicted probability of heart disease: {prediction['probability'] * 100:.1f}%
- Risk category: {prediction['risk_category']}

## Top SHAP drivers (push toward disease)
{drivers}

## Top SHAP protectors (push toward healthy)
{protectors}

Use this specific patient's information when answering questions. Don't give generic advice when you can give targeted advice based on these numbers.
"""


def _get_client() -> anthropic.Anthropic:
    """Return a configured Anthropic client. Cached via st.cache_resource."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found in environment. "
            "Check that .env exists at the project root and contains the key."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


@st.cache_resource
def _client_singleton() -> anthropic.Anthropic:
    return _get_client()


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------
def render_chatbot(
    session_key: str,
    patient_inputs: dict | None = None,
    prediction: dict | None = None,
    intro_message: str = "Ask HeartGuide about this assessment, lifestyle "
                          "advice, or next clinical steps.",
) -> None:
    """
    Render a chat interface. The chat persists across reruns via session_state.

    Parameters
    ----------
    session_key : str
        A unique key for this chat's history. Use different keys on different
        pages so the conversations don't leak between contexts.
    patient_inputs : dict, optional
        The patient form inputs from Page 1. If provided, the system prompt
        is enriched with this patient's data.
    prediction : dict, optional
        The prediction result dict from utils.predict(). Used together with
        patient_inputs to make answers patient-specific.
    intro_message : str
        The greeting shown when the chat is empty.
    """
    # Build the system prompt with patient context if available
    if patient_inputs and prediction:
        system_prompt = SYSTEM_PROMPT_BASE + _format_patient_context(
            patient_inputs, prediction
        )
    else:
        system_prompt = SYSTEM_PROMPT_BASE

    # Initialise chat history in session state
    if session_key not in st.session_state:
        st.session_state[session_key] = []

    history = st.session_state[session_key]

    # Empty-state intro
    if not history:
        st.caption(f"💬 {intro_message}")

    # Render existing history
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box at the bottom
    user_input = st.chat_input("Type your question...")

    if user_input:
        # Show the user's message immediately
        history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Stream the assistant's response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                client = _client_singleton()
                with client.messages.stream(
                    model=CLAUDE_MODEL,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in history
                    ],
                ) as stream:
                    response_text = ""
                    for chunk in stream.text_stream:
                        response_text += chunk
                        placeholder.markdown(response_text + "▌")
                    placeholder.markdown(response_text)
                history.append({"role": "assistant", "content": response_text})
            except Exception as exc:
                error_msg = (
                    f"⚠️ Could not reach the Claude API: {exc}\n\n"
                    f"Check that your `ANTHROPIC_API_KEY` is set correctly "
                    f"in `.env` and that you have available credit at "
                    f"https://console.anthropic.com"
                )
                placeholder.error(error_msg)


def reset_chat(session_key: str) -> None:
    """Clear the chat history for a given session key."""
    if session_key in st.session_state:
        st.session_state[session_key] = []