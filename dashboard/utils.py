"""
dashboard/utils.py
------------------
Shared utilities for the CardioPredict dashboard.

Provides:
  - Cached model loading (one load per session, not per page interaction)
  - A clean predict() function that takes raw patient inputs and returns
    a risk score plus the SHAP explanation
  - Lookup tables for translating user-facing labels (e.g. chest pain types)
    into the encoded values the model was trained on

This module is the single point of contact between the dashboard pages and
the trained model. If the model changes, only this file needs updating.

Note on ST slope encoding:
  The two source datasets use different slope codes (Mendeley: 0-3, UCI: 1-3),
  and the categorical meanings do not perfectly align. We expose the raw
  category numbers in the UI rather than claiming clinical labels (upsloping,
  flat, downsloping) that may not be consistent across datasets.
"""

from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from src.db import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "heart_model.pkl"
EXPLAINER_PATH = MODELS_DIR / "shap_explainer.pkl"


# ---------------------------------------------------------------------------
# Cached model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_bundle() -> dict:
    """
    Load the trained model bundle (model + scaler + feature columns).

    Cached with @st.cache_resource so the model is loaded once per session,
    not on every UI interaction. Loading XGBoost + scaler takes ~1 second,
    which adds up fast without caching.
    """
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_shap_explainer():
    """Load the persisted SHAP TreeExplainer."""
    return joblib.load(EXPLAINER_PATH)


# ---------------------------------------------------------------------------
# Label → encoded value lookups
# ---------------------------------------------------------------------------
# These dictionaries translate user-friendly labels (what the form shows)
# into the numeric encodings the model was trained on.

SEX_MAP = {"Female": 0, "Male": 1}

CHEST_PAIN_MAP = {
    "Typical angina (classic chest pain on exertion)": 1,
    "Atypical angina (chest discomfort, not exertion-related)": 2,
    "Non-anginal pain (chest pain, unlikely cardiac)": 3,
    "Asymptomatic (no chest pain)": 4,
}

RESTING_ECG_MAP = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2,
}

# Slope codes are kept as raw categories — see module docstring for why.
SLOPE_MAP = {
    "Slope category 0": 0,
    "Slope category 1": 1,
    "Slope category 2": 2,
    "Slope category 3": 3,
}

YES_NO_MAP = {"No": 0, "Yes": 1}


# ---------------------------------------------------------------------------
# Feature row construction
# ---------------------------------------------------------------------------
def build_feature_row(
    age: int,
    sex: str,
    chest_pain: str,
    resting_bp: int,
    cholesterol: int,
    fasting_blood_sugar_high: str,
    resting_ecg: str,
    max_heart_rate: int,
    exercise_angina: str,
    oldpeak: float,
    slope: str,
    num_vessels: int,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Construct a one-row DataFrame matching the exact feature schema the model
    was trained on. The order of columns in feature_columns is sacred — the
    model will mispredict silently if it's wrong.

    Derived features (age_group, bp_category, cholesterol_band) are computed
    here using the same bin edges as src/features.py.
    """

    # --- Raw clinical features ---
    row = {
        "age": age,
        "sex": SEX_MAP[sex],
        "resting_bp": resting_bp,
        "cholesterol": cholesterol,
        "cholesterol_was_missing": 0,  # Always 0 for user-entered data
        "fasting_blood_sugar": YES_NO_MAP[fasting_blood_sugar_high],
        "max_heart_rate": max_heart_rate,
        "exercise_angina": YES_NO_MAP[exercise_angina],
        "oldpeak": oldpeak,
        "num_vessels": num_vessels,
    }

    # --- One-hot encoded categoricals ---
    cp_value = CHEST_PAIN_MAP[chest_pain]
    for i in range(1, 5):
        row[f"chest_pain_type_{i}"] = 1 if cp_value == i else 0

    ecg_value = RESTING_ECG_MAP[resting_ecg]
    for i in range(0, 3):
        row[f"resting_ecg_{i}"] = 1 if ecg_value == i else 0

    slope_value = SLOPE_MAP[slope]
    for i in range(0, 4):
        row[f"slope_{i}"] = 1 if slope_value == i else 0

    # --- Derived: age_group ---
    if age < 40:
        age_group = "young"
    elif age < 60:
        age_group = "middle"
    else:
        age_group = "senior"
    for ag in ["young", "middle", "senior"]:
        row[f"age_group_{ag}"] = 1 if ag == age_group else 0

    # --- Derived: bp_category ---
    if resting_bp < 120:
        bp_cat = "normal"
    elif resting_bp < 130:
        bp_cat = "elevated"
    elif resting_bp < 180:
        bp_cat = "high"
    else:
        bp_cat = "crisis"
    for bc in ["normal", "elevated", "high", "crisis"]:
        row[f"bp_category_{bc}"] = 1 if bc == bp_cat else 0

    # --- Derived: cholesterol_band ---
    if cholesterol < 200:
        chol_band = "desirable"
    elif cholesterol < 240:
        chol_band = "borderline"
    elif cholesterol < 300:
        chol_band = "high"
    else:
        chol_band = "very_high"
    for cb in ["desirable", "borderline", "high", "very_high"]:
        row[f"cholesterol_band_{cb}"] = 1 if cb == chol_band else 0

    # --- Build DataFrame in exact training column order ---
    df = pd.DataFrame([row])

    # Ensure all training columns are present (defensive — should be a no-op).
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Return in canonical column order so the model sees what it expects.
    return df[feature_columns].astype(float)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(feature_row: pd.DataFrame) -> dict:
    """
    Run a prediction and return both the probability and the SHAP explanation
    for that prediction.

    Returns a dict with:
      - probability     : float in [0, 1], the model's confidence in 'has disease'
      - risk_category   : 'Low' / 'Moderate' / 'High' based on probability
      - shap_values     : signed SHAP values per feature (signed contributions)
      - shap_base_value : the model's baseline (expected output before features)
      - top_drivers     : list of (feature_name, shap_value) for the top 5
                          features pushing toward disease (most positive SHAP)
      - top_protectors  : list of (feature_name, shap_value) for the top 5
                          features pushing away from disease (most negative SHAP)
    """
    bundle = load_model_bundle()
    model = bundle["model"]
    scaler = bundle["scaler"]
    uses_scaled = bundle["uses_scaled_input"]
    feature_cols = bundle["feature_columns"]

    # Scale only if the winning model needs it (LR yes, tree models no).
    if uses_scaled:
        X = pd.DataFrame(
            scaler.transform(feature_row),
            columns=feature_cols,
            index=feature_row.index,
        )
    else:
        X = feature_row

    proba = float(model.predict_proba(X)[0, 1])

    # Risk category bands. These are pragmatic clinical thresholds — the
    # thesis methodology chapter should note these are illustrative, not
    # validated cut-offs against any external clinical benchmark.
    if proba < 0.33:
        category = "Low"
    elif proba < 0.66:
        category = "Moderate"
    else:
        category = "High"

    # SHAP explanation for this specific prediction
    explainer = load_shap_explainer()
    shap_values_full = explainer.shap_values(X)
    shap_row = shap_values_full[0]  # we predicted on one row

    # Sort features by signed SHAP contribution
    contributions = sorted(
        zip(feature_cols, shap_row),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top_drivers = [(name, float(val)) for name, val in contributions[:5]
                   if val > 0]
    top_protectors = [(name, float(val)) for name, val in contributions[-5:]
                      if val < 0]

    return {
        "probability": proba,
        "risk_category": category,
        "shap_values": shap_row,
        "shap_base_value": float(explainer.expected_value),
        "top_drivers": top_drivers,
        "top_protectors": top_protectors,
    }


# ---------------------------------------------------------------------------
# Helper for prettier feature names in UI
# ---------------------------------------------------------------------------
PRETTY_NAMES = {
    "age": "Age",
    "sex": "Sex",
    "resting_bp": "Resting Blood Pressure",
    "cholesterol": "Cholesterol",
    "cholesterol_was_missing": "Cholesterol missing flag",
    "fasting_blood_sugar": "Fasting Blood Sugar > 120",
    "max_heart_rate": "Max Heart Rate",
    "exercise_angina": "Exercise-induced Angina",
    "oldpeak": "ST Depression (oldpeak)",
    "num_vessels": "Diseased Vessel Count",
    "chest_pain_type_1": "Chest Pain: Typical Angina",
    "chest_pain_type_2": "Chest Pain: Atypical Angina",
    "chest_pain_type_3": "Chest Pain: Non-anginal",
    "chest_pain_type_4": "Chest Pain: Asymptomatic",
    "resting_ecg_0": "Resting ECG: Normal",
    "resting_ecg_1": "Resting ECG: ST-T abnormal",
    "resting_ecg_2": "Resting ECG: LVH",
    "slope_0": "ST Slope: Category 0",
    "slope_1": "ST Slope: Category 1",
    "slope_2": "ST Slope: Category 2",
    "slope_3": "ST Slope: Category 3",
    "age_group_young": "Age group: Young (<40)",
    "age_group_middle": "Age group: Middle (40-60)",
    "age_group_senior": "Age group: Senior (60+)",
    "bp_category_normal": "BP: Normal",
    "bp_category_elevated": "BP: Elevated",
    "bp_category_high": "BP: High",
    "bp_category_crisis": "BP: Crisis",
    "cholesterol_band_desirable": "Cholesterol: Desirable",
    "cholesterol_band_borderline": "Cholesterol: Borderline",
    "cholesterol_band_high": "Cholesterol: High",
    "cholesterol_band_very_high": "Cholesterol: Very High",
}


def pretty(feature_name: str) -> str:
    """Return a human-readable label for a feature column name."""
    return PRETTY_NAMES.get(feature_name, feature_name)