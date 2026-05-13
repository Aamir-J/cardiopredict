"""
src/explain.py
--------------
SHAP explainability for the trained XGBoost model.

Generates two outputs that feed both the thesis and the dashboard:

  1. Global feature importance (beeswarm + bar plot)
     "Across all patients, which features drive risk most?"

  2. Local explanation utilities (function used by the dashboard)
     "For THIS specific patient, why does the model predict what it does?"

SHAP (SHapley Additive exPlanations, Lundberg & Lee, NeurIPS 2017) computes
each feature's marginal contribution to a prediction using game-theoretic
principles. Unlike feature importance from .feature_importances_, SHAP
values are signed (positive = pushed toward 'has disease', negative = pushed
toward 'no disease') and additive (sum of SHAP values + baseline = model
output for that patient).

Outputs:
  - models/shap_summary_bar.png      : global importance bar chart
  - models/shap_summary_beeswarm.png : global beeswarm plot (richer view)
  - models/shap_explainer.pkl        : the SHAP TreeExplainer, persisted
                                        for the dashboard's per-patient
                                        explanations.

Run with:
    python -m src.explain
"""

from __future__ import annotations

import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.db import PROJECT_ROOT, Tables, get_engine

MODELS_DIR = PROJECT_ROOT / "models"
SHAP_BAR_PATH = MODELS_DIR / "shap_summary_bar.png"
SHAP_BEESWARM_PATH = MODELS_DIR / "shap_summary_beeswarm.png"
SHAP_EXPLAINER_PATH = MODELS_DIR / "shap_explainer.pkl"

NON_FEATURE_COLS = ["target", "source"]


def main() -> int:
    print("CardioPredict — SHAP explainability")
    print("=" * 60)

    # 1. Load trained model bundle
    print("\n  Loading trained model ...")
    bundle = joblib.load(MODELS_DIR / "heart_model.pkl")
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    print(f"    model:    {bundle['model_name']}")
    print(f"    features: {len(feature_cols)}")

    # 2. Load the engineered features (we'll explain on the full set so the
    #    plots reflect the whole population, not just the test split).
    print("\n  Loading patients_features ...")
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {Tables.FEATURES}", engine)
    X = df[feature_cols].astype(float)
    print(f"    samples:  {len(X):,}")

    # 3. Build the TreeExplainer and compute SHAP values
    print("\n  Computing SHAP values (this can take 30–60 seconds) ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print(f"    SHAP values shape: {shap_values.shape}")

    # 4. Global summary — bar plot (mean absolute SHAP per feature)
    print("\n  Generating global summary plots ...")
    plt.figure()
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_cols,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(SHAP_BAR_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    ✓ bar plot       → {SHAP_BAR_PATH}")

    # 5. Beeswarm — richer view showing feature distributions
    plt.figure()
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_cols,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(SHAP_BEESWARM_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    ✓ beeswarm plot  → {SHAP_BEESWARM_PATH}")

    # 6. Persist the explainer so the dashboard can reuse it for per-patient
    #    explanations without re-fitting.
    joblib.dump(explainer, SHAP_EXPLAINER_PATH)
    print(f"    ✓ explainer obj  → {SHAP_EXPLAINER_PATH}")

    # 7. Print top features by mean absolute SHAP for a quick sanity check
    print("\n  Top 10 features by mean |SHAP|:")
    print("  " + "-" * 56)
    mean_abs = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    for _, row in mean_abs.head(10).iterrows():
        print(f"    {row['feature']:30s}  {row['mean_abs_shap']:.4f}")

    print("\n" + "=" * 60)
    print(f"  ✓ SHAP analysis complete")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())