"""
src/train.py
------------
Trains three classifiers on the engineered feature table and selects the best.

Models compared:
  - Logistic Regression  : interpretable linear baseline
  - Random Forest        : non-linear ensemble, robust to feature scaling
  - XGBoost              : gradient-boosted trees, current SOTA on tabular

Primary metric: RECALL (sensitivity).
  In medical screening, a false negative — predicting healthy when the
  patient has disease — has higher clinical and economic cost than a false
  positive. Recall = TP / (TP + FN) directly captures this trade-off.
  See methodology chapter for full justification.

Secondary metrics tracked: accuracy, AUC-ROC, F1, precision.

Output:
  - models/heart_model.pkl       : the winning trained model + scaler
  - models/feature_columns.json  : the exact column order used at training
                                   (the dashboard must use the same order)
  - model_metrics table          : per-model results, written to SQLite

Notes on methodology choices:
  - Train/test split: 80/20, stratified by target, random_state=42 for
    reproducibility. The methodology chapter documents the seed.
  - Scaler: StandardScaler fitted on the TRAINING SPLIT ONLY, then applied
    to test. Fitting on the full dataset would leak test-set information.
  - The scaler is persisted alongside the model so the dashboard can apply
    identical transformations to user-entered patient data at inference.
  - 5-fold cross-validation reported for stability assessment.

Run with:
    python -m src.train
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.db import PROJECT_ROOT, Tables, get_engine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "heart_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURES_PATH = MODELS_DIR / "feature_columns.json"

# Columns to exclude from features (target is the label, source is metadata).
NON_FEATURE_COLS = ["target", "source"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def _load_features() -> tuple[pd.DataFrame, pd.Series]:
    """Load the engineered feature table; return X, y."""
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {Tables.FEATURES}", engine)

    # Boolean columns from one-hot encoding sometimes come back as bool/string
    # from SQLite; force everything to numeric.
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)

    return X, y


def _evaluate_model(
    name: str,
    model,
    X_train, X_test, y_train, y_test,
    cv_scoring: str = "recall",
) -> dict:
    """
    Fit a model, evaluate on the held-out test set, and run cross-validation.

    Returns a dict of metrics ready to be written to model_metrics table.
    """
    # Fit
    model.fit(X_train, y_train)

    # Test-set predictions
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred
    )

    # Metrics
    metrics = {
        "model_name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    # 5-fold CV for stability — reported on training set only.
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=cv_scoring)
    metrics["cv_recall_mean"] = cv_scores.mean()
    metrics["cv_recall_std"] = cv_scores.std()

    # Confusion matrix (stored as JSON string for SQLite).
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = json.dumps(cm.tolist())

    # Timestamp.
    metrics["trained_at"] = datetime.now().isoformat(timespec="seconds")

    return metrics


def _print_metrics_row(m: dict) -> None:
    """Pretty-print a metrics dict."""
    print(f"  {m['model_name']:25s}")
    print(f"    accuracy       : {m['accuracy']:.4f}")
    print(f"    AUC-ROC        : {m['auc_roc']:.4f}")
    print(f"    recall (PRIMARY): {m['recall']:.4f}")
    print(f"    precision      : {m['precision']:.4f}")
    print(f"    F1             : {m['f1']:.4f}")
    print(f"    CV recall      : {m['cv_recall_mean']:.4f} "
          f"± {m['cv_recall_std']:.4f}")
    cm = json.loads(m["confusion_matrix"])
    print(f"    confusion matrix: {cm}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> int:
    print("CardioPredict — Model training")
    _print_section("Load features and split")

    X, y = _load_features()
    feature_cols = list(X.columns)
    print(f"  features:  {len(feature_cols)} columns")
    print(f"  samples:   {len(X):,}")
    print(f"  positive:  {(y == 1).sum():,} ({(y == 1).mean():.1%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"  train:     {len(X_train):,}")
    print(f"  test:      {len(X_test):,}")

    # Fit scaler on training split only (no leakage).
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )

    _print_section("Train and evaluate three models")

    # Each model gets its own configuration. class_weight='balanced' on LR/RF
    # tilts the loss to penalise false negatives — appropriate given recall
    # is our primary metric.
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }

    results = []
    for name, model in models.items():
        # LR benefits from scaled features; tree models don't need scaling
        # but it doesn't hurt either. Use scaled inputs for LR, raw for trees.
        if name == "LogisticRegression":
            metrics = _evaluate_model(name, model,
                                      X_train_scaled, X_test_scaled,
                                      y_train, y_test)
        else:
            metrics = _evaluate_model(name, model,
                                      X_train, X_test,
                                      y_train, y_test)
        _print_metrics_row(metrics)
        print()
        results.append((name, model, metrics))

    _print_section("Select best model by primary metric (recall)")

    results.sort(key=lambda r: r[2]["recall"], reverse=True)
    best_name, best_model, best_metrics = results[0]
    print(f"  Winner: {best_name}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    print(f"  AUC-ROC: {best_metrics['auc_roc']:.4f}")

    _print_section("Persist artefacts")

    # Save model + scaler bundled together in a dict for easy loading.
    joblib.dump(
        {
            "model": best_model,
            "scaler": scaler,
            "model_name": best_name,
            "feature_columns": feature_cols,
            "uses_scaled_input": (best_name == "LogisticRegression"),
        },
        MODEL_PATH,
    )
    print(f"  ✓ model     → {MODEL_PATH}")

    # Also save feature column order separately as JSON for the dashboard.
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  ✓ features  → {FEATURES_PATH}")

    # Write metrics for all models to model_metrics table.
    metrics_df = pd.DataFrame([m for _, _, m in results])
    metrics_df["is_winner"] = (metrics_df["model_name"] == best_name).astype(int)
    engine = get_engine()
    metrics_df.to_sql(Tables.MODEL_METRICS, engine,
                      if_exists="replace", index=False)
    print(f"  ✓ metrics   → table '{Tables.MODEL_METRICS}'")

    _print_section("Summary")
    print(f"  {len(results)} models trained, {best_name} selected")
    print(f"  Best recall: {best_metrics['recall']:.4f}")
    print(f"  Best AUC:    {best_metrics['auc_roc']:.4f}")
    print(f"  Best F1:     {best_metrics['f1']:.4f}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())