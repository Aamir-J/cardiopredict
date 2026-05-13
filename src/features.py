"""
src/features.py
---------------
Feature engineering for the CardioPredict model.

Reads the unified `patients_clean` table and produces a model-ready
`patients_features` table with:

  - Derived clinical features (age groups, BP categories, cholesterol bands)
    that improve interpretability for SHAP and the chatbot's explanations.

  - One-hot encoding of categorical features (chest_pain_type, resting_ecg,
    slope) so they can be consumed by linear models without imposing a false
    ordinal relationship.

  - Original numerical features kept unscaled. Scaling happens in train.py
    using a StandardScaler that's fitted on the training split only and
    persisted alongside the model — this prevents test-set leakage.

Why derived features?
  Bare numerical inputs work for the model but make SHAP plots and the
  chatbot's explanations less intelligible. A SHAP value attributed to
  "age=58" is technically correct but harder to communicate than "patient
  is in the senior age group". Derived features sit alongside the raw
  numerics, giving us interpretability without losing predictive power.

Run with:
    python -m src.features

Idempotent: drops and re-creates patients_features on each run.
"""

from __future__ import annotations

import sys

import pandas as pd

from src.db import Tables, get_engine, row_count


# ---------------------------------------------------------------------------
# Derived clinical features
# ---------------------------------------------------------------------------
def _add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket age into three clinically meaningful groups.

    young  : 0–40    (low baseline cardiac risk)
    middle : 40–60   (rising risk; primary screening target)
    senior : 60+     (high baseline risk; established disease likely)

    Bins are right-exclusive so 40 falls in 'middle', 60 falls in 'senior'.
    """
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 60, 200],
        labels=["young", "middle", "senior"],
        right=False,
    ).astype(str)
    return df


def _add_bp_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify resting BP using American Heart Association 2017 thresholds.

    normal     : < 120
    elevated   : 120–129
    high       : 130–179
    crisis     : 180+

    These are systolic-only categories — the unified dataset doesn't have
    diastolic, so we use systolic alone, which is the AHA's primary cutoff.
    """
    df = df.copy()
    df["bp_category"] = pd.cut(
        df["resting_bp"],
        bins=[0, 120, 130, 180, 500],
        labels=["normal", "elevated", "high", "crisis"],
        right=False,
    ).astype(str)
    return df


def _add_cholesterol_band(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify total cholesterol per US NCEP ATP III guidelines.

    desirable      : < 200
    borderline     : 200–239
    high           : 240–299
    very_high      : 300+

    The very_high band is added because Mendeley's data skews toward
    cardiology referrals — many patients exceed the standard 'high'
    threshold of 240, and lumping them together loses signal.
    """
    df = df.copy()
    df["cholesterol_band"] = pd.cut(
        df["cholesterol"],
        bins=[0, 200, 240, 300, 1000],
        labels=["desirable", "borderline", "high", "very_high"],
        right=False,
    ).astype(str)
    return df


# ---------------------------------------------------------------------------
# One-hot encoding for nominal categorical variables
# ---------------------------------------------------------------------------
def _one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the truly-categorical features.

    Why one-hot, not ordinal:
      - chest_pain_type has 4 categories (typical/atypical/non-anginal/asymp)
        but they're not ordered — asymptomatic isn't 'more' than typical.
      - resting_ecg has 3 categories (normal/ST-T abnormal/LVH) — also nominal.
      - slope has 3 categories (upsloping/flat/downsloping) — nominal.

    age_group, bp_category, cholesterol_band ARE ordinal (young < middle < senior)
    but we one-hot them too for consistency, since most ML libraries treat
    string columns as categorical anyway. Tree-based models handle either fine.
    """
    df = df.copy()

    categorical_cols = [
        "chest_pain_type",
        "resting_ecg",
        "slope",
        "age_group",
        "bp_category",
        "cholesterol_band",
    ]

    # Use prefix to keep generated column names readable.
    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols,
                        dtype=int)

    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main() -> int:
    """Run the feature engineering pipeline. Returns POSIX exit code."""
    print("CardioPredict — Feature engineering")
    print("=" * 60)

    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {Tables.CLEAN}", engine)
    print(f"\n  loaded patients_clean: {len(df):,} rows × {df.shape[1]} cols")

    print("\n  Stage 1 — Derived clinical features")
    print("  " + "-" * 56)
    df = _add_age_group(df)
    print(f"    + age_group       (young / middle / senior)")
    df = _add_bp_category(df)
    print(f"    + bp_category     (normal / elevated / high / crisis)")
    df = _add_cholesterol_band(df)
    print(f"    + cholesterol_band (desirable / borderline / high / very_high)")

    print(f"\n  After derived features: {len(df):,} rows × {df.shape[1]} cols")

    print("\n  Stage 2 — One-hot encoding categoricals")
    print("  " + "-" * 56)
    cols_before = df.shape[1]
    df = _one_hot_encode(df)
    cols_after = df.shape[1]
    print(f"    columns: {cols_before} → {cols_after} "
          f"(+{cols_after - cols_before} encoded columns)")

    print(f"\n  Final shape:           {len(df):,} rows × {df.shape[1]} cols")

    # Quick sanity check — verify target distribution preserved.
    print("\n  Sanity checks:")
    print(f"    target=1 (positive):  {(df['target'] == 1).sum():,}")
    print(f"    target=0 (negative):  {(df['target'] == 0).sum():,}")
    print(f"    Mendeley records:     "
          f"{(df['source'] == 'mendeley_india').sum():,}")
    print(f"    UCI records:          "
          f"{(df['source'] == 'uci_cleveland').sum():,}")
    print(f"    any nulls remaining:  {df.isnull().sum().sum()}")

    # Write to database.
    df.to_sql(Tables.FEATURES, engine, if_exists="replace", index=False)

    print("\n" + "=" * 60)
    print(f"  ✓ wrote {len(df):,} rows to '{Tables.FEATURES}'")
    print("=" * 60)

    # Print final column list for visibility.
    print("\nFinal feature columns:")
    for col in df.columns:
        print(f"  {col}")

    return 0


if __name__ == "__main__":
    sys.exit(main())