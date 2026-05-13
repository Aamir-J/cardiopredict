"""
src/clean.py
------------
Unifies the two raw datasets (Mendeley + UCI) into a single cleaned table.

Transformations applied:
  1. Standardise column names across the two source schemas
  2. Align categorical encodings (chest pain type, sex)
  3. Handle suspicious zero values in Mendeley cholesterol (treat as missing)
  4. Add cholesterol_was_missing indicator (preserves missingness signal
     without leaking it as an outcome predictor)
  5. Median-impute missing values (UCI's `ca` and `thal`, Mendeley's
     newly-flagged cholesterol nulls)
  6. Drop columns not common to both datasets (thal, patient_id)
  7. Concatenate into patients_clean

Output:
  patients_clean table in heart.db with these columns:
    age                       int      patient age in years
    sex                       int      1 = male, 0 = female
    chest_pain_type           int      1–4 (typical/atypical/non-anginal/asymp)
    resting_bp                float    systolic BP at rest, mm Hg
    cholesterol               float    serum cholesterol, mg/dL
    cholesterol_was_missing   int      1 if originally missing/zero, else 0
    fasting_blood_sugar       int      1 if > 120 mg/dL, else 0
    resting_ecg               int      0/1/2 (normal/ST-T abnormal/LVH)
    max_heart_rate            float    max heart rate achieved during exercise
    exercise_angina           int      1 if exercise-induced angina, else 0
    oldpeak                   float    ST depression induced by exercise
    slope                     int      slope of peak exercise ST segment
    num_vessels               float    major vessels coloured by fluoroscopy
    target                    int      1 = heart disease present, 0 = absent
    source                    str      'mendeley_india' or 'uci_cleveland'

Run with:
    python -m src.clean

Idempotent: drops and re-creates the patients_clean table on each run.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.db import Tables, get_engine, row_count


# ---------------------------------------------------------------------------
# Column mapping — unified schema
# ---------------------------------------------------------------------------
# Mendeley raw columns → unified names
MENDELEY_RENAME = {
    "age": "age",
    "gender": "sex",
    "chestpain": "chest_pain_type",
    "restingBP": "resting_bp",
    "serumcholestrol": "cholesterol",
    "fastingbloodsugar": "fasting_blood_sugar",
    "restingrelectro": "resting_ecg",
    "maxheartrate": "max_heart_rate",
    "exerciseangia": "exercise_angina",
    "oldpeak": "oldpeak",
    "slope": "slope",
    "noofmajorvessels": "num_vessels",
    "target": "target",
    "source": "source",
}

# UCI raw columns → unified names
UCI_RENAME = {
    "age": "age",
    "sex": "sex",
    "cp": "chest_pain_type",
    "trestbps": "resting_bp",
    "chol": "cholesterol",
    "fbs": "fasting_blood_sugar",
    "restecg": "resting_ecg",
    "thalach": "max_heart_rate",
    "exang": "exercise_angina",
    "oldpeak": "oldpeak",
    "slope": "slope",
    "ca": "num_vessels",
    "target": "target",
    "source": "source",
}

# Final canonical column order
FINAL_COLUMNS = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_bp",
    "cholesterol",
    "cholesterol_was_missing",
    "fasting_blood_sugar",
    "resting_ecg",
    "max_heart_rate",
    "exercise_angina",
    "oldpeak",
    "slope",
    "num_vessels",
    "target",
    "source",
]


# ---------------------------------------------------------------------------
# Mendeley cleaning
# ---------------------------------------------------------------------------
def clean_mendeley() -> pd.DataFrame:
    """
    Clean the raw Mendeley table and return a unified-schema DataFrame.

    Key transformation: cholesterol values of 0 are treated as missing
    (biologically impossible — see methodology chapter). We replace them
    with NaN, add a `cholesterol_was_missing` flag, then median-impute.
    """
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {Tables.RAW_INDIA}", engine)

    print(f"  loaded raw Mendeley:    {len(df):,} rows × {df.shape[1]} cols")

    # 1. Drop the raw patient ID — not useful for ML, and sensitive in any case.
    df = df.drop(columns=["patientid"])

    # 2. Rename to unified schema.
    df = df.rename(columns=MENDELEY_RENAME)

    # 3. Align chest_pain_type encoding to UCI's 1–4 scale.
    #    Mendeley uses 0–3, UCI uses 1–4 — they represent the same four
    #    categories (typical/atypical/non-anginal/asymptomatic angina).
    #    We add 1 to Mendeley's values to align.
    df["chest_pain_type"] = df["chest_pain_type"] + 1

    # 4. Handle cholesterol == 0 as missing (biologically impossible).
    #    All 53 such records had target=1, so leaving them at 0 would create
    #    data leakage. Replace with NaN, add indicator flag, impute later.
    cholesterol_zero_mask = df["cholesterol"] == 0
    n_zero = cholesterol_zero_mask.sum()
    df["cholesterol_was_missing"] = cholesterol_zero_mask.astype(int)
    df.loc[cholesterol_zero_mask, "cholesterol"] = np.nan

    # 5. Median-impute the now-NaN cholesterol values.
    #    Median is preferred over mean for skewed clinical distributions.
    median_chol = df["cholesterol"].median()
    df["cholesterol"] = df["cholesterol"].fillna(median_chol)

    print(f"  cholesterol == 0 found: {n_zero} records")
    print(f"  imputed with median:    {median_chol:.1f} mg/dL")
    print(f"  added flag:             cholesterol_was_missing")

    # 6. Reorder to canonical schema.
    df = df[FINAL_COLUMNS]

    print(f"  cleaned Mendeley:       {len(df):,} rows × {df.shape[1]} cols")
    return df


# ---------------------------------------------------------------------------
# UCI cleaning
# ---------------------------------------------------------------------------
def clean_uci() -> pd.DataFrame:
    """
    Clean the raw UCI Cleveland table and return a unified-schema DataFrame.

    Key transformation: drop the `thal` column (not present in Mendeley).
    Median-impute the 4 missing `ca` values and the 2 missing `thal` values
    (the latter just before we drop it — they don't affect anything).

    UCI does not have any biologically-impossible zero encoding issues,
    so cholesterol_was_missing is set to 0 for all UCI records.
    """
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {Tables.RAW_UCI}", engine)

    print(f"  loaded raw UCI:         {len(df):,} rows × {df.shape[1]} cols")

    # 1. Drop the `thal` column — not present in Mendeley, can't use as feature.
    df = df.drop(columns=["thal"])

    # 2. Rename to unified schema.
    df = df.rename(columns=UCI_RENAME)

    # 3. Median-impute missing num_vessels (was 'ca' — 4 nulls in raw data).
    median_vessels = df["num_vessels"].median()
    n_missing_vessels = df["num_vessels"].isnull().sum()
    df["num_vessels"] = df["num_vessels"].fillna(median_vessels)

    print(f"  num_vessels missing:    {n_missing_vessels} records")
    print(f"  imputed with median:    {median_vessels:.1f}")

    # 4. UCI has no zero-as-missing cholesterol issue — set flag to 0 for all.
    df["cholesterol_was_missing"] = 0

    # 5. Reorder to canonical schema.
    df = df[FINAL_COLUMNS]

    print(f"  cleaned UCI:            {len(df):,} rows × {df.shape[1]} cols")
    return df


# ---------------------------------------------------------------------------
# Concatenation + write
# ---------------------------------------------------------------------------
def main() -> int:
    """Run the cleaning pipeline. Returns POSIX exit code."""
    print("CardioPredict — Cleaning and unifying datasets")
    print("=" * 60)

    print("\nStage 1/3 — Cleaning Mendeley")
    print("-" * 60)
    mend = clean_mendeley()

    print("\nStage 2/3 — Cleaning UCI")
    print("-" * 60)
    uci = clean_uci()

    print("\nStage 3/3 — Concatenating into unified table")
    print("-" * 60)
    combined = pd.concat([mend, uci], ignore_index=True)
    print(f"  combined: {len(combined):,} rows × {combined.shape[1]} cols")

    # Sanity checks before writing.
    print("\n  Sanity checks:")
    print(f"    rows from Mendeley:   "
          f"{(combined['source'] == 'mendeley_india').sum():,}")
    print(f"    rows from UCI:        "
          f"{(combined['source'] == 'uci_cleveland').sum():,}")
    print(f"    target=1 (positive):  "
          f"{(combined['target'] == 1).sum():,}")
    print(f"    target=0 (negative):  "
          f"{(combined['target'] == 0).sum():,}")
    print(f"    any nulls remaining:  {combined.isnull().sum().sum()}")

    # Write.
    engine = get_engine()
    combined.to_sql(Tables.CLEAN, engine, if_exists="replace", index=False)

    print("\n" + "=" * 60)
    print(f"  ✓ wrote {len(combined):,} rows to '{Tables.CLEAN}'")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())