"""
src/ingest.py
-------------
Pulls the source datasets into SQLite, into separate raw tables.

  - patients_raw_india  : Mendeley Indian hospital CVD (1,000 patients)
  - patients_raw_uci    : UCI Cleveland heart disease (303 patients, benchmark)

Mendeley is downloaded manually (login-free, requires a download click).
UCI Cleveland is fetched programmatically via the `ucimlrepo` package.

NOTE — Kaggle India dataset (data/raw/kaggle_india_heart.csv):
  This dataset was evaluated and EXCLUDED from training based on a synthetic-
  data audit. Findings: zero missing values across 26 columns × 10,000 rows;
  uniform distributions with means at exact range midpoints; biologically
  inconsistent lipid panel (LDL + HDL + Trig/5 does not approximate Total
  Cholesterol). The dataset description does not name a clinical source.
  These markers are consistent with synthetic generation via numpy.random.
  Decision: retained on disk for methodology-chapter audit reference only;
  not loaded into the database, not used for training.

Run with:
    python -m src.ingest

Idempotent: re-running drops and re-creates the raw tables, so it's safe to
run repeatedly during development.
"""

from __future__ import annotations

import sys

import pandas as pd

from src.db import RAW_DIR, Tables, get_engine, row_count

# ---------------------------------------------------------------------------
# Expected file path for the manually-downloaded Mendeley dataset
# ---------------------------------------------------------------------------
MENDELEY_CSV = RAW_DIR / "mendeley_cardiovascular.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _print_section(title: str) -> None:
    """Print a visible section header."""
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def _write_to_db(df: pd.DataFrame, table: str) -> None:
    """Write a DataFrame to the named SQLite table, replacing if exists."""
    engine = get_engine()
    df.to_sql(table, engine, if_exists="replace", index=False)
    print(f"  → wrote {len(df):,} rows × {df.shape[1]} cols to '{table}'")


# ---------------------------------------------------------------------------
# Mendeley — manual download required
# ---------------------------------------------------------------------------
def ingest_mendeley() -> bool:
    """Load Mendeley Indian hospital CVD dataset. Returns True on success."""
    _print_section("1/2  Mendeley Indian Hospital CVD Dataset")

    if not MENDELEY_CSV.exists():
        print(f"  ✗ File not found: {MENDELEY_CSV}")
        print()
        print("  To download:")
        print("    1. Visit https://data.mendeley.com/datasets/dzz48mvjht/1")
        print("    2. Click 'Download All' (no login required)")
        print("    3. Extract the ZIP and rename the CSV to:")
        print(f"       {MENDELEY_CSV.name}")
        print(f"    4. Place it in: {RAW_DIR}/")
        return False

    df = pd.read_csv(MENDELEY_CSV)
    print(f"  loaded {len(df):,} rows × {df.shape[1]} columns")
    print(f"  columns: {list(df.columns)}")

    df["source"] = "mendeley_india"
    _write_to_db(df, Tables.RAW_INDIA)
    return True


# ---------------------------------------------------------------------------
# UCI Cleveland — programmatic fetch
# ---------------------------------------------------------------------------
def ingest_uci() -> bool:
    """Fetch UCI Cleveland Heart Disease dataset via ucimlrepo."""
    _print_section("2/2  UCI Cleveland Heart Disease Dataset (benchmark)")

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("  ✗ ucimlrepo not installed. Run: pip install ucimlrepo")
        return False

    try:
        print("  fetching from UCI ML Repository (id=45) ...")
        heart = fetch_ucirepo(id=45)
    except Exception as exc:
        print(f"  ✗ fetch failed: {exc}")
        print("  Check your internet connection, or download manually from:")
        print("  https://archive.ics.uci.edu/dataset/45/heart+disease")
        return False

    # ucimlrepo returns features and targets separately; rejoin them.
    X = heart.data.features
    y = heart.data.targets
    df = pd.concat([X, y], axis=1)

    # The 'num' column is 0..4 (severity); binarise to match the other dataset:
    # 0 = no disease, 1+ = disease present.
    if "num" in df.columns:
        df["target"] = (df["num"] > 0).astype(int)
        df = df.drop(columns=["num"])

    df["source"] = "uci_cleveland"

    print(f"  loaded {len(df):,} rows × {df.shape[1]} columns")
    print(f"  columns: {list(df.columns)}")

    _write_to_db(df, Tables.RAW_UCI)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    """Run all ingest steps. Returns POSIX exit code."""
    print("CardioPredict — Ingesting raw datasets into SQLite")
    print(f"Database: {get_engine().url}")

    results = {
        "mendeley": ingest_mendeley(),
        "uci":      ingest_uci(),
    }

    _print_section("Summary")
    for name, ok in results.items():
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {name}")

    print("\nCurrent table state:")
    for table in (Tables.RAW_INDIA, Tables.RAW_UCI):
        print(f"  {table:30s} {row_count(table):>6,} rows")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())