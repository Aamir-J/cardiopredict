"""
src/db.py
---------
Database layer for CardioPredict.

Single source of truth for:
  - the SQLite database location (data/heart.db)
  - the SQLAlchemy engine
  - the table names used across the project
"""

from __future__ import annotations

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# Paths — resolved relative to the project root (parent of src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "heart.db"

# Ensure required directories exist (idempotent).
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Table names — single source of truth. Use these constants, not strings.
# ---------------------------------------------------------------------------
class Tables:
    """Canonical table names for the project."""

    # Raw ingest tables — one per source dataset, untouched after ingest.
    RAW_INDIA = "patients_raw_india"           # Mendeley Indian hospital
    RAW_KAGGLE_INDIA = "patients_raw_kaggle"   # Kaggle India dataset
    RAW_UCI = "patients_raw_uci"               # UCI Cleveland (benchmark)

    # Cleaned, unified table — output of clean.py, input to features.py.
    CLEAN = "patients_clean"

    # Feature-engineered, model-ready table — output of features.py.
    FEATURES = "patients_features"

    # Per-model performance metrics — written by train.py, read by dashboard.
    MODEL_METRICS = "model_metrics"


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------
def get_engine(read_only: bool = False) -> Engine:
    """
    Return a SQLAlchemy engine for the project SQLite database.

    Parameters
    ----------
    read_only : bool
        If True, opens the database in read-only mode. Used by the chatbot's
        text-to-SQL tool to ensure no write access from LLM-generated queries.
    """
    if read_only:
        url = f"sqlite:///file:{DB_PATH}?mode=ro&uri=true"
        return create_engine(url, connect_args={"uri": True})

    return create_engine(f"sqlite:///{DB_PATH}")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def list_tables() -> list[str]:
    """Return the list of tables currently in the database."""
    from sqlalchemy import inspect
    return inspect(get_engine()).get_table_names()


def table_exists(name: str) -> bool:
    """Check whether a given table is present."""
    return name in list_tables()


def row_count(table: str) -> int:
    """Return the row count of a given table, or 0 if it doesn't exist."""
    if not table_exists(table):
        return 0
    from sqlalchemy import text
    with get_engine().connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
        return result.scalar_one()


if __name__ == "__main__":
    # Smoke test — print database state.
    print(f"Database path: {DB_PATH}")
    print(f"Database exists: {DB_PATH.exists()}")
    if DB_PATH.exists():
        tables = list_tables()
        print(f"Tables ({len(tables)}):")
        for t in tables:
            print(f"  {t}: {row_count(t)} rows")
    else:
        print("Database has not been created yet — run `python -m src.ingest` first.")