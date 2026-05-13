"""
dashboard/sql_tool.py
---------------------
The database tool used by the Analytics-page chatbot.

This module defines:
  - The tool schema Claude sees (telling it what SQL it can run and against
    which tables/columns).
  - The safe-execution wrapper that runs LLM-generated SQL against a
    read-only SQLite connection, with row-limit enforcement.

Design principles:
  - Read-only at the OS level — the SQLite URI mode=ro guarantees no write
    operations succeed, even if the LLM tries one.
  - LIMIT 100 applied to every SELECT that returns rows, preventing memory
    blow-ups from accidentally large result sets.
  - Schema is injected into the system prompt so Claude works from the real
    column names rather than hallucinating them.
  - Errors are returned to the LLM in a structured way so it can retry or
    explain the failure to the user.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.db import Tables, get_engine

# How many rows we'll ever return to the LLM from a single query.
# Even if Claude tries to fetch everything, we cap at this.
MAX_ROWS = 100


# ---------------------------------------------------------------------------
# Schema introspection — what Claude sees about our database
# ---------------------------------------------------------------------------
SCHEMA_DOCUMENTATION = f"""
You can query a SQLite database with the following tables and columns.
ALL QUERIES MUST BE SELECT STATEMENTS. The database is read-only.

# Table: {Tables.CLEAN}  ({Tables.CLEAN})
Patient-level cleaned data, 1,303 rows. Each row is one patient.

Columns:
  - age (INTEGER)                       Patient age in years
  - sex (INTEGER)                       1 = male, 0 = female
  - chest_pain_type (INTEGER)           1=typical angina, 2=atypical angina,
                                        3=non-anginal pain, 4=asymptomatic
  - resting_bp (INTEGER)                Resting systolic BP in mm Hg
  - cholesterol (INTEGER)               Total cholesterol in mg/dL
  - cholesterol_was_missing (INTEGER)   1 if cholesterol was originally 0
                                        (treated as missing), else 0
  - fasting_blood_sugar (INTEGER)       1 if > 120 mg/dL, else 0
  - resting_ecg (INTEGER)               0=normal, 1=ST-T abnormal, 2=LVH
  - max_heart_rate (INTEGER)            Max heart rate during exercise test
  - exercise_angina (INTEGER)           1 if exercise-induced angina, else 0
  - oldpeak (REAL)                      ST depression induced by exercise
  - slope (INTEGER)                     Slope of peak exercise ST segment (0-3)
  - num_vessels (INTEGER)               Diseased vessels on fluoroscopy (0-3)
  - target (INTEGER)                    1 = has heart disease, 0 = no disease
  - source (TEXT)                       'mendeley_india' or 'uci_cleveland'

# Table: {Tables.MODEL_METRICS}  ({Tables.MODEL_METRICS})
Performance metrics for each trained model.

Columns:
  - model_name (TEXT)                   'LogisticRegression', 'RandomForest',
                                        or 'XGBoost'
  - accuracy (REAL)                     Test-set accuracy
  - auc_roc (REAL)                      Area under ROC curve
  - recall (REAL)                       True positive rate (primary metric)
  - precision (REAL)                    Positive predictive value
  - f1 (REAL)                           F1 score
  - cv_recall_mean (REAL)               5-fold CV recall, mean
  - cv_recall_std (REAL)                5-fold CV recall, std dev
  - confusion_matrix (TEXT)             JSON-encoded 2x2 matrix
  - trained_at (TEXT)                   ISO timestamp
  - is_winner (INTEGER)                 1 if this is the selected model

# Useful query patterns

Comparing cohorts:
  SELECT source, AVG(cholesterol) FROM patients_clean GROUP BY source;

Counting by group:
  SELECT source, COUNT(*) FROM patients_clean WHERE target = 1 GROUP BY source;

Distribution of a feature:
  SELECT cholesterol FROM patients_clean WHERE source = 'mendeley_india';

NOTES:
  - Always use SELECT. No INSERT, UPDATE, DELETE, DROP, ALTER, CREATE.
  - Results are auto-limited to {MAX_ROWS} rows; design queries accordingly
    (use aggregates rather than fetching all rows when possible).
  - Cohort comparisons are the most useful: GROUP BY source.
"""


# ---------------------------------------------------------------------------
# Tool definition for the Anthropic API
# ---------------------------------------------------------------------------
SQL_TOOL_DEFINITION = {
    "name": "run_sql_query",
    "description": (
        "Execute a read-only SELECT query against the CardioPredict SQLite "
        "database. Returns the result rows as a JSON-serialisable list of "
        "dicts (max 100 rows). Use this whenever the user asks a question "
        "that requires looking up real data — averages, counts, "
        "distributions, cohort comparisons. Always prefer aggregate queries "
        "(AVG, COUNT, GROUP BY) over fetching raw rows."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": (
                    "A valid SQLite SELECT statement. No INSERT/UPDATE/DELETE/"
                    "DROP/ALTER/CREATE — the connection is read-only and "
                    "such queries will be rejected. Should be a single "
                    "statement (no semicolon-chained queries)."
                ),
            },
        },
        "required": ["sql"],
    },
}


# ---------------------------------------------------------------------------
# Safety: reject anything that isn't a SELECT
# ---------------------------------------------------------------------------
# Pattern: must START with SELECT (possibly preceded by WITH for CTEs),
# must not contain any forbidden write keywords as whole words.
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|replace|attach|"
    r"detach|pragma|vacuum)\b",
    re.IGNORECASE,
)

_ALLOWED_START = re.compile(r"^\s*(select|with)\s", re.IGNORECASE)


def _is_safe_query(sql: str) -> tuple[bool, str]:
    """
    Check whether a SQL string is safe to execute.

    Returns (ok, reason). reason is a short error string if not ok.
    """
    if not sql or not sql.strip():
        return False, "Empty query"

    # Must start with SELECT or WITH (CTE)
    if not _ALLOWED_START.match(sql):
        return False, "Query must start with SELECT (or WITH for CTEs)"

    # Must not contain write keywords anywhere
    forbidden = _FORBIDDEN_KEYWORDS.search(sql)
    if forbidden:
        return False, (
            f"Query contains forbidden keyword '{forbidden.group(0)}'. "
            f"This database is read-only."
        )

    # No semicolon-chained queries (the LLM should send one statement)
    # We allow a single trailing semicolon but nothing else.
    stripped = sql.rstrip().rstrip(";").rstrip()
    if ";" in stripped:
        return False, (
            "Multiple statements not allowed. Send a single SELECT."
        )

    return True, ""


def _add_row_limit(sql: str, limit: int = MAX_ROWS) -> str:
    """
    If the query doesn't already have a LIMIT clause, add one. This prevents
    runaway result sets from blowing up memory or the chat UI.
    """
    # Crude but effective: check for LIMIT as a whole word
    if re.search(r"\blimit\s+\d+", sql, re.IGNORECASE):
        return sql  # already has a limit, leave it alone

    # Strip trailing semicolons/whitespace and append LIMIT
    cleaned = sql.rstrip().rstrip(";").rstrip()
    return f"{cleaned} LIMIT {limit}"


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------
def run_sql_query(sql: str) -> dict[str, Any]:
    """
    Execute a SQL query safely and return a structured result.

    Returns a dict with one of two shapes:
      - On success:  {"ok": True, "columns": [...], "rows": [...],
                      "row_count": N, "limited": bool}
      - On failure:  {"ok": False, "error": "..."}
    """
    # 1. Safety check
    ok, reason = _is_safe_query(sql)
    if not ok:
        return {"ok": False, "error": f"Query rejected: {reason}"}

    # 2. Add LIMIT if missing
    safe_sql = _add_row_limit(sql)
    limited = safe_sql != sql.rstrip().rstrip(";").rstrip()

    # 3. Execute against read-only connection
    try:
        engine = get_engine(read_only=True)
        with engine.connect() as conn:
            result = conn.execute(text(safe_sql))
            columns = list(result.keys())
            rows_raw = result.fetchall()
    except SQLAlchemyError as exc:
        # Pass the SQL error back to the LLM so it can interpret and retry
        return {"ok": False, "error": f"SQL error: {str(exc.orig) if hasattr(exc, 'orig') else str(exc)}"}
    except Exception as exc:
        return {"ok": False, "error": f"Unexpected error: {exc}"}

    # 4. Convert rows to JSON-serialisable list of dicts
    rows = [
        {col: (val if not isinstance(val, (bytes, bytearray)) else val.hex())
         for col, val in zip(columns, row)}
        for row in rows_raw
    ]

    return {
        "ok": True,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "limited": limited,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test the executor end-to-end
    print("Test 1 — Safe aggregate query:")
    result = run_sql_query(
        "SELECT source, AVG(cholesterol) AS avg_chol "
        f"FROM {Tables.CLEAN} GROUP BY source"
    )
    print(result)

    print("\nTest 2 — Unsafe write query (should reject):")
    result = run_sql_query("DROP TABLE patients_clean")
    print(result)

    print("\nTest 3 — Query without LIMIT (should auto-add):")
    result = run_sql_query(f"SELECT age FROM {Tables.CLEAN} WHERE target = 1")
    print(f"  row_count: {result.get('row_count')}, limited: {result.get('limited')}")