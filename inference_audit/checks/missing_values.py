"""
inference_audit/checks/missing_values.py

Detects null, whitespace-only, and suspiciously short text in a dataset
column. Per Technical Feasibility section of the design doc, this is the
simplest of the five checks -- no external ML dependency, pure pandas.
"""

import pandas as pd
from inference_audit.report import CheckResult


def check_missing_values(df: pd.DataFrame, text_col: str, min_length: int = 3) -> CheckResult:
    """
    Flags null, whitespace-only, and below-length-threshold text samples.

    Args:
        df:         The dataset as a pandas DataFrame.
        text_col:   Name of the column containing text to check.
        min_length: Minimum character length (after stripping) for text
                    to NOT be flagged as suspiciously short. Default 3.

    Returns:
        CheckResult with score (0-100), warning (str or None), and
        details (dict with raw counts for null/whitespace/short rows).

    Never raises. If the check cannot run (e.g. missing column, empty
    dataframe), returns CheckResult(score=None, ...) with an explanation.
    """
    # --- Guard clauses: never let this function crash the whole audit ---
    if text_col not in df.columns:
        return CheckResult(
            score=None,
            warning=f"Column '{text_col}' not found in dataset.",
            details={"error": "missing_column"},
        )

    total_rows = len(df)
    if total_rows == 0:
        return CheckResult(
            score=None,
            warning="Dataset is empty (0 rows) — nothing to check.",
            details={"error": "empty_dataframe"},
        )

    # --- The three flag conditions ---
    # 1. Null / NaN values
    is_null = df[text_col].isna()

    # 2. Whitespace-only strings (e.g. "   ", "\t\n")
    #    Guard: only check .str.strip() on non-null values, since
    #    .str accessor on NaN would itself produce NaN, not crash --
    #    but we handle it explicitly for clarity.
    is_whitespace_only = df[text_col].fillna("").astype(str).str.strip() == ""
    # Don't double-count rows that are already null as "whitespace-only"
    is_whitespace_only = is_whitespace_only & ~is_null

    # 3. Below-length-threshold text (after stripping whitespace)
    stripped_lengths = df[text_col].fillna("").astype(str).str.strip().str.len()
    is_too_short = (stripped_lengths < min_length) & ~is_null & ~is_whitespace_only

    # --- Combine all three into one "flagged" mask ---
    is_flagged = is_null | is_whitespace_only | is_too_short
    flagged_count = int(is_flagged.sum())

    null_count = int(is_null.sum())
    whitespace_count = int(is_whitespace_only.sum())
    short_count = int(is_too_short.sum())

    # --- Score formula, per design doc: 100 x (1 - missing_rate) ---
    missing_rate = flagged_count / total_rows
    score = round(100 * (1 - missing_rate), 2)

    # --- Build a human-readable warning only if something was found ---
    warning = None
    if flagged_count > 0:
        warning = (
            f"{flagged_count}/{total_rows} rows ({missing_rate:.1%}) flagged: "
            f"{null_count} null, {whitespace_count} whitespace-only, "
            f"{short_count} below {min_length}-character threshold."
        )

    details = {
        "total_rows": total_rows,
        "flagged_count": flagged_count,
        "null_count": null_count,
        "whitespace_only_count": whitespace_count,
        "too_short_count": short_count,
        "missing_rate": round(missing_rate, 4),
        "min_length_threshold": min_length,
    }

    return CheckResult(score=score, warning=warning, details=details)
