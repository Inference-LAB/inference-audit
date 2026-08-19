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
        text_col:   Name of the column containing text to check. Must
                    contain string values (or null) -- see non-string
                    handling below.
        min_length: Minimum character length (after stripping) for text
                    to NOT be flagged as suspiciously short. Must be >= 0.
                    Default 3.

    Returns:
        CheckResult with score (0-100), warning (str or None), and
        details (dict with raw counts for null/whitespace/short rows).

    Never raises. Returns CheckResult(score=None, ...) for edge cases:
    missing column, empty dataframe, invalid min_length, or a column
    containing non-string values (numeric IDs, etc.) -- this check is
    restricted to text columns rather than silently coercing other
    types via str(), which could produce misleading results (e.g. a
    numeric ID like 5 becoming "5", length 1, flagged as "too short"
    even though it was never text in the first place).
    """
    # Defense-in-depth only: primary column validation happens once,
    # centrally, in loader.py before any check runs (per design doc's
    # Public API section). This guard exists so the check still honors
    # its "never raises" contract if ever called directly (e.g. in an
    # isolated unit test), not as the main validation path.
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

    # Director review: validate min_length explicitly rather than
    # allowing a negative value to silently produce a meaningless check
    # (every string has length >= 0, so a negative threshold would flag
    # nothing, ever, with no explanation why).
    if min_length < 0:
        return CheckResult(
            score=None,
            warning=f"min_length must be >= 0 (got {min_length}).",
            details={"error": "invalid_min_length", "min_length": min_length},
        )

    # Director review: previously used .astype(str) unconditionally,
    # which silently coerces non-string values (e.g. numeric IDs) before
    # the length check -- meaning a numeric ID column would produce
    # misleading "too short" flags rather than an honest "this isn't a
    # text column" signal. Now explicitly restricted to text columns:
    # non-null values that aren't strings cause the check to bail out
    # with a clear explanation, rather than silently reinterpreting them.
    non_null_values = df[text_col].dropna()
    if len(non_null_values) > 0:
        is_string_mask = non_null_values.apply(lambda v: isinstance(v, str))
        non_string_count = int((~is_string_mask).sum())
        if non_string_count > 0:
            return CheckResult(
                score=None,
                warning=(
                    f"Column '{text_col}' contains {non_string_count} non-string "
                    f"value(s) (e.g. numeric IDs). This check is intended for text "
                    f"columns -- results would be misleading if these were silently "
                    f"coerced to strings for the length check."
                ),
                details={"error": "non_string_values", "non_string_count": non_string_count},
            )

    is_null = df[text_col].isna()

    # Whitespace-only strings (e.g. "   ", "\t\n")
    is_whitespace_only = df[text_col].fillna("").astype(str).str.strip() == ""
    is_whitespace_only = is_whitespace_only & ~is_null

    # Below-length-threshold text (after stripping whitespace)
    stripped_lengths = df[text_col].fillna("").astype(str).str.strip().str.len()
    is_too_short = (stripped_lengths < min_length) & ~is_null & ~is_whitespace_only

    is_flagged = is_null | is_whitespace_only | is_too_short
    flagged_count = int(is_flagged.sum())

    null_count = int(is_null.sum())
    whitespace_count = int(is_whitespace_only.sum())
    short_count = int(is_too_short.sum())

    missing_rate = flagged_count / total_rows
    score = round(100 * (1 - missing_rate), 2)

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
