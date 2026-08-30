"""
inference_audit/checks/annotation_consistency.py

Flags low-confidence annotations that indicate genuine annotator
disagreement. Per the design doc's Public API: conf_col is optional --
if not provided, this check returns score=None with a clear explanation
rather than failing, since not every dataset has a confidence column.
"""

import pandas as pd
from inference_audit.report import CheckResult


def check_annotation_consistency(
    df: pd.DataFrame,
    conf_col: str = None,
    confidence_threshold: float = 0.6,
) -> CheckResult:
    """
    Flags samples with annotation confidence below a threshold.

    Args:
        df:                     The dataset as a pandas DataFrame.
        conf_col:               Name of the confidence column. If None
                                 (not provided), this check is skipped
                                 gracefully -- not every dataset has one.
        confidence_threshold:   Confidence below this value is flagged
                                 as low-confidence. Must be in [0, 1].
                                 Default 0.6.

    Returns:
        CheckResult with score (0-100), warning (str or None), and
        details (dict with low-confidence count, null count in the
        confidence column, and the threshold used).

    Never raises. Returns CheckResult(score=None, ...) for: no conf_col
    provided (expected, normal case -- not an error), missing column,
    empty dataframe, invalid threshold, non-numeric confidence values,
    or all confidence values being null.
    """
    total_rows = len(df)

    # No confidence column provided is the EXPECTED, normal case for
    # many datasets -- per design doc's Public API, this is a graceful
    # skip, not an error condition.
    if conf_col is None:
        return CheckResult(
            score=None,
            warning="No confidence column provided — annotation consistency check skipped.",
            details={"error": "no_confidence_column"},
        )

    # Defense-in-depth only: primary column validation happens once,
    # centrally, in loader.py before any check runs. This guard exists
    # so the check still honors its "never raises" contract if ever
    # called directly, not as the main validation path.
    if conf_col not in df.columns:
        return CheckResult(
            score=None,
            warning=f"Column '{conf_col}' not found in dataset.",
            details={"error": "missing_column"},
        )

    if total_rows == 0:
        return CheckResult(
            score=None,
            warning="Dataset is empty (0 rows) — nothing to check.",
            details={"error": "empty_dataframe"},
        )

    # Validate threshold explicitly -- same discipline the director
    # asked for on label_distribution's max_acceptable_ratio. A
    # confidence threshold outside [0, 1] is meaningless, since
    # confidence values themselves are expected in that range.
    if not (0.0 <= confidence_threshold <= 1.0):
        return CheckResult(
            score=None,
            warning=f"confidence_threshold must be between 0 and 1 (got {confidence_threshold}).",
            details={"error": "invalid_confidence_threshold", "confidence_threshold": confidence_threshold},
        )

    # Applying the same null-handling discipline the director's review
    # required for label_distribution: null confidence values are
    # computed and reported explicitly, never silently dropped from the
    # denominator without a trace.
    is_null = df[conf_col].isna()
    null_count = int(is_null.sum())
    non_null_values = df[conf_col].dropna()

    if len(non_null_values) == 0:
        return CheckResult(
            score=None,
            warning=f"All {total_rows} values in '{conf_col}' are null — cannot assess annotation consistency.",
            details={"error": "all_confidence_null", "null_count": null_count, "total_rows": total_rows},
        )

    # Applying the same non-numeric-value discipline the director's
    # review required for missing_values: a confidence column should
    # contain numeric values, not silently coerced strings/objects.
    is_numeric_mask = non_null_values.apply(lambda v: isinstance(v, (int, float)) and not isinstance(v, bool))
    non_numeric_count = int((~is_numeric_mask).sum())
    if non_numeric_count > 0:
        return CheckResult(
            score=None,
            warning=(
                f"Column '{conf_col}' contains {non_numeric_count} non-numeric value(s). "
                f"This check requires numeric confidence values."
            ),
            details={"error": "non_numeric_values", "non_numeric_count": non_numeric_count},
        )

    non_null_count = len(non_null_values)
    is_low_confidence = non_null_values < confidence_threshold
    low_confidence_count = int(is_low_confidence.sum())

    # Score formula, per design doc: 100 x (1 - low_confidence_rate).
    # Rate is computed over non-null values only -- nulls are reported
    # separately (see warning/details below), not silently folded into
    # either the numerator or denominator.
    low_confidence_rate = low_confidence_count / non_null_count
    score = round(100 * (1 - low_confidence_rate), 2)

    warning_parts = []
    if low_confidence_count > 0:
        warning_parts.append(
            f"{low_confidence_count}/{non_null_count} annotated samples "
            f"({low_confidence_rate:.1%}) have confidence below {confidence_threshold}."
        )
    if null_count > 0:
        null_rate = null_count / total_rows
        warning_parts.append(
            f"{null_count}/{total_rows} rows ({null_rate:.1%}) have null confidence "
            f"values and were excluded from this calculation."
        )
    warning = " ".join(warning_parts) if warning_parts else None

    details = {
        "total_rows": total_rows,
        "non_null_count": non_null_count,
        "null_count": null_count,
        "low_confidence_count": low_confidence_count,
        "low_confidence_rate": round(low_confidence_rate, 4),
        "confidence_threshold": confidence_threshold,
    }

    return CheckResult(score=score, warning=warning, details=details)
