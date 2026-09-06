"""
inference_audit/checks/annotation_consistency.py

Flags low-confidence annotations that indicate genuine annotator
disagreement. conf_col is optional -- if not provided, this check
returns score=None with a clear explanation rather than failing.
"""

import pandas as pd
from inference_audit.report import CheckResult
from inference_audit.config import ANNOTATION_CONFIDENCE_THRESHOLD, validate_probability_threshold


def check_annotation_consistency(
    df: pd.DataFrame,
    conf_col: str = None,
    confidence_threshold: float = ANNOTATION_CONFIDENCE_THRESHOLD,
) -> CheckResult:
    """
    Flags samples with annotation confidence below a threshold.

    Args:
        df:                     The dataset as a pandas DataFrame.
        conf_col:               Name of the confidence column. If None,
                                 this check is skipped gracefully.
        confidence_threshold:   Confidence below this value is flagged
                                 as low-confidence. Must be in [0, 1].
                                 Default from config.py
                                 (ANNOTATION_CONFIDENCE_THRESHOLD) --
                                 moved there per review, so it's tunable
                                 without touching check logic and stays
                                 consistent with how other checks
                                 configure tunables.

    Never raises. Returns CheckResult(score=None, ...) for: no conf_col
    provided, missing column, empty dataframe, invalid threshold,
    non-numeric confidence values, or all confidence values being null.
    """
    total_rows = len(df)

    if conf_col is None:
        return CheckResult(
            score=None,
            warning="No confidence column provided — annotation consistency check skipped.",
            details={"error": "no_confidence_column"},
        )

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

    # Centralized validation, per review -- was previously duplicated
    # inline in every check that takes a [0,1]-bounded threshold.
    validation_error = validate_probability_threshold("confidence_threshold", confidence_threshold)
    if validation_error:
        return CheckResult(
            score=None,
            warning=validation_error,
            details={"error": "invalid_confidence_threshold", "confidence_threshold": confidence_threshold},
        )

    is_null = df[conf_col].isna()
    null_count = int(is_null.sum())
    non_null_values = df[conf_col].dropna()

    if len(non_null_values) == 0:
        return CheckResult(
            score=None,
            warning=f"All {total_rows} values in '{conf_col}' are null — cannot assess annotation consistency.",
            details={"error": "all_confidence_null", "null_count": null_count, "total_rows": total_rows},
        )

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
