"""
inference_audit/checks/label_distribution.py

Detects class imbalance in a labeled dataset -- the #1 cause of misleading
accuracy claims in dataset papers, per the project brief.
"""

import pandas as pd
from inference_audit.report import CheckResult


def check_label_distribution(
    df: pd.DataFrame,
    label_col: str,
    max_acceptable_ratio: float = 20.0,
) -> CheckResult:
    """
    Measures class imbalance via majority:minority count ratio.

    Args:
        df:                     The dataset as a pandas DataFrame.
        label_col:              Name of the column containing class labels.
        max_acceptable_ratio:   Ratio at or beyond which score bottoms out
                                 at 0. Must be > 1.0. Default 20.

    Returns:
        CheckResult with score (0-100), warning (str or None), and
        details (dict with per-class counts, imbalance_ratio, and
        null_count -- null-labeled rows are always reported explicitly,
        never silently dropped from the reported picture).

    Never raises. Returns CheckResult(score=None, ...) for edge cases:
    missing column, empty dataframe, invalid max_acceptable_ratio,
    all-null labels, or fewer than 2 non-null classes present.
    """
    # Defense-in-depth only: primary column validation happens once,
    # centrally, in loader.py before any check runs (per design doc's
    # Public API section). This guard exists so the check still honors
    # its "never raises" contract if ever called directly (e.g. in an
    # isolated unit test), not as the main validation path.
    if label_col not in df.columns:
        return CheckResult(
            score=None,
            warning=f"Column '{label_col}' not found in dataset.",
            details={"error": "missing_column"},
        )

    total_rows = len(df)
    if total_rows == 0:
        return CheckResult(
            score=None,
            warning="Dataset is empty (0 rows) — nothing to check.",
            details={"error": "empty_dataframe"},
        )

    # Director review: validate max_acceptable_ratio explicitly. A value
    # <= 1.0 makes max_excess <= 0, which would either divide by zero or
    # produce a nonsensical (negative/inverted) scoring curve -- there
    # must be real room between "balanced" (ratio=1.0) and "maximally
    # imbalanced" (ratio=max_acceptable_ratio) for the formula to mean
    # anything.
    if max_acceptable_ratio <= 1.0:
        return CheckResult(
            score=None,
            warning=(
                f"max_acceptable_ratio must be > 1.0 (got {max_acceptable_ratio}) — "
                f"a value at or below 1.0 leaves no range between 'balanced' and "
                f"'maximally imbalanced' for scoring."
            ),
            details={"error": "invalid_max_acceptable_ratio", "max_acceptable_ratio": max_acceptable_ratio},
        )

    # Director review: null labels were being silently excluded from
    # class_counts (via dropna()) while total_rows still included them --
    # meaning majority_count + minority_count would NOT sum to total_rows
    # whenever nulls were present, with no indication why. null_count is
    # now computed explicitly and reported in EVERY return path below,
    # not just the success path.
    is_null = df[label_col].isna()
    null_count = int(is_null.sum())

    class_counts = df[label_col].dropna().value_counts()

    if len(class_counts) == 0:
        return CheckResult(
            score=None,
            warning=f"All {total_rows} values in '{label_col}' are null — cannot assess distribution.",
            details={"error": "all_labels_null", "null_count": null_count, "total_rows": total_rows},
        )

    if len(class_counts) == 1:
        # Director review: single-class case must also report null_count,
        # not just note the single class in isolation.
        only_label = class_counts.index[0]
        warning = f"Only one class present ('{only_label}') — imbalance ratio undefined."
        if null_count > 0:
            warning += f" Additionally, {null_count}/{total_rows} rows have null labels."
        return CheckResult(
            score=None,
            warning=warning,
            details={
                "error": "single_class",
                "class_counts": class_counts.to_dict(),
                "null_count": null_count,
                "total_rows": total_rows,
            },
        )

    majority_count = int(class_counts.max())
    minority_count = int(class_counts.min())
    imbalance_ratio = majority_count / minority_count

    # Score formula: 1:1 balance (ratio=1.0) -> score 100;
    # ratio >= max_acceptable_ratio -> score 0; linear in between.
    # Normalized against EXCESS beyond the 1:1 baseline (see Week 2 bug
    # writeup -- raw ratio/max_ratio wrongly scored balanced data < 100).
    excess = imbalance_ratio - 1.0
    max_excess = max_acceptable_ratio - 1.0
    capped_excess = min(excess / max_excess, 1.0)
    score = round(100 * (1 - capped_excess), 2)

    # Director review: null-labeled rows are now always mentioned in the
    # warning when present, alongside (not instead of) the imbalance
    # warning -- both can be true at once and both matter to a reader.
    warning_parts = []
    if imbalance_ratio > 1.0:
        majority_label = class_counts.idxmax()
        minority_label = class_counts.idxmin()
        warning_parts.append(
            f"Class imbalance detected: '{majority_label}' has {majority_count} samples, "
            f"'{minority_label}' has only {minority_count} (ratio {imbalance_ratio:.1f}:1)."
        )
    if null_count > 0:
        null_rate = null_count / total_rows
        warning_parts.append(
            f"{null_count}/{total_rows} rows ({null_rate:.1%}) have null labels and "
            f"were excluded from the class distribution calculation."
        )
    warning = " ".join(warning_parts) if warning_parts else None

    details = {
        "total_rows": total_rows,
        "num_classes": len(class_counts),
        "class_counts": class_counts.to_dict(),
        "majority_count": majority_count,
        "minority_count": minority_count,
        "imbalance_ratio": round(imbalance_ratio, 2),
        "max_acceptable_ratio": max_acceptable_ratio,
        "null_count": null_count,
        "null_rate": round(null_count / total_rows, 4),
    }

    return CheckResult(score=score, warning=warning, details=details)
