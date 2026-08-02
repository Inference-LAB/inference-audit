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
                                 at 0. Default 20 (i.e. 20:1 imbalance or
                                 worse scores 0). Configurable per design doc.

    Returns:
        CheckResult with score (0-100), warning (str or None), and
        details (dict with per-class counts and imbalance_ratio).

    Never raises. Returns CheckResult(score=None, ...) if the check
    cannot run (missing column, empty dataframe, or fewer than 2 classes
    present -- imbalance is undefined for a single-class dataset).
    """
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

    # Drop nulls before counting classes -- a null label isn't a "class"
    class_counts = df[label_col].dropna().value_counts()

    if len(class_counts) == 0:
        return CheckResult(
            score=None,
            warning=f"All values in '{label_col}' are null — cannot assess distribution.",
            details={"error": "all_labels_null"},
        )

    if len(class_counts) == 1:
        # A single-class dataset is technically "perfectly balanced" in
        # the sense that there's no imbalance to measure -- but it's a
        # meaningful edge case worth flagging distinctly, not silently
        # scoring 100 as if this were a healthy multi-class dataset.
        only_label = class_counts.index[0]
        return CheckResult(
            score=None,
            warning=f"Only one class present ('{only_label}') — imbalance ratio undefined.",
            details={
                "error": "single_class",
                "class_counts": class_counts.to_dict(),
            },
        )

    majority_count = int(class_counts.max())
    minority_count = int(class_counts.min())
    imbalance_ratio = majority_count / minority_count

    # --- Score formula, per design doc ---
    # 1:1 balance (ratio=1.0) -> score 100
    # ratio >= max_acceptable_ratio -> score 0
    # linear in between.
    #
    # NOTE: imbalance_ratio is 1.0 even for perfectly balanced data
    # (majority/minority = 1), not 0 -- so we normalize against the
    # EXCESS ratio beyond the 1:1 baseline, not the raw ratio itself.
    # (Caught via testing: raw ratio/max_ratio wrongly scored a 50/50
    # split as 95, not 100.)
    excess = imbalance_ratio - 1.0
    max_excess = max_acceptable_ratio - 1.0
    capped_excess = min(excess / max_excess, 1.0)
    score = round(100 * (1 - capped_excess), 2)

    warning = None
    if imbalance_ratio > 1.0:
        majority_label = class_counts.idxmax()
        minority_label = class_counts.idxmin()
        warning = (
            f"Class imbalance detected: '{majority_label}' has {majority_count} samples, "
            f"'{minority_label}' has only {minority_count} (ratio {imbalance_ratio:.1f}:1)."
        )

    details = {
        "total_rows": total_rows,
        "num_classes": len(class_counts),
        "class_counts": class_counts.to_dict(),
        "majority_count": majority_count,
        "minority_count": minority_count,
        "imbalance_ratio": round(imbalance_ratio, 2),
        "max_acceptable_ratio": max_acceptable_ratio,
    }

    return CheckResult(score=score, warning=warning, details=details)
