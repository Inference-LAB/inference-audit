import pandas as pd
from inference_audit.report import CheckResult

def check_label_distribution(df: pd.DataFrame, label_col: str, max_acceptable_ratio: float = 20.0) -> CheckResult:
    # Defense-in-depth only: primary column validation happens once,
    # centrally, in loader.py before any check runs (per design doc's
    # Public API section). This guard exists so the check still honors
    # its "never raises" contract if ever called directly (e.g. in an
    # isolated unit test), not as the main validation path.
    if label_col not in df.columns:
        return CheckResult(score=None, warning=f"Column '{label_col}' not found in dataset.", details={"error": "missing_column"})
    total_rows = len(df)
    if total_rows == 0:
        return CheckResult(score=None, warning="Dataset is empty (0 rows) — nothing to check.", details={"error": "empty_dataframe"})
    class_counts = df[label_col].dropna().value_counts()
    if len(class_counts) == 0:
        return CheckResult(score=None, warning=f"All values in '{label_col}' are null — cannot assess distribution.", details={"error": "all_labels_null"})
    if len(class_counts) == 1:
        only_label = class_counts.index[0]
        return CheckResult(score=None, warning=f"Only one class present ('{only_label}') — imbalance ratio undefined.", details={"error": "single_class", "class_counts": class_counts.to_dict()})
    majority_count = int(class_counts.max())
    minority_count = int(class_counts.min())
    imbalance_ratio = majority_count / minority_count
    excess = imbalance_ratio - 1.0
    max_excess = max_acceptable_ratio - 1.0
    capped_excess = min(excess / max_excess, 1.0)
    score = round(100 * (1 - capped_excess), 2)
    warning = None
    if imbalance_ratio > 1.0:
        majority_label = class_counts.idxmax()
        minority_label = class_counts.idxmin()
        warning = f"Class imbalance detected: '{majority_label}' has {majority_count} samples, '{minority_label}' has only {minority_count} (ratio {imbalance_ratio:.1f}:1)."
    details = {"total_rows": total_rows, "num_classes": len(class_counts), "class_counts": class_counts.to_dict(), "majority_count": majority_count, "minority_count": minority_count, "imbalance_ratio": round(imbalance_ratio, 2), "max_acceptable_ratio": max_acceptable_ratio}
    return CheckResult(score=score, warning=warning, details=details)
