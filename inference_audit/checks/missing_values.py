import pandas as pd
from inference_audit.report import CheckResult

def check_missing_values(df: pd.DataFrame, text_col: str, min_length: int = 3) -> CheckResult:
    # Defense-in-depth only: primary column validation happens once,
    # centrally, in loader.py before any check runs (per design doc's
    # Public API section). This guard exists so the check still honors
    # its "never raises" contract if ever called directly (e.g. in an
    # isolated unit test), not as the main validation path.
    if text_col not in df.columns:
        return CheckResult(score=None, warning=f"Column '{text_col}' not found in dataset.", details={"error": "missing_column"})
    total_rows = len(df)
    if total_rows == 0:
        return CheckResult(score=None, warning="Dataset is empty (0 rows) — nothing to check.", details={"error": "empty_dataframe"})
    is_null = df[text_col].isna()
    is_whitespace_only = df[text_col].fillna("").astype(str).str.strip() == ""
    is_whitespace_only = is_whitespace_only & ~is_null
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
        warning = f"{flagged_count}/{total_rows} rows ({missing_rate:.1%}) flagged: {null_count} null, {whitespace_count} whitespace-only, {short_count} below {min_length}-character threshold."
    details = {"total_rows": total_rows, "flagged_count": flagged_count, "null_count": null_count, "whitespace_only_count": whitespace_count, "too_short_count": short_count, "missing_rate": round(missing_rate, 4), "min_length_threshold": min_length}
    return CheckResult(score=score, warning=warning, details=details)
