"""
inference_audit/checks/language_contamination.py

Flags samples in an unexpected language for the dataset. Implements the
REFRAMED approach from the Week 1 design doc: langdetect has no language
profile for Roman Urdu, so "not detected as Urdu" cannot be used as the
contamination signal (it would flag ~100% of a clean Roman Urdu corpus).

IMPORTANT, READ BEFORE CHANGING THRESHOLDS: live testing during Week 4
build showed that langdetect's confidence output is NOT a usable
uncertainty signal for Roman Urdu -- genuinely clean Roman Urdu text
gets confidently (~0.9999 probability) misdetected as unrelated
languages (Indonesian, Tagalog, etc.), at rates indistinguishable by
threshold tuning from true contamination. Raising confidence_threshold
does not help; the wrongly-flagged samples already sit near maximum
confidence. This is a structural limitation of langdetect, not a
parameter to tune away -- see Week 1 Task 2 findings and Known Risk #3
in the design doc.
"""

import pandas as pd
from collections import Counter
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from inference_audit.report import CheckResult

# Per Role Guide: this line matters more than most of the rest of the
# module. Without it, langdetect's internal random sampling makes
# results non-deterministic between runs on identical input (confirmed
# via Week 1 testing: 2/5 texts flipped detected language across
# repeated runs before this was set).
DetectorFactory.seed = 0


def check_language_contamination(
    df: pd.DataFrame,
    text_col: str,
    expected_language: str = "auto",
    confidence_threshold: float = 0.9,
) -> CheckResult:
    """
    Flags samples detected as an unexpected language.

    Two modes:

    1. SPECIFIC LANGUAGE (e.g. expected_language="en"): reliable. Flags
       rows confidently detected as anything other than the specified
       language. Works well for languages langdetect actually has a
       profile for (verified: 100% accuracy on real English/French/
       Spanish sanity checks in Week 1).

    2. "auto" (default): BEST-EFFORT HEURISTIC, not a reliable signal,
       for datasets in languages langdetect doesn't recognize (like
       Roman Urdu). Since there's no ground-truth "expected" language to
       compare against, this mode uses majority-vote: whatever language
       langdetect detects most often in the dataset becomes the
       operational baseline, and rows confidently detected as something
       else are flagged. Measured on real Roman Urdu test data: this
       reduces false positives from ~62% (naive "any confident
       detection = contamination") down to ~10%, but does NOT achieve
       clean separation. Use with this limitation in mind.

    Never raises. Returns CheckResult(score=None, ...) for: missing
    column, empty dataframe, invalid confidence_threshold, or zero rows
    with detectable text (all empty/too-short/numeric-only).
    """
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

    if not (0.0 <= confidence_threshold <= 1.0):
        return CheckResult(
            score=None,
            warning=f"confidence_threshold must be between 0 and 1 (got {confidence_threshold}).",
            details={"error": "invalid_confidence_threshold", "confidence_threshold": confidence_threshold},
        )

    texts = df[text_col].fillna("").astype(str)
    detections = {}
    skipped_undetectable_count = 0

    for idx, text in texts.items():
        try:
            langs = detect_langs(text)
            detections[idx] = (langs[0].lang, langs[0].prob)
        except LangDetectException:
            skipped_undetectable_count += 1

    if len(detections) == 0:
        return CheckResult(
            score=None,
            warning=f"No text in '{text_col}' could be language-detected "
                    f"(all {total_rows} rows were empty, whitespace, or too short).",
            details={"error": "no_detectable_text", "skipped_undetectable_count": skipped_undetectable_count},
        )

    if expected_language == "auto":
        detected_labels = [lang for lang, prob in detections.values()]
        baseline_language = Counter(detected_labels).most_common(1)[0][0]
        mode_description = f"auto (majority-vote baseline: '{baseline_language}')"
    else:
        baseline_language = expected_language
        mode_description = f"expected='{expected_language}'"

    contaminated_ids = [
        idx for idx, (lang, prob) in detections.items()
        if lang != baseline_language and prob > confidence_threshold
    ]
    contamination_count = len(contaminated_ids)
    detectable_count = len(detections)

    contamination_rate = contamination_count / detectable_count
    score = round(100 * (1 - contamination_rate), 2)

    warning_parts = []
    if contamination_count > 0:
        warning_parts.append(
            f"{contamination_count}/{detectable_count} detectable rows "
            f"({contamination_rate:.1%}) confidently detected as a language "
            f"other than '{baseline_language}'."
        )
    if skipped_undetectable_count > 0:
        warning_parts.append(
            f"{skipped_undetectable_count} row(s) skipped (text too short/empty "
            f"for language detection)."
        )
    if expected_language == "auto":
        warning_parts.append(
            "NOTE: 'auto' mode is a best-effort heuristic for languages langdetect "
            "doesn't recognize -- see module docstring for measured limitations."
        )
    warning = " ".join(warning_parts) if warning_parts else None

    details = {
        "total_rows": total_rows,
        "detectable_count": detectable_count,
        "skipped_undetectable_count": skipped_undetectable_count,
        "contamination_count": contamination_count,
        "contamination_rate": round(contamination_rate, 4),
        "baseline_language": baseline_language,
        "mode": mode_description,
        "confidence_threshold": confidence_threshold,
    }

    return CheckResult(score=score, warning=warning, details=details)
