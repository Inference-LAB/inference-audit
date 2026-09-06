"""
inference_audit/checks/language_contamination.py

Flags samples confidently detected as one of a specific, deliberately
chosen "concern list" of languages -- NOT via majority-vote against
whatever langdetect guesses most often (replaced entirely in Week 4
per Lead Engineer review -- see git history for the prior approach and
why it was abandoned: majority-vote over langdetect's essentially
random guessing for Roman Urdu produced an unstable, uninterpretable
score that swung 3x depending on dataset composition/size).
"""

import pandas as pd
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from inference_audit.report import CheckResult
from inference_audit.config import (
    LANGUAGE_CONCERN_LANGUAGES,
    LANGUAGE_CONFIDENCE_THRESHOLD,
    validate_probability_threshold,
)

DetectorFactory.seed = 0


def check_language_contamination(
    df: pd.DataFrame,
    text_col: str,
    concern_languages=LANGUAGE_CONCERN_LANGUAGES,
    confidence_threshold: float = LANGUAGE_CONFIDENCE_THRESHOLD,
) -> CheckResult:
    """
    Flags samples confidently detected as a language on the concern list.

    Args:
        df:                     The dataset as a pandas DataFrame.
        text_col:               Name of the column containing text to check.
        concern_languages:      Iterable of ISO 639-1 codes to treat as
                                 contamination risks if confidently
                                 detected. Default from config.py
                                 (LANGUAGE_CONCERN_LANGUAGES) -- see
                                 that module for the documented rationale
                                 (English/Hindi as realistic risks for a
                                 Roman Urdu corpus). Moved to config per
                                 review, so this can be tuned per-dataset
                                 without touching check logic.
        confidence_threshold:   Minimum langdetect confidence to trust a
                                 detection. Must be in [0, 1]. Default
                                 from config.py (LANGUAGE_CONFIDENCE_THRESHOLD).
                                 BOUNDARY: a detection with confidence
                                 EXACTLY equal to this threshold DOES
                                 count as a match (>=, inclusive) -- made
                                 explicit per review; see test suite for
                                 a boundary-value test confirming this.

    KNOWN, BOUNDED LIMITATION: genuine Roman Urdu text can still be
    confidently misdetected specifically as English at a measured rate
    (~11-12% on test data) -- see tests/test_language_contamination.py
    for the documented baseline and tolerance around this figure. This
    is a real characteristic of langdetect's behavior on Roman Urdu,
    not something this check can fully eliminate.

    Never raises. Returns CheckResult(score=None, ...) for: missing
    column, empty dataframe, invalid confidence_threshold, empty
    concern_languages, or zero rows with detectable text.
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

    validation_error = validate_probability_threshold("confidence_threshold", confidence_threshold)
    if validation_error:
        return CheckResult(
            score=None,
            warning=validation_error,
            details={"error": "invalid_confidence_threshold", "confidence_threshold": confidence_threshold},
        )

    concern_set = set(concern_languages) if concern_languages else set()
    if not concern_set:
        return CheckResult(
            score=None,
            warning="concern_languages must be a non-empty list of language codes.",
            details={"error": "empty_concern_languages"},
        )

    # Performance: avoid the astype(str) copy when the column is
    # already string-typed (per review -- "if this shows up in
    # profiling"). Falls back to the safe full-cast path if a non-string
    # value slips through an object-dtype column, rather than risking
    # an AttributeError deep in detect_langs().
    if df[text_col].dtype == object:
        texts = df[text_col].fillna("")
        try:
            _ = texts.str.len()  # cheap sanity check that all values behave as strings
        except (AttributeError, TypeError):
            texts = texts.astype(str)
    else:
        texts = df[text_col].fillna("").astype(str)

    detection_cache = {}
    unique_texts = texts.unique()
    for text in unique_texts:
        try:
            langs = detect_langs(text)
            detection_cache[text] = (langs[0].lang, langs[0].prob)
        except LangDetectException:
            detection_cache[text] = None

    skipped_undetectable_count = 0
    contamination_count = 0
    detectable_count = 0

    for text in texts:
        result = detection_cache[text]
        if result is None:
            skipped_undetectable_count += 1
            continue
        detectable_count += 1
        lang, prob = result
        # BOUNDARY (made explicit per review): >= , not > -- a
        # detection exactly at the configured threshold counts as a
        # confident match. See test suite for a boundary-value test.
        if lang in concern_set and prob >= confidence_threshold:
            contamination_count += 1

    if detectable_count == 0:
        return CheckResult(
            score=None,
            warning=f"No text in '{text_col}' could be language-detected "
                    f"(all {total_rows} rows were empty, whitespace, or too short).",
            details={"error": "no_detectable_text", "skipped_undetectable_count": skipped_undetectable_count},
        )

    contamination_rate = contamination_count / detectable_count
    score = round(100 * (1 - contamination_rate), 2)

    warning_parts = []
    if contamination_count > 0:
        warning_parts.append(
            f"{contamination_count}/{detectable_count} detectable rows "
            f"({contamination_rate:.1%}) confidently detected as a concern-list "
            f"language ({', '.join(sorted(concern_set))})."
        )
    if skipped_undetectable_count > 0:
        warning_parts.append(
            f"{skipped_undetectable_count} row(s) skipped (text too short/empty "
            f"for language detection)."
        )
    warning = " ".join(warning_parts) if warning_parts else None

    details = {
        "total_rows": total_rows,
        "unique_text_count": len(unique_texts),
        "detectable_count": detectable_count,
        "skipped_undetectable_count": skipped_undetectable_count,
        "contamination_count": contamination_count,
        "contamination_rate": round(contamination_rate, 4),
        "concern_languages": sorted(concern_set),
        "confidence_threshold": confidence_threshold,
    }

    return CheckResult(score=score, warning=warning, details=details)
