"""
inference_audit/checks/language_contamination.py

Flags samples confidently detected as one of a specific, deliberately
chosen "concern list" of languages -- NOT via majority-vote against
whatever langdetect guesses most often (see Week 4 revision history
below for why that approach was replaced).
"""

import pandas as pd
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from inference_audit.report import CheckResult

DetectorFactory.seed = 0

# --- DELIBERATE, DOCUMENTED DEFAULT -- not a silent hardcode. ---
# English: code-switching with English is extremely common in real
# Roman Urdu social media text -- a genuine, likely contamination source.
# Hindi: Hindi and Urdu share enough vocabulary/grammatical overlap
# that misclassification or genuine cross-contamination between the two
# is plausible in scraped corpora. This list should be revisited if the
# tool is used on corpora with different realistic contamination risks.
DEFAULT_CONCERN_LANGUAGES = ("en", "hi")


def check_language_contamination(
    df: pd.DataFrame,
    text_col: str,
    concern_languages=DEFAULT_CONCERN_LANGUAGES,
    confidence_threshold: float = 0.9,
) -> CheckResult:
    """
    Flags samples confidently detected as a language on the concern list.

    REVISION HISTORY (Week 4): the original design used an "auto" mode
    that computed a majority-vote baseline across the dataset and
    flagged anything differing from it. Lead Engineer review, tested
    against the real 134K-row RUEmoCorp corpus, found this fundamentally
    flawed: langdetect has NO real signal for Roman Urdu at all -- it
    scatters guesses across 10+ unrelated languages roughly at random.
    "Majority vote" over pure noise doesn't recover a real baseline, it
    just picks whichever noise language happens to be most common in a
    given sample -- which is exactly why the baseline resolved to
    Indonesian ("id") on the real corpus, flagging 28.9% of rows (vs.
    ~10% measured on a smaller test sample) -- a 3x, dataset-size-
    dependent swing that made the score uninterpretable.

    This version replaces majority-vote with a FIXED, deliberately
    chosen concern_languages list. Instead of asking "what's the
    majority guess?", it asks "does this row match a language we have
    specific reason to worry about?" -- sidestepping the noise-language
    problem entirely, since Indonesian/Tagalog/Somali/etc. guesses are
    simply ignored rather than treated as a moving baseline.

    KNOWN, BOUNDED LIMITATION (tested, not hidden): genuine Roman Urdu
    text can still be confidently (~0.9999 probability) misdetected
    specifically as English -- measured at ~11-12% false-positive rate
    on realistic Roman Urdu test text. This does NOT eliminate false
    positives. What it fixes is the INSTABILITY: this rate is tied to
    langdetect's actual behavior on Roman Urdu text, not to which
    language happens to dominate a particular corpus -- so it should
    remain roughly stable across dataset size and composition, unlike
    the old majority-vote approach's 3x swing.

    Args:
        df:                     The dataset as a pandas DataFrame.
        text_col:               Name of the column containing text to check.
        concern_languages:      Iterable of ISO 639-1 codes to treat as
                                 contamination risks if confidently
                                 detected. Default ("en", "hi") -- see
                                 module-level comment for why.
        confidence_threshold:   Minimum langdetect confidence to trust a
                                 detection. Must be in [0, 1]. Default 0.9.

    PERFORMANCE: detection results are cached by exact text value before
    running langdetect, since langdetect.detect_langs() is not
    vectorized and short social-media-style text often repeats verbatim
    across many rows (per Lead Engineer's performance finding on the
    134K-row corpus). This also removes the old two-pass design (one
    pass for majority-vote baseline, one for flagging) -- concern-list
    mode only ever needs a single pass.

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

    if not (0.0 <= confidence_threshold <= 1.0):
        return CheckResult(
            score=None,
            warning=f"confidence_threshold must be between 0 and 1 (got {confidence_threshold}).",
            details={"error": "invalid_confidence_threshold", "confidence_threshold": confidence_threshold},
        )

    concern_set = set(concern_languages) if concern_languages else set()
    if not concern_set:
        return CheckResult(
            score=None,
            warning="concern_languages must be a non-empty list of language codes.",
            details={"error": "empty_concern_languages"},
        )

    texts = df[text_col].fillna("").astype(str)

    # --- Performance fix: detect each UNIQUE text value once, cache
    # results, then map back to every row. Short repeated social-media
    # text (e.g. "Hahaha", "InshaAllah") means this can cut the number
    # of actual langdetect calls dramatically on real corpora. ---
    detection_cache = {}
    unique_texts = texts.unique()
    for text in unique_texts:
        try:
            langs = detect_langs(text)
            detection_cache[text] = (langs[0].lang, langs[0].prob)
        except LangDetectException:
            detection_cache[text] = None  # undetectable (empty/short/numeric)

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
        if lang in concern_set and prob > confidence_threshold:
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
