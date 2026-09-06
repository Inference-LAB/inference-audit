"""
inference_audit/config.py

Centralizes tunable scoring/threshold parameters across checks, so they
can be adjusted without touching check logic -- per review requesting
consistency in how tunables are configured across the project.
"""

from typing import Optional


# --- check_annotation_consistency ---
ANNOTATION_CONFIDENCE_THRESHOLD = 0.6

# --- check_language_contamination ---
# DELIBERATE, DOCUMENTED DEFAULT -- not a silent hardcode.
# English: code-switching with English is extremely common in real
# Roman Urdu social media text -- a genuine, likely contamination source.
# Hindi: Hindi and Urdu share enough vocabulary/grammatical overlap
# that misclassification or genuine cross-contamination between the two
# is plausible in scraped corpora. Revisit this list if the tool is
# used on corpora with different realistic contamination risks.
LANGUAGE_CONCERN_LANGUAGES = ("en", "hi")
LANGUAGE_CONFIDENCE_THRESHOLD = 0.9


def validate_probability_threshold(name: str, value: float) -> Optional[str]:
    """
    Centralized validation for any [0, 1]-bounded threshold parameter,
    used by every check that takes a probability/confidence threshold
    (rather than each check re-implementing the same bounds check).

    Returns None if valid, or a ready-to-use warning message string if
    invalid -- the caller decides how to surface it (e.g. wrapping it
    in a CheckResult(score=None, warning=..., ...)).
    """
    if not (0.0 <= value <= 1.0):
        return f"{name} must be between 0 and 1 (got {value})."
    return None
