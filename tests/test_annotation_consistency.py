"""
tests/test_annotation_consistency.py
"""
import pytest
import pandas as pd
from pathlib import Path
from inference_audit.checks.annotation_consistency import check_annotation_consistency

FIXTURES = Path("tests/fixtures")

# Fixture is generated with a fixed random seed (42), making its exact
# composition deterministic and reproducible. Per review, replacing the
# previous vague 30-50 range with the actual measured value + a small
# tolerance for floating-point rounding -- this is easier to maintain
# than an arbitrary range disconnected from how the fixture is built,
# and still tolerant to trivial rounding differences across environments.
EXPECTED_LOW_CONFIDENCE_SCORE = 46.4
SCORE_TOLERANCE = 1.0


class TestAnnotationConsistency:
    def test_high_confidence_scores_high(self):
        df = pd.read_csv(FIXTURES / "high_confidence.csv")
        result = check_annotation_consistency(df, conf_col="confidence")
        assert result.score >= 90, f"got {result.score}. Details: {result.details}"
        assert result.warning is None

    def test_low_confidence_matches_documented_baseline(self):
        """
        Uses the fixture's actual, reproducible score (measured directly
        from the seeded fixture generator) plus a small tolerance,
        rather than an arbitrary range -- if the scoring formula
        changes, this test's expected value should be re-measured and
        updated deliberately, not left vague.
        """
        df = pd.read_csv(FIXTURES / "low_confidence.csv")
        result = check_annotation_consistency(df, conf_col="confidence")
        assert abs(result.score - EXPECTED_LOW_CONFIDENCE_SCORE) <= SCORE_TOLERANCE, (
            f"Expected {EXPECTED_LOW_CONFIDENCE_SCORE} +/- {SCORE_TOLERANCE}, got {result.score}. "
            f"If the scoring formula changed intentionally, re-measure and update "
            f"EXPECTED_LOW_CONFIDENCE_SCORE rather than widening this tolerance."
        )
        assert result.warning is not None

    def test_no_confidence_column_provided_is_graceful_skip(self):
        df = pd.read_csv(FIXTURES / "high_confidence.csv")
        result = check_annotation_consistency(df, conf_col=None)
        assert result.score is None
        assert "skipped" in result.warning.lower()

    def test_missing_column_returns_none_score(self):
        df = pd.read_csv(FIXTURES / "high_confidence.csv")
        result = check_annotation_consistency(df, conf_col="nonexistent_column")
        assert result.score is None

    def test_empty_dataframe_returns_none_score(self):
        result = check_annotation_consistency(pd.DataFrame({"confidence": []}), conf_col="confidence")
        assert result.score is None

    def test_invalid_threshold_above_one_returns_none(self):
        df = pd.read_csv(FIXTURES / "high_confidence.csv")
        result = check_annotation_consistency(df, conf_col="confidence", confidence_threshold=1.5)
        assert result.score is None
        assert "confidence_threshold" in result.warning

    def test_invalid_threshold_negative_returns_none(self):
        df = pd.read_csv(FIXTURES / "high_confidence.csv")
        result = check_annotation_consistency(df, conf_col="confidence", confidence_threshold=-0.1)
        assert result.score is None

    def test_null_confidence_values_reported_explicitly(self):
        df = pd.DataFrame({"confidence": [0.9, 0.8, None, 0.3, None, 0.95]})
        result = check_annotation_consistency(df, conf_col="confidence")
        assert result.details["null_count"] == 2
        assert result.details["non_null_count"] == 4

    def test_all_null_confidence_returns_none(self):
        df = pd.DataFrame({"confidence": [None, None, None]})
        result = check_annotation_consistency(df, conf_col="confidence")
        assert result.score is None

    def test_non_numeric_values_returns_none(self):
        df = pd.DataFrame({"confidence": [0.9, "high", 0.3, "low"]})
        result = check_annotation_consistency(df, conf_col="confidence")
        assert result.score is None
        assert result.details["non_numeric_count"] == 2

    def test_default_threshold_comes_from_config(self):
        """Confirms the default is sourced from config.py, not a code-embedded literal."""
        from inference_audit.config import ANNOTATION_CONFIDENCE_THRESHOLD
        df = pd.read_csv(FIXTURES / "high_confidence.csv")
        result = check_annotation_consistency(df, conf_col="confidence")
        assert result.details["confidence_threshold"] == ANNOTATION_CONFIDENCE_THRESHOLD
