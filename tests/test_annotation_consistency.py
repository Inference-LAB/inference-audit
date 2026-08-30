"""
tests/test_annotation_consistency.py

Per Role Guide Week 4 test suite pattern.
"""
import pytest
import pandas as pd
from pathlib import Path
from inference_audit.checks.annotation_consistency import check_annotation_consistency

FIXTURES = Path("tests/fixtures")


class TestAnnotationConsistency:
    def test_high_confidence_scores_high(self):
        """High-confidence annotations (all > 0.7) must score >= 90."""
        df = pd.read_csv(FIXTURES / "high_confidence.csv")
        result = check_annotation_consistency(df, conf_col="confidence")
        assert result.score >= 90, (
            f"Expected score >= 90 for high_confidence fixture, got {result.score}. "
            f"Details: {result.details}"
        )
        assert result.warning is None

    def test_low_confidence_scores_in_expected_range(self):
        """
        Dataset with 30% low-confidence annotations must score 30-50,
        per Role Guide spec. Unlike missing_values and near_duplicates,
        this formula matches the Role Guide's expected range exactly --
        no discrepancy to flag here.
        """
        df = pd.read_csv(FIXTURES / "low_confidence.csv")
        result = check_annotation_consistency(df, conf_col="confidence")
        assert 30 <= result.score <= 50, (
            f"Expected score in 30-50 for low_confidence fixture, got {result.score}. "
            f"Details: {result.details}"
        )
        assert result.warning is not None

    def test_no_confidence_column_provided_is_graceful_skip(self):
        """
        conf_col=None is the EXPECTED, normal case for datasets without
        annotation confidence data -- must return score=None without
        treating it as an error condition.
        """
        df = pd.read_csv(FIXTURES / "high_confidence.csv")
        result = check_annotation_consistency(df, conf_col=None)
        assert result.score is None
        assert "skipped" in result.warning.lower()

    def test_missing_column_returns_none_score(self):
        """A specified but nonexistent confidence column must return score=None, never raise."""
        df = pd.read_csv(FIXTURES / "high_confidence.csv")
        result = check_annotation_consistency(df, conf_col="nonexistent_column")
        assert result.score is None
        assert result.warning is not None

    def test_empty_dataframe_returns_none_score(self):
        """An empty dataframe must return score=None, never raise or divide by zero."""
        df = pd.DataFrame({"confidence": []})
        result = check_annotation_consistency(df, conf_col="confidence")
        assert result.score is None
        assert result.warning is not None

    def test_invalid_threshold_above_one_returns_none(self):
        """confidence_threshold outside [0, 1] is meaningless -- must reject, not silently misbehave."""
        df = pd.read_csv(FIXTURES / "high_confidence.csv")
        result = check_annotation_consistency(df, conf_col="confidence", confidence_threshold=1.5)
        assert result.score is None
        assert "confidence_threshold" in result.warning

    def test_invalid_threshold_negative_returns_none(self):
        """A negative threshold is equally meaningless."""
        df = pd.read_csv(FIXTURES / "high_confidence.csv")
        result = check_annotation_consistency(df, conf_col="confidence", confidence_threshold=-0.1)
        assert result.score is None
        assert "confidence_threshold" in result.warning

    def test_null_confidence_values_reported_explicitly(self):
        """
        Null confidence values must be computed and reported explicitly
        (same discipline the director required for label_distribution),
        not silently dropped from the denominator without a trace.
        """
        df = pd.DataFrame({"confidence": [0.9, 0.8, None, 0.3, None, 0.95]})
        result = check_annotation_consistency(df, conf_col="confidence")
        assert result.details["null_count"] == 2
        assert result.details["non_null_count"] == 4
        assert "null" in result.warning.lower()

    def test_all_null_confidence_returns_none(self):
        """A confidence column that is entirely null has nothing to assess."""
        df = pd.DataFrame({"confidence": [None, None, None]})
        result = check_annotation_consistency(df, conf_col="confidence")
        assert result.score is None
        assert "null" in result.warning.lower()

    def test_non_numeric_values_returns_none(self):
        """
        Non-numeric values in the confidence column (e.g. strings) must
        not be silently miscompared against a numeric threshold --
        same discipline the director required for missing_values'
        non-string handling.
        """
        df = pd.DataFrame({"confidence": [0.9, "high", 0.3, "low"]})
        result = check_annotation_consistency(df, conf_col="confidence")
        assert result.score is None
        assert "non-numeric" in result.warning.lower()
        assert result.details["non_numeric_count"] == 2
