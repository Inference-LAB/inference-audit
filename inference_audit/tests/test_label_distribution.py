"""
tests/test_label_distribution.py

Per Role Guide Week 3 test suite pattern.
"""
import pytest
import pandas as pd
from pathlib import Path
from inference_audit.checks.label_distribution import check_label_distribution

FIXTURES = Path("tests/fixtures")


class TestLabelDistribution:
    def test_balanced_dataset_scores_high(self):
        """Perfectly balanced 5-class dataset must score >= 90 (design doc: 1:1 -> 100)."""
        df = pd.read_csv(FIXTURES / "balanced.csv")
        result = check_label_distribution(df, label_col="label")
        assert result.score >= 90, (
            f"Expected score >= 90 for balanced dataset, got {result.score}. "
            f"Details: {result.details}"
        )
        assert result.warning is None

    def test_severe_imbalance_scores_low(self):
        """950:50 class split (19:1 ratio) must score <= 30 per Role Guide spec."""
        df = pd.read_csv(FIXTURES / "severe_imbalance.csv")
        result = check_label_distribution(df, label_col="label")
        assert result.score <= 30, (
            f"Expected score <= 30 for severely imbalanced dataset, got {result.score}. "
            f"Details: {result.details}"
        )
        assert result.warning is not None
        assert result.details["imbalance_ratio"] > 15

    def test_missing_column_returns_none_score(self):
        """A missing label column must return score=None, never raise."""
        df = pd.read_csv(FIXTURES / "balanced.csv")
        result = check_label_distribution(df, label_col="nonexistent_column")
        assert result.score is None
        assert result.warning is not None

    def test_single_class_returns_none_score(self):
        """A dataset with only one class present has undefined imbalance -- must not crash."""
        df = pd.DataFrame({"label": ["only_one"] * 10})
        result = check_label_distribution(df, label_col="label")
        assert result.score is None
        assert "one class" in result.warning.lower()

    def test_empty_dataframe_returns_none_score(self):
        """An empty dataframe must return score=None, never raise or divide by zero."""
        df = pd.DataFrame({"label": []})
        result = check_label_distribution(df, label_col="label")
        assert result.score is None
        assert result.warning is not None

    def test_max_acceptable_ratio_equals_one_returns_none(self):
        """max_acceptable_ratio == 1.0 leaves no scoring range -- must reject, not crash."""
        df = pd.read_csv(FIXTURES / "balanced.csv")
        result = check_label_distribution(df, label_col="label", max_acceptable_ratio=1.0)
        assert result.score is None
        assert "max_acceptable_ratio" in result.warning

    def test_null_labels_mixed_with_valid_labels(self):
        """Null labels must be reported explicitly, not silently dropped from the picture."""
        df = pd.DataFrame({"label": ["a"] * 40 + ["b"] * 40 + [None] * 20})
        result = check_label_distribution(df, label_col="label")
        assert result.details["null_count"] == 20
        assert result.details["total_rows"] == 100
        # majority + minority should NOT silently equal total_rows when nulls exist
        assert result.details["majority_count"] + result.details["minority_count"] == 80
        assert "null" in result.warning.lower()

    def test_single_class_with_nulls_reports_null_count(self):
        """Single-class + nulls must mention the null count, not just the single class."""
        df = pd.DataFrame({"label": ["only_one"] * 30 + [None] * 10})
        result = check_label_distribution(df, label_col="label")
        assert result.score is None
        assert result.details["null_count"] == 10
        assert "10" in result.warning and "null" in result.warning.lower()
