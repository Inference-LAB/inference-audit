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
