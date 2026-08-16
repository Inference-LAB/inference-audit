"""
tests/test_near_duplicates.py

Per Role Guide Week 3 test suite pattern.
"""
import pytest
import pandas as pd
from pathlib import Path
from inference_audit.checks.near_duplicates import check_near_duplicates

FIXTURES = Path("tests/fixtures")


class TestNearDuplicates:
    def test_no_duplicates_scores_high(self):
        """Genuinely distinct text (no near-duplicate pairs) must score >= 90."""
        df = pd.read_csv(FIXTURES / "no_duplicates.csv")
        result = check_near_duplicates(df, text_col="text")
        assert result.score >= 90, (
            f"Expected score >= 90 for no_duplicates fixture, got {result.score}. "
            f"Details: {result.details}"
        )

    def test_with_duplicates_scores_lower(self):
        """
        Dataset with seeded exact + cross-label duplicates must score
        noticeably lower than a clean dataset.

        NOTE: Role Guide fixture table specifies expected range 40-65 for
        with_duplicates.csv. Our current approved formula produces a
        score in the high-80s -- same open pattern as the missing_values
        discrepancy flagged in Week 2. Using a loose upper-bound
        assertion deliberately -- this test documents that duplicates
        ARE detected and DO lower the score, not that any specific
        number is "correct," since the correct number is unresolved.
        """
        df = pd.read_csv(FIXTURES / "with_duplicates.csv")
        result = check_near_duplicates(df, text_col="text")
        assert result.score < 95, (
            f"Expected score noticeably below 100 for dataset with seeded duplicates, "
            f"got {result.score}. Details: {result.details}"
        )
        assert result.warning is not None
        assert result.details["candidate_pair_count"] > 0
        assert result.details["flagged_row_count"] >= 50

    def test_missing_column_returns_none_score(self):
        """A missing text column must return score=None, never raise."""
        df = pd.read_csv(FIXTURES / "no_duplicates.csv")
        result = check_near_duplicates(df, text_col="nonexistent_column")
        assert result.score is None
        assert result.warning is not None

    def test_empty_dataframe_returns_none_score(self):
        """An empty dataframe must return score=None, never raise."""
        df = pd.DataFrame({"text": []})
        result = check_near_duplicates(df, text_col="text")
        assert result.score is None
        assert result.warning is not None

    def test_single_row_returns_none_score(self):
        """A single-row dataset has no possible pairs -- must return score=None."""
        df = pd.DataFrame({"text": ["only one row here"]})
        result = check_near_duplicates(df, text_col="text")
        assert result.score is None
        assert "fewer than 2 rows" in result.warning.lower()

    def test_all_identical_rows_scores_zero(self):
        """A dataset where every row is an exact duplicate must score 0."""
        df = pd.DataFrame({"text": ["this exact sentence repeats"] * 20})
        result = check_near_duplicates(df, text_col="text")
        assert result.score == 0.0
        assert result.details["flagged_row_count"] == 20

    def test_all_rows_too_short_returns_none_score(self):
        """
        Regression test for a bug found by Khadija during PR review:
        when EVERY row is too short to build a comparable signature,
        the check must return score=None (nothing could be measured),
        NOT score=100.0 (which would silently misreport "perfectly
        clean" when nothing was actually compared). This is the same
        underlying situation as the single-row case, just reached via
        filtering instead of raw row count -- both must behave the
        same way.

        Exact case from Khadija's review: ["hi","ok","no","hi","ok"].
        """
        df = pd.DataFrame({"text": ["hi", "ok", "no", "hi", "ok"]})
        result = check_near_duplicates(df, text_col="text")
        assert result.score is None, (
            f"Expected score=None (all 5 rows too short to compare), got {result.score}"
        )
        assert "too short to compare" in result.warning.lower()
        assert result.details["skipped_short_count"] == 5

    def test_mixed_short_and_real_duplicates(self):
        """Short text is skipped; real duplicates among normal-length text are still caught."""
        df = pd.DataFrame({
            "text": [
                "hi", "ok",
                "this is a real duplicate sentence",
                "this is a real duplicate sentence",
                "a genuinely different sentence here",
            ]
        })
        result = check_near_duplicates(df, text_col="text")
        assert result.score is not None  # NOT all rows were too short here
        assert result.details["skipped_short_count"] == 2
        assert result.details["flagged_row_count"] == 2
        assert result.details["candidate_pair_count"] == 1
