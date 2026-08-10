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
        with_duplicates.csv (50 exact + 10 cross-label duplicates / 500
        rows = 12% seeded duplication). Our current approved formula
        (100 x (1 - flagged_row_rate)) produces a score in the high-80s
        for this fixture (observed 78.2-87.2 across fixture regenerations,
        since exact score shifts slightly with the random content used
        for the non-duplicate rows) -- not 40-65. Same pattern as the
        missing_values discrepancy flagged in Week 2. Flagged to
        Khadija/director as a combined open question (both checks show
        the linear formula scoring more leniently than the Role Guide's
        illustrative ranges expect). Using a loose upper-bound assertion
        here deliberately -- this test documents that duplicates ARE
        detected and DO lower the score, not that any specific number is
        "correct," since the correct number is exactly what's unresolved.
        """
        df = pd.read_csv(FIXTURES / "with_duplicates.csv")
        result = check_near_duplicates(df, text_col="text")
        assert result.score < 95, (
            f"Expected score noticeably below 100 for dataset with seeded duplicates, "
            f"got {result.score}. Details: {result.details}"
        )
        assert result.warning is not None
        assert result.details["candidate_pair_count"] > 0
        assert result.details["flagged_row_count"] >= 50  # the 50 exact duplicates must all be caught

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

    def test_short_text_does_not_produce_false_positives(self):
        """
        Regression test for a real bug found during Week 3 development:
        text shorter than 3 characters produces zero 3-grams, so
        MinHash.update() is never called and the signature stays at its
        default state -- meaning any two short texts would trivially
        match each other regardless of actual content ("hi" and "no"
        were incorrectly flagged as duplicates before this fix). Short
        rows must be excluded from comparison, not silently mismatched.
        """
        df = pd.DataFrame({"text": ["hi", "ok", "no", "yo", "go"]})
        result = check_near_duplicates(df, text_col="text")
        assert result.score == 100.0, (
            f"Expected score 100 (all rows too short to compare, none "
            f"should falsely match), got {result.score}"
        )
        assert result.details["skipped_short_count"] == 5
        assert result.details["candidate_pair_count"] == 0

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
        assert result.details["skipped_short_count"] == 2
        assert result.details["flagged_row_count"] == 2
        assert result.details["candidate_pair_count"] == 1
