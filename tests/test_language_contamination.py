"""
tests/test_language_contamination.py

Per Role Guide Week 4 test suite pattern.
"""
import pytest
import pandas as pd
from pathlib import Path
from inference_audit.checks.language_contamination import check_language_contamination

FIXTURES = Path("tests/fixtures")


class TestLanguageContamination:
    def test_default_concern_languages_flags_real_english_contamination(self):
        """
        concern_languages=["fr"] against an English dataset with real
        French contamination must correctly identify it -- this is the
        same guarantee the old "specific language" mode provided,
        preserved under the new API.
        """
        texts = ["This is a normal English sentence about daily life."] * 8 + \
                ["Ceci est une phrase française tout a fait normale."] * 2
        df = pd.DataFrame({"text": texts})
        result = check_language_contamination(df, text_col="text", concern_languages=["fr"])
        assert result.score == 80.0
        assert result.details["contamination_count"] == 2

    def test_stability_across_dataset_size(self):
        """
        Regression test for the Lead Engineer's core Week 4 finding:
        the OLD majority-vote "auto" mode produced wildly different
        scores depending on dataset size (10% on a small sample, 28.9%
        on the real 134K-row corpus) because the baseline itself was
        computed from noisy data. The NEW concern-list approach must
        produce a STABLE score regardless of sample size, since it
        checks against a fixed list, not a data-derived baseline.
        """
        df_small = pd.read_csv(FIXTURES / "clean_urdu.csv")  # 500 rows
        result_small = check_language_contamination(df_small, text_col="text")

        # A much larger, independently-generated sample of the same
        # underlying clean Roman Urdu content.
        import random
        random.seed(99)
        samples = ["yeh bohat acha din tha aj", "mera dil khush hai", "kya haal hai bhai",
                   "bohat mazedar khana tha", "yeh drama bilkul boring hai", "aj mausam acha hai",
                   "mujhe yeh pasand nahi aya", "sab theek hai alhamdulillah"]
        df_large = pd.DataFrame({"text": [f"{random.choice(samples)} number {i}" for i in range(5000)]})
        result_large = check_language_contamination(df_large, text_col="text")

        # Should be close (within a few points), NOT a 3x swing like
        # the old majority-vote approach exhibited.
        assert abs(result_small.score - result_large.score) < 5, (
            f"Expected stable score across dataset sizes, got {result_small.score} "
            f"(500 rows) vs {result_large.score} (5000 rows) -- difference too large, "
            f"suggests instability has crept back in."
        )

    def test_known_bounded_false_positive_rate_on_clean_roman_urdu(self):
        """
        DOCUMENTED, KNOWN LIMITATION (not a bug): genuine Roman Urdu
        text can still be confidently misdetected as English at a real,
        measured rate (~11-12%). This test documents that the rate is
        BOUNDED and roughly consistent, not that it is zero -- a
        perfect score here would indicate the test fixture changed in
        a way that hides this real, understood limitation rather than
        the limitation being solved.
        """
        df = pd.read_csv(FIXTURES / "clean_urdu.csv")
        result = check_language_contamination(df, text_col="text")
        assert 80 <= result.score <= 95, (
            f"Expected score in the known bounded range 80-95 for clean Roman Urdu "
            f"(reflecting the documented ~11-12% false-positive rate), got {result.score}"
        )

    def test_contaminated_dataset_scores_lower_than_clean(self):
        """A dataset with real seeded English contamination must score meaningfully lower than clean."""
        df_clean = pd.read_csv(FIXTURES / "clean_urdu.csv")
        df_contaminated = pd.read_csv(FIXTURES / "contaminated.csv")
        result_clean = check_language_contamination(df_clean, text_col="text")
        result_contaminated = check_language_contamination(df_contaminated, text_col="text")
        assert result_contaminated.score < result_clean.score

    def test_missing_column_returns_none_score(self):
        """A missing text column must return score=None, never raise."""
        result = check_language_contamination(pd.DataFrame({"text": ["a"]}), text_col="nonexistent")
        assert result.score is None
        assert result.warning is not None

    def test_empty_dataframe_returns_none_score(self):
        """An empty dataframe must return score=None, never raise."""
        result = check_language_contamination(pd.DataFrame({"text": []}), text_col="text")
        assert result.score is None
        assert result.warning is not None

    def test_invalid_confidence_threshold_returns_none(self):
        """confidence_threshold outside [0, 1] is meaningless -- must reject."""
        result = check_language_contamination(
            pd.DataFrame({"text": ["hello world"]}), text_col="text", confidence_threshold=1.5
        )
        assert result.score is None
        assert "confidence_threshold" in result.warning

    def test_empty_concern_languages_returns_none(self):
        """An empty concern_languages list has nothing to check against -- must reject, not silently pass everything."""
        result = check_language_contamination(
            pd.DataFrame({"text": ["hello world"]}), text_col="text", concern_languages=[]
        )
        assert result.score is None
        assert "concern_languages" in result.warning

    def test_all_text_undetectable_returns_none(self):
        """langdetect raises on empty/whitespace/numeric-only text -- must not crash."""
        df = pd.DataFrame({"text": ["", "   ", "123", None]})
        result = check_language_contamination(df, text_col="text")
        assert result.score is None
        assert result.details["skipped_undetectable_count"] == 4

    def test_deduplication_reduces_unique_text_count_reported(self):
        """
        Performance fix verification: details must report unique_text_count
        separately from total_rows, confirming the dedup optimization is
        active (per Lead Engineer's performance finding on the 134K-row
        corpus with heavy short-message repetition).
        """
        df = pd.DataFrame({"text": ["repeated message here"] * 100})
        result = check_language_contamination(df, text_col="text")
        assert result.details["total_rows"] == 100
        assert result.details["unique_text_count"] == 1
