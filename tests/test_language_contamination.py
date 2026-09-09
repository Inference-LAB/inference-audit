"""
tests/test_language_contamination.py
"""
import pytest
import pandas as pd
from pathlib import Path
from inference_audit.checks.language_contamination import check_language_contamination
from langdetect import detect_langs

FIXTURES = Path("tests/fixtures")

# Measured, documented baseline for the known false-positive rate on
# clean Roman Urdu text (langdetect confidently misdetecting genuine
# Roman Urdu as English) -- plus a tolerance for run-to-run/environment
# variation, rather than an arbitrary implementation-dependent range.
# If this baseline drifts significantly on re-measurement, that's a
# signal worth investigating (langdetect version change, fixture
# change), not just a number to silently widen.
EXPECTED_FALSE_POSITIVE_RATE = 0.116
FALSE_POSITIVE_RATE_TOLERANCE = 0.05


class TestLanguageContamination:
    def test_concern_language_catches_real_contamination(self):
        texts = ["This is a normal English sentence about daily life."] * 8 + \
                ["Ceci est une phrase française tout a fait normale."] * 2
        df = pd.DataFrame({"text": texts})
        result = check_language_contamination(df, text_col="text", concern_languages=["fr"])
        assert result.score == 80.0
        assert result.details["contamination_count"] == 2

    def test_stability_across_dataset_size(self):
        df_small = pd.read_csv(FIXTURES / "clean_urdu.csv")
        result_small = check_language_contamination(df_small, text_col="text")
        import random
        random.seed(99)
        samples = ["yeh bohat acha din tha aj", "mera dil khush hai", "kya haal hai bhai",
                   "bohat mazedar khana tha", "yeh drama bilkul boring hai", "aj mausam acha hai",
                   "mujhe yeh pasand nahi aya", "sab theek hai alhamdulillah"]
        df_large = pd.DataFrame({"text": [f"{random.choice(samples)} number {i}" for i in range(5000)]})
        result_large = check_language_contamination(df_large, text_col="text")
        assert abs(result_small.score - result_large.score) < 5

    def test_false_positive_rate_within_documented_tolerance(self):
        """
        Per review: replaced the previous arbitrary score-range
        assertion (80-95) with a direct assertion on the actual
        false-positive RATE against a documented, measured baseline
        plus tolerance -- this is what the check's known limitation
        docstring describes, made concrete and testable rather than
        an indirect range on the derived score.
        """
        df = pd.read_csv(FIXTURES / "clean_urdu.csv")
        result = check_language_contamination(df, text_col="text")
        measured_rate = result.details["contamination_rate"]
        assert abs(measured_rate - EXPECTED_FALSE_POSITIVE_RATE) <= FALSE_POSITIVE_RATE_TOLERANCE, (
            f"Expected false-positive rate {EXPECTED_FALSE_POSITIVE_RATE} +/- "
            f"{FALSE_POSITIVE_RATE_TOLERANCE}, got {measured_rate}. If langdetect's "
            f"behavior genuinely changed (version update), re-measure and update "
            f"EXPECTED_FALSE_POSITIVE_RATE deliberately rather than widening tolerance."
        )

    def test_contaminated_dataset_scores_lower_than_clean(self):
        df_clean = pd.read_csv(FIXTURES / "clean_urdu.csv")
        df_contaminated = pd.read_csv(FIXTURES / "contaminated.csv")
        result_clean = check_language_contamination(df_clean, text_col="text")
        result_contaminated = check_language_contamination(df_contaminated, text_col="text")
        assert result_contaminated.score < result_clean.score

    def test_missing_column_returns_none_score(self):
        result = check_language_contamination(pd.DataFrame({"text": ["a"]}), text_col="nonexistent")
        assert result.score is None

    def test_empty_dataframe_returns_none_score(self):
        result = check_language_contamination(pd.DataFrame({"text": []}), text_col="text")
        assert result.score is None

    def test_invalid_confidence_threshold_returns_none(self):
        result = check_language_contamination(
            pd.DataFrame({"text": ["hello world"]}), text_col="text", confidence_threshold=1.5
        )
        assert result.score is None

    def test_empty_concern_languages_returns_none(self):
        result = check_language_contamination(
            pd.DataFrame({"text": ["hello world"]}), text_col="text", concern_languages=[]
        )
        assert result.score is None

    def test_all_text_undetectable_returns_none(self):
        df = pd.DataFrame({"text": ["", "   ", "123", None]})
        result = check_language_contamination(df, text_col="text")
        assert result.score is None
        assert result.details["skipped_undetectable_count"] == 4

    def test_deduplication_reported_in_details(self):
        df = pd.DataFrame({"text": ["repeated message here"] * 100})
        result = check_language_contamination(df, text_col="text")
        assert result.details["total_rows"] == 100
        assert result.details["unique_text_count"] == 1

    def test_boundary_exact_threshold_counts_as_match(self):
        """
        Per review: made the >= boundary explicit and directly tested.
        A detection with confidence EXACTLY equal to confidence_threshold
        must count as a match (inclusive boundary), not be excluded.
        """
        english_text = "This is a normal English sentence about daily life today."
        exact_prob = detect_langs(english_text)[0].prob
        df = pd.DataFrame({"text": [english_text] * 5})
        result = check_language_contamination(
            df, text_col="text", concern_languages=["en"], confidence_threshold=exact_prob
        )
        assert result.details["contamination_count"] == 5, (
            "A detection exactly at confidence_threshold must count as a match (>=)"
        )

    def test_defaults_come_from_config(self):
        """Confirms defaults are sourced from config.py, not code-embedded literals."""
        from inference_audit.config import LANGUAGE_CONCERN_LANGUAGES, LANGUAGE_CONFIDENCE_THRESHOLD
        df = pd.read_csv(FIXTURES / "clean_urdu.csv")
        result = check_language_contamination(df, text_col="text")
        assert result.details["concern_languages"] == sorted(LANGUAGE_CONCERN_LANGUAGES)
        assert result.details["confidence_threshold"] == LANGUAGE_CONFIDENCE_THRESHOLD
