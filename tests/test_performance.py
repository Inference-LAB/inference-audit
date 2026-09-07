"""
tests/test_performance.py

Per Role Guide: full audit on a 100K-row dataset must complete in
under 60 seconds on CPU. Each check is timed individually (not just
the combined total) so a regression in any single check is identified
specifically, not just "the audit got slower."

DATASET REALISM: modeled on the Lead Engineer's finding from the real
134K-row RUEmoCorp corpus -- short social-media-style text repeats
heavily. This benchmark uses 100,000 rows drawn from a pool of 5,000
unique messages (5% uniqueness), not 100,000 fully-unique messages,
since that matches the actual target dataset's characteristics. A
"worst case" fully-unique scenario is tested separately below, since
performance depends meaningfully on how repetitive the real data is.
"""
import time
import random
import pandas as pd
import pytest
from inference_audit.checks.missing_values import check_missing_values
from inference_audit.checks.label_distribution import check_label_distribution
from inference_audit.checks.near_duplicates import check_near_duplicates
from inference_audit.checks.annotation_consistency import check_annotation_consistency
from inference_audit.checks.language_contamination import check_language_contamination

N = 100_000
TOTAL_BUDGET_SECONDS = 60

SUBJECTS = ["the weather", "my neighbor", "the football match", "this recipe", "the election results",
            "a new phone", "the movie plot", "traffic downtown", "the school exam", "my garden"]
VERBS = ["surprised everyone", "changed completely", "was disappointing", "went as expected", "caused chaos"]
DETAILS = ["according to reports", "based on what I saw", "much to our surprise", "after several delays"]


def _build_realistic_dataset(n=N, seed=42):
    """100K rows, 5,000 unique text values -- models real social-media
    repetition, per Lead Engineer's finding on the actual target corpus."""
    random.seed(seed)
    unique_pool = list(set(
        f"{random.choice(SUBJECTS)} {random.choice(VERBS)} {random.choice(DETAILS)} msg{i}"
        for i in range(5000)
    ))
    texts = [random.choice(unique_pool) for _ in range(n)]
    labels = (["class_A"] * int(n * 0.9)) + (["class_B"] * (n - int(n * 0.9)))
    random.shuffle(labels)
    confidences = [round(random.uniform(0.5, 1.0), 2) for _ in range(n)]
    return pd.DataFrame({"text": texts, "label": labels, "confidence": confidences})


class TestPerformance:
    def test_missing_values_under_budget(self):
        df = _build_realistic_dataset()
        t0 = time.time()
        result = check_missing_values(df, text_col="text")
        elapsed = time.time() - t0
        assert elapsed < TOTAL_BUDGET_SECONDS
        assert result.score is not None
        print(f"\ncheck_missing_values: {elapsed:.3f}s")

    def test_label_distribution_under_budget(self):
        df = _build_realistic_dataset()
        t0 = time.time()
        result = check_label_distribution(df, label_col="label")
        elapsed = time.time() - t0
        assert elapsed < TOTAL_BUDGET_SECONDS
        assert result.score is not None
        print(f"\ncheck_label_distribution: {elapsed:.3f}s")

    def test_annotation_consistency_under_budget(self):
        df = _build_realistic_dataset()
        t0 = time.time()
        result = check_annotation_consistency(df, conf_col="confidence")
        elapsed = time.time() - t0
        assert elapsed < TOTAL_BUDGET_SECONDS
        assert result.score is not None
        print(f"\ncheck_annotation_consistency: {elapsed:.3f}s")

    def test_near_duplicates_under_budget_realistic(self):
        """
        Regression test for the Week 5 performance fix: before the
        MinHash-caching + non-exhaustive-pairing optimizations, this
        check scaled worse than linearly (26.8s at just 20K rows,
        extrapolating well past 60s at 100K, and the full 100K run was
        killed under memory pressure during initial testing).
        """
        df = _build_realistic_dataset()
        t0 = time.time()
        result = check_near_duplicates(df, text_col="text")
        elapsed = time.time() - t0
        assert elapsed < TOTAL_BUDGET_SECONDS, (
            f"check_near_duplicates took {elapsed:.1f}s on 100K rows, "
            f"exceeding the {TOTAL_BUDGET_SECONDS}s budget"
        )
        assert result.score is not None
        print(f"\ncheck_near_duplicates: {elapsed:.3f}s")

    def test_language_contamination_under_budget_realistic(self):
        """
        Regression test for the deduplication-caching performance fix
        (measured 85.7x speedup on repetitive text during development).
        """
        df = _build_realistic_dataset()
        t0 = time.time()
        result = check_language_contamination(df, text_col="text")
        elapsed = time.time() - t0
        assert elapsed < TOTAL_BUDGET_SECONDS
        assert result.score is not None
        print(f"\ncheck_language_contamination: {elapsed:.3f}s")

    def test_full_audit_combined_under_60_seconds(self):
        """
        The actual Role Guide requirement: full audit (all 5 checks
        combined) on a 100K-row dataset must complete in under 60
        seconds on CPU. Uses ONE shared dataset across all 5 checks,
        matching how Auditor.audit() will actually run them together.
        """
        df = _build_realistic_dataset()
        t0 = time.time()
        r1 = check_missing_values(df, text_col="text")
        r2 = check_label_distribution(df, label_col="label")
        r3 = check_annotation_consistency(df, conf_col="confidence")
        r4 = check_near_duplicates(df, text_col="text")
        r5 = check_language_contamination(df, text_col="text")
        total_elapsed = time.time() - t0

        print(f"\n--- Full audit, 100K rows, combined ---")
        print(f"Total: {total_elapsed:.3f}s (budget: {TOTAL_BUDGET_SECONDS}s)")

        assert total_elapsed < TOTAL_BUDGET_SECONDS, (
            f"Full audit took {total_elapsed:.1f}s on 100K rows, "
            f"exceeding the {TOTAL_BUDGET_SECONDS}s budget"
        )
        for r in (r1, r2, r3, r4, r5):
            assert r.score is not None, f"A check returned score=None unexpectedly: {r.warning}"

    def test_near_duplicates_worst_case_fully_unique_text(self):
        """
        WORST-CASE scenario: if the real dataset turns out to have far
        less repetition than assumed (fully unique text), the
        MinHash-caching optimization provides no benefit (nothing to
        cache), so this measures the check's performance floor rather
        than assuming the realistic-repetition scenario always holds.
        Uses a smaller N (10K, not 100K) since this scenario is
        significantly more expensive and this test exists to establish
        a bound, not to re-run the full budget check.
        """
        random.seed(7)
        n = 10_000
        texts = [f"{random.choice(SUBJECTS)} {random.choice(VERBS)} {random.choice(DETAILS)} unique{i}" for i in range(n)]
        df = pd.DataFrame({"text": texts})
        t0 = time.time()
        result = check_near_duplicates(df, text_col="text")
        elapsed = time.time() - t0
        print(f"\ncheck_near_duplicates (10K rows, fully unique -- worst case): {elapsed:.3f}s")
        assert result.score is not None
        # No hard budget assertion here -- this test's purpose is to
        # SURFACE the worst-case number for the team to see, not to
        # gate the build on a scenario we don't yet know is realistic
        # for the actual corpus.
