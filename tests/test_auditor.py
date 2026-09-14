"""Tests for inference_audit/auditor.py

Uses monkeypatch to stub out load_dataset() and all five check
functions, so these tests verify ORCHESTRATION (does Auditor call the
right things, with the right arguments, and assemble the result
correctly) rather than re-testing each check's own logic, which
already has its own dedicated test file.
"""

import pandas as pd
import pytest

from inference_audit.auditor import Auditor
from inference_audit.report import CheckResult
from inference_audit.config import LANGUAGE_CONCERN_LANGUAGES


@pytest.fixture
def stub_checks(monkeypatch):
    """Replaces every check function Auditor.audit() calls with a stub
    that records how it was called and returns a fixed CheckResult."""
    calls = {}

    def make_stub(name, score=75):
        def stub(*args, **kwargs):
            calls[name] = {"args": args, "kwargs": kwargs}
            return CheckResult(score=score, warning=None, details={})
        return stub

    fake_df = pd.DataFrame({"label": ["a", "b"], "text": ["x", "y"]})
    monkeypatch.setattr(
        "inference_audit.auditor.load_dataset",
        lambda path, label_col, text_col: fake_df,
    )
    monkeypatch.setattr(
        "inference_audit.auditor.check_label_distribution",
        make_stub("label_distribution"),
    )
    monkeypatch.setattr(
        "inference_audit.auditor.check_near_duplicates",
        make_stub("near_duplicates"),
    )
    monkeypatch.setattr(
        "inference_audit.auditor.check_language_contamination",
        make_stub("language_contamination"),
    )
    monkeypatch.setattr(
        "inference_audit.auditor.check_missing_values",
        make_stub("missing_values"),
    )
    monkeypatch.setattr(
        "inference_audit.auditor.check_annotation_consistency",
        make_stub("annotation_consistency"),
    )
    return calls


def test_audit_returns_report_with_all_five_checks(stub_checks):
    auditor = Auditor()
    report = auditor.audit("dataset.csv", label_col="label", text_col="text")

    assert set(report.checks.keys()) == {
        "label_distribution", "near_duplicates", "language_contamination",
        "missing_values", "annotation_consistency",
    }
    assert report.rows == 2
    assert report.dataset_path == "dataset.csv"


def test_audit_uses_default_concern_languages_when_not_provided(stub_checks):
    auditor = Auditor()
    auditor.audit("dataset.csv", label_col="label", text_col="text")

    call = stub_checks["language_contamination"]
    assert call["kwargs"]["concern_languages"] == LANGUAGE_CONCERN_LANGUAGES


def test_audit_uses_provided_concern_languages_when_given(stub_checks):
    auditor = Auditor()
    auditor.audit(
        "dataset.csv", label_col="label", text_col="text",
        concern_languages=("es", "fr"),
    )

    call = stub_checks["language_contamination"]
    assert call["kwargs"]["concern_languages"] == ("es", "fr")


def test_audit_passes_through_explicit_empty_concern_languages(stub_checks):
    """An explicitly empty tuple should NOT be silently replaced with
    the default -- only None (not provided) triggers the default."""
    auditor = Auditor()
    auditor.audit(
        "dataset.csv", label_col="label", text_col="text",
        concern_languages=(),
    )

    call = stub_checks["language_contamination"]
    assert call["kwargs"]["concern_languages"] == ()


def test_audit_passes_conf_col_to_annotation_consistency(stub_checks):
    auditor = Auditor()
    auditor.audit(
        "dataset.csv", label_col="label", text_col="text", conf_col="confidence",
    )

    call = stub_checks["annotation_consistency"]
    assert call["kwargs"]["conf_col"] == "confidence"


def test_audit_propagates_file_not_found(monkeypatch):
    def raise_not_found(path, label_col, text_col):
        raise FileNotFoundError(f"no such file: {path}")

    monkeypatch.setattr("inference_audit.auditor.load_dataset", raise_not_found)

    auditor = Auditor()
    with pytest.raises(FileNotFoundError):
        auditor.audit("missing.csv", label_col="label", text_col="text")


def test_audit_propagates_value_error(monkeypatch):
    def raise_value_error(path, label_col, text_col):
        raise ValueError("bad column")

    monkeypatch.setattr("inference_audit.auditor.load_dataset", raise_value_error)

    auditor = Auditor()
    with pytest.raises(ValueError):
        auditor.audit("dataset.csv", label_col="wrong", text_col="text")