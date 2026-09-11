"""Tests for inference_audit/report.py"""

import json

import pytest

from inference_audit.report import CheckResult, AuditReport


def _make_report(checks):
    return AuditReport(
        dataset_path="dataset.csv",
        rows=100,
        label_col="label",
        text_col="text",
        audit_version="dev",
        timestamp="2026-01-01T00:00:00",
        checks=checks,
    )


def test_check_result_defaults_details_to_empty_dict():
    result = CheckResult(score=90, warning=None)
    assert result.details == {}


def test_overall_score_weights_label_distribution_and_near_duplicates_higher():
    # Two checks at 100, three at 0 -- label_distribution/near_duplicates
    # weighted 1.5x, the other three at 1.0x.
    checks = {
        "label_distribution": CheckResult(score=100, warning=None),
        "near_duplicates": CheckResult(score=100, warning=None),
        "language_contamination": CheckResult(score=0, warning=None),
        "missing_values": CheckResult(score=0, warning=None),
        "annotation_consistency": CheckResult(score=0, warning=None),
    }
    report = _make_report(checks)
    # weighted_sum = 100*1.5 + 100*1.5 + 0+0+0 = 300
    # total_weight = 1.5+1.5+1+1+1 = 6
    # 300/6 = 50
    assert report.overall_score == 50


def test_overall_score_excludes_skipped_checks_not_zero():
    checks = {
        "label_distribution": CheckResult(score=80, warning=None),
        "near_duplicates": CheckResult(score=None, warning="skipped"),
        "language_contamination": CheckResult(score=None, warning="skipped"),
        "missing_values": CheckResult(score=None, warning="skipped"),
        "annotation_consistency": CheckResult(score=None, warning="skipped"),
    }
    report = _make_report(checks)
    # Only label_distribution counted -- weighted average of just itself.
    assert report.overall_score == 80


def test_overall_score_is_none_when_all_checks_skipped():
    checks = {
        "label_distribution": CheckResult(score=None, warning="skipped"),
        "near_duplicates": CheckResult(score=None, warning="skipped"),
        "language_contamination": CheckResult(score=None, warning="skipped"),
        "missing_values": CheckResult(score=None, warning="skipped"),
        "annotation_consistency": CheckResult(score=None, warning="skipped"),
    }
    report = _make_report(checks)
    assert report.overall_score is None


def test_to_json_writes_valid_json_with_correct_structure(tmp_path):
    checks = {
        "label_distribution": CheckResult(
            score=61, warning="imbalance", details={"imbalance_ratio": 49.3}
        ),
    }
    report = _make_report(checks)
    output_path = tmp_path / "report.json"
    report.to_json(str(output_path))

    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset"] == "dataset.csv"
    assert data["overall_score"] == 61
    assert data["checks"]["label_distribution"]["score"] == 61
    # details are flattened into the check dict, not nested
    assert data["checks"]["label_distribution"]["imbalance_ratio"] == 49.3


def test_to_json_does_not_escape_non_ascii_text(tmp_path):
    checks = {
        "missing_values": CheckResult(
            score=90, warning=None, details={"sample": "یہ اردو ہے"}
        ),
    }
    report = _make_report(checks)
    output_path = tmp_path / "report.json"
    report.to_json(str(output_path))

    raw_text = output_path.read_text(encoding="utf-8")
    assert "یہ اردو ہے" in raw_text
    assert "\\u0627" not in raw_text  # would appear if ensure_ascii=True


def test_to_json_raises_clear_error_on_bad_path():
    checks = {"missing_values": CheckResult(score=90, warning=None)}
    report = _make_report(checks)
    with pytest.raises(OSError, match="Could not write JSON report"):
        report.to_json("/nonexistent_dir_xyz/report.json")


def test_save_delegates_to_report_renderer(monkeypatch, tmp_path):
    calls = {}

    def fake_render_html(report_obj, path):
        calls["report"] = report_obj
        calls["path"] = path

    import inference_audit.report_renderer as renderer_module
    monkeypatch.setattr(renderer_module, "render_html", fake_render_html)

    checks = {"missing_values": CheckResult(score=90, warning=None)}
    report = _make_report(checks)
    output_path = str(tmp_path / "report.html")
    report.save(output_path)

    assert calls["report"] is report
    assert calls["path"] == output_path