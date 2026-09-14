"""Tests for inference_audit/report_renderer.py"""

import base64

import pytest

from inference_audit.report import AuditReport, CheckResult
from inference_audit.report_renderer import render_html, _build_score_chart


def _make_report(checks=None):
    if checks is None:
        checks = {
            "label_distribution": CheckResult(score=61, warning="imbalance"),
            "near_duplicates": CheckResult(score=90, warning=None),
            "language_contamination": CheckResult(score=95, warning=None),
            "missing_values": CheckResult(score=88, warning="some flagged"),
            "annotation_consistency": CheckResult(score=None, warning="skipped"),
        }
    return AuditReport(
        dataset_path="dataset.csv",
        rows=100,
        label_col="label",
        text_col="text",
        audit_version="dev",
        timestamp="2026-01-01T00:00:00",
        checks=checks,
    )


def test_build_score_chart_returns_valid_base64_png():
    report = _make_report()
    chart_b64 = _build_score_chart(report.checks)

    # A valid PNG file starts with this exact byte signature once decoded.
    decoded = base64.b64decode(chart_b64)
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_build_score_chart_handles_skipped_checks_without_crashing():
    checks = {
        "label_distribution": CheckResult(score=None, warning="skipped"),
        "near_duplicates": CheckResult(score=None, warning="skipped"),
    }
    # Should not raise, even with every check skipped.
    chart_b64 = _build_score_chart(checks)
    assert len(chart_b64) > 0


def test_render_html_writes_a_file(tmp_path):
    report = _make_report()
    output_path = tmp_path / "report.html"
    render_html(report, str(output_path))

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "dataset.csv" in content


def test_render_html_embeds_chart_as_base64_image(tmp_path):
    report = _make_report()
    output_path = tmp_path / "report.html"
    render_html(report, str(output_path))

    content = output_path.read_text(encoding="utf-8")
    assert "data:image/png;base64," in content


def test_render_html_shows_skipped_check_distinctly(tmp_path):
    report = _make_report()
    output_path = tmp_path / "report.html"
    render_html(report, str(output_path))

    content = output_path.read_text(encoding="utf-8")
    assert "Skipped" in content


def test_render_html_raises_clear_error_on_bad_path():
    report = _make_report()
    with pytest.raises(OSError, match="Could not write HTML report"):
        render_html(report, "/nonexistent_dir_xyz/report.html")


def test_render_html_has_no_external_dependencies(tmp_path):
    """Design doc requirement: the HTML report must be self-contained,
    no external CSS/JS links."""
    report = _make_report()
    output_path = tmp_path / "report.html"
    render_html(report, str(output_path))

    content = output_path.read_text(encoding="utf-8")
    assert "<link " not in content  # no external stylesheet links
    assert "src=\"http" not in content  # no externally-hosted scripts/images