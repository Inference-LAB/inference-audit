"""Tests for inference_audit/cli.py

Uses Typer's CliRunner to invoke the CLI as a subprocess-like call
without actually spawning a process. Auditor.audit() is monkeypatched
to return a fixed AuditReport, so these tests verify CLI behavior
(argument parsing, validation, exit codes, output routing) in
isolation from the real audit pipeline, which already has its own
tests elsewhere.
"""

import pytest
from typer.testing import CliRunner

from inference_audit.cli import app
from inference_audit.report import AuditReport, CheckResult

runner = CliRunner()


def get_output(result):
    """Combines stdout and stderr into one string for assertions.

    typer.echo(..., err=True) writes to stderr, which result.stdout
    does not include in newer Click/Typer versions (older versions
    merged the two streams by default; current versions keep them
    separate). Error-path tests need to check both, since the CLI
    correctly writes user-facing errors to stderr, not stdout.
    """
    text = result.stdout or ""
    try:
        if result.stderr:
            text += result.stderr
    except ValueError:
        # Some CliRunner configurations don't capture stderr separately
        # at all, in which case it's already included in result.stdout.
        pass
    return text


def _fake_report(overall_score=75, path="dataset.csv"):
    return AuditReport(
        dataset_path=path,
        rows=10,
        label_col="label",
        text_col="text",
        audit_version="dev",
        timestamp="2026-01-01T00:00:00",
        checks={"missing_values": CheckResult(score=overall_score, warning=None)},
    )


@pytest.fixture
def fake_csv(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_text("label,text\na,hello\nb,world\n")
    return str(path)


def test_run_requires_run_subcommand():
    """Confirms the callback fix is actually in place -- Typer should
    NOT collapse to a single implicit command."""
    result = runner.invoke(app, ["--help"])
    assert "run" in result.stdout


def test_missing_required_options_fails(fake_csv):
    result = runner.invoke(app, ["run", fake_csv])
    assert result.exit_code != 0


def test_successful_run_writes_html_by_default(monkeypatch, fake_csv, tmp_path):
    monkeypatch.setattr(
        "inference_audit.cli.Auditor.audit",
        lambda self, *a, **k: _fake_report(),
    )
    output_path = tmp_path / "out.html"
    result = runner.invoke(app, [
        "run", fake_csv,
        "--label-col", "label", "--text-col", "text",
        "--output", str(output_path),
    ])
    assert result.exit_code == 0
    assert "Overall score" in result.stdout


def test_json_output_routes_to_to_json(monkeypatch, fake_csv, tmp_path):
    calls = {}
    monkeypatch.setattr(
        "inference_audit.cli.Auditor.audit",
        lambda self, *a, **k: _fake_report(),
    )
    monkeypatch.setattr(
        AuditReport, "to_json", lambda self, path: calls.setdefault("to_json", path)
    )
    monkeypatch.setattr(
        AuditReport, "save", lambda self, path: calls.setdefault("save", path)
    )

    output_path = tmp_path / "out.json"
    result = runner.invoke(app, [
        "run", fake_csv,
        "--label-col", "label", "--text-col", "text",
        "--output", str(output_path),
    ])
    assert result.exit_code == 0
    assert "to_json" in calls
    assert "save" not in calls


def test_invalid_output_extension_rejected(fake_csv, tmp_path):
    result = runner.invoke(app, [
        "run", fake_csv,
        "--label-col", "label", "--text-col", "text",
        "--output", str(tmp_path / "out.txt"),
    ])
    assert result.exit_code == 1
    assert "must end in" in get_output(result)


def test_fail_below_out_of_range_rejected(fake_csv):
    result = runner.invoke(app, [
        "run", fake_csv,
        "--label-col", "label", "--text-col", "text",
        "--fail-below", "150",
    ])
    assert result.exit_code == 1
    assert "between 0 and 100" in get_output(result)


def test_fail_below_triggers_exit_1_when_score_low(monkeypatch, fake_csv, tmp_path):
    monkeypatch.setattr(
        "inference_audit.cli.Auditor.audit",
        lambda self, *a, **k: _fake_report(overall_score=40),
    )
    result = runner.invoke(app, [
        "run", fake_csv,
        "--label-col", "label", "--text-col", "text",
        "--output", str(tmp_path / "out.html"),
        "--fail-below", "70",
    ])
    assert result.exit_code == 1
    assert "below threshold" in result.stdout


def test_fail_below_passes_when_score_high(monkeypatch, fake_csv, tmp_path):
    monkeypatch.setattr(
        "inference_audit.cli.Auditor.audit",
        lambda self, *a, **k: _fake_report(overall_score=95),
    )
    result = runner.invoke(app, [
        "run", fake_csv,
        "--label-col", "label", "--text-col", "text",
        "--output", str(tmp_path / "out.html"),
        "--fail-below", "70",
    ])
    assert result.exit_code == 0


def test_invalid_concern_language_code_rejected(fake_csv, tmp_path):
    result = runner.invoke(app, [
        "run", fake_csv,
        "--label-col", "label", "--text-col", "text",
        "--output", str(tmp_path / "out.html"),
        "--concern-languages", "english,hi",  # "english" is not a valid 2-letter code
    ])
    assert result.exit_code == 1
    assert "invalid language code" in get_output(result)


def test_valid_concern_languages_parsed_and_passed(monkeypatch, fake_csv, tmp_path):
    captured = {}

    def fake_audit(self, path, label_col, text_col, conf_col=None, concern_languages=None):
        captured["concern_languages"] = concern_languages
        return _fake_report()

    monkeypatch.setattr("inference_audit.cli.Auditor.audit", fake_audit)
    result = runner.invoke(app, [
        "run", fake_csv,
        "--label-col", "label", "--text-col", "text",
        "--output", str(tmp_path / "out.html"),
        "--concern-languages", "en,hi",
    ])
    assert result.exit_code == 0
    assert captured["concern_languages"] == ("en", "hi")


def test_file_not_found_gives_clean_error_not_traceback(fake_csv, tmp_path):
    result = runner.invoke(app, [
        "run", "does_not_exist.csv",
        "--label-col", "label", "--text-col", "text",
        "--output", str(tmp_path / "out.html"),
    ])
    assert result.exit_code == 1
    assert "Traceback" not in get_output(result)
    assert "Error" in get_output(result)


def test_unexpected_exception_is_caught_at_cli_boundary(monkeypatch, fake_csv, tmp_path):
    def raise_unexpected(self, *a, **k):
        raise RuntimeError("something unrelated broke")

    monkeypatch.setattr("inference_audit.cli.Auditor.audit", raise_unexpected)
    result = runner.invoke(app, [
        "run", fake_csv,
        "--label-col", "label", "--text-col", "text",
        "--output", str(tmp_path / "out.html"),
    ])
    assert result.exit_code == 1
    assert "Unexpected error" in get_output(result)
    assert "Traceback (most recent call last)" not in get_output(result)


def test_all_checks_skipped_warns_instead_of_crashing_on_fail_below(monkeypatch, fake_csv, tmp_path):
    report_with_no_score = AuditReport(
        dataset_path=fake_csv, rows=10, label_col="label", text_col="text",
        audit_version="dev", timestamp="2026-01-01T00:00:00",
        checks={"missing_values": CheckResult(score=None, warning="skipped")},
    )
    monkeypatch.setattr(
        "inference_audit.cli.Auditor.audit",
        lambda self, *a, **k: report_with_no_score,
    )
    result = runner.invoke(app, [
        "run", fake_csv,
        "--label-col", "label", "--text-col", "text",
        "--output", str(tmp_path / "out.html"),
        "--fail-below", "70",
    ])
    assert "cannot compare against --fail-below" in result.stdout