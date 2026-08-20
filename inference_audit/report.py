"""
inference_audit/report.py

Defines the two data structures every part of inference-audit talks to:

- CheckResult: what a single quality check returns.
- AuditReport: what a full audit run returns a collection of
  CheckResults plus dataset metadata, with the logic to compute an
  overall score and write output files.

Nothing else in this codebase should invent its own way of representing
a check's outcome. If it doesn't fit into a CheckResult, it doesn't
belong in a check function.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import datetime
from importlib.metadata import version, PackageNotFoundError


@dataclass
class CheckResult:
    """
    The result of one quality check run against a dataset.

    score:   A number from 0 to 100, where 100 means "no issue found."
             This is None when the check could not run at all — for
             example, a confidence column wasn't provided. A skipped
             check is not the same as a failed check, and callers
             should treat None as "not measured," not as "zero."

    warning: A short, human-readable sentence describing what's wrong,
             or None if the check found nothing worth flagging.

    details: A dictionary of check-specific numbers a developer might
             want to inspect — e.g. how many rows were affected, what
             the raw ratio was before it got converted into a score.
             Keeping these separate from `score` and `warning` means
             every check can report whatever numbers make sense for it
             without changing the shape of CheckResult itself.
    """

    score: Optional[int]
    warning: Optional[str]
    details: dict = field(default_factory=dict)


@dataclass
class AuditReport:
    """
    The full result of running an audit on a dataset.

    This object is built by Auditor.audit() — nothing else should
    construct one by hand. It holds one CheckResult per quality check,
    plus enough metadata (which file, which columns, when) that the
    report is meaningful on its own, without needing to know what
    command produced it.
    """

    dataset_path: str
    rows: int
    label_col: str
    text_col: str
    audit_version: str
    timestamp: str
    checks: dict[str, CheckResult]

    # How much each check counts toward the overall score.
    # label_distribution and near_duplicates count for more because,
    # historically, they're the two issues most likely to get a paper
    # rejected,RUEMO Corpus faced the same issue.
    _WEIGHTS = {
        "label_distribution": 1.5,
        "near_duplicates": 1.5,
        "language_contamination": 1.0,
        "missing_values": 1.0,
        "annotation_consistency": 1.0,
    }

    @property
    def overall_score(self) -> Optional[int]:
        """
        A single number summarizing the whole audit, weighted by
        how much each check matters.

        Checks that were skipped (score=None) are left out of the
        calculation entirely rather than counted as zero — a skipped
        check is missing information, not a failing grade. If every
        check was skipped, there's nothing to average, so this
        returns None rather than a misleading number.
        """
        scored = {
            name: result
            for name, result in self.checks.items()
            if result.score is not None
        }

        if not scored:
            return None

        total_weight = sum(self._WEIGHTS.get(name, 1.0) for name in scored)
        weighted_sum = sum(
            result.score * self._WEIGHTS.get(name, 1.0)
            for name, result in scored.items()
        )

        return round(weighted_sum / total_weight)

    def to_json(self, path: str) -> None:
        """
        Writes the report as a JSON file — the format meant for
        pipelines and scripts to read, not for a person to open.

        ensure_ascii=False is deliberate, not an oversight: without it,
        every non-English character (e.g. Urdu script) gets rewritten
        as an escape code like \\u0645 instead of the actual letter,
        which makes the report unreadable for the exact datasets this
        tool is built to audit.
        """
        data = {
            "inference_audit_version": self.audit_version,
            "dataset": self.dataset_path,
            "rows": self.rows,
            "label_col": self.label_col,
            "text_col": self.text_col,
            "audit_timestamp": self.timestamp,
            "overall_score": self.overall_score,
            "checks": {
                name: {
                    "score": result.score,
                    "warning": result.warning,
                    **result.details,
                }
                for name, result in self.checks.items()
            },
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError as e:
            raise OSError(
                f"Could not write JSON report to '{path}'. "
                f"Check that the folder exists and is writable."
            ) from e

    def save(self, path: str) -> None:
        """
        Writes the report as a self-contained HTML file — the format
        meant for a person to open in a browser and read.

        The actual rendering lives in report_renderer.py. It's
        imported here, inside the method, rather than at the top of
        this file, because report_renderer.py imports AuditReport from
        this file importing it at the top would create a circular
        import (this file waiting on a file that's waiting on this
        file).
        """
        from inference_audit.report_renderer import render_html

        try:
            render_html(self, path)
        except OSError as e:
            raise OSError(
                f"Could not write HTML report to '{path}'. "
                f"Check that the folder exists and is writable."
            ) from e

from importlib.metadata import packages_distributions, version, PackageNotFoundError

def _get_audit_version() -> str:
    """
    Reads the installed package version by looking up whatever
    distribution name currently provides the `inference_audit` import
    package -- rather than hardcoding a specific PyPI name. This means
    the code doesn't need to change if the PyPI distribution name is
    ever renamed (e.g. once the naming conflict is resolved).
    """
    try:
        distributions = packages_distributions().get("inference_audit", [])
        if distributions:
            return version(distributions[0])
    except AttributeError:
        # packages_distributions() was added in Python 3.10 -- on 3.9,
        # this attribute won't exist. Fall through to "dev".
        pass
    except PackageNotFoundError:
        pass

    return "dev"