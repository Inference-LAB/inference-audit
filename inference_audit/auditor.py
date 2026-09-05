"""Auditor: orchestrates dataset loading and runs all quality checks."""

import datetime

from inference_audit.loader import load_dataset
from inference_audit.report import AuditReport, _get_audit_version
from inference_audit.checks.label_distribution import check_label_distribution
from inference_audit.checks.near_duplicates import check_near_duplicates
from inference_audit.checks.language_contamination import (
    check_language_contamination,
    DEFAULT_CONCERN_LANGUAGES,
)
from inference_audit.checks.missing_values import check_missing_values
from inference_audit.checks.annotation_consistency import check_annotation_consistency


class Auditor:
    """Runs a full quality audit on a dataset and returns an AuditReport."""

    def audit(
        self,
        path: str,
        label_col: str,
        text_col: str,
        conf_col: str = None,
        concern_languages=None,
    ) -> AuditReport:
        """Loads the dataset, runs all five checks, and returns an AuditReport.

        Args:
            path: Dataset file path (.csv, .json, or .parquet).
            label_col: Name of the label column.
            text_col: Name of the text column.
            conf_col: Optional confidence column for annotation_consistency.
            concern_languages: Optional iterable of ISO 639-1 language codes
                to flag if confidently detected in language_contamination.
                Defaults to ("en", "hi") if not provided -- see
                check_language_contamination's docstring. Override this for
                datasets where the realistic contamination risk is a
                different language pair.

        Returns:
            A fully populated AuditReport.

        Raises:
            FileNotFoundError, ValueError: propagated from load_dataset()
            if the dataset can't be loaded or validated.
        """
        df = load_dataset(path, label_col, text_col)

        checks = {
            "label_distribution": check_label_distribution(df, label_col),
            "near_duplicates": check_near_duplicates(df, text_col),
            "language_contamination": check_language_contamination(
                df,
                text_col,
                concern_languages=concern_languages or DEFAULT_CONCERN_LANGUAGES,
            ),
            "missing_values": check_missing_values(df, text_col),
            "annotation_consistency": check_annotation_consistency(
                df, conf_col=conf_col
            ),
        }

        return AuditReport(
            dataset_path=path,
            rows=len(df),
            label_col=label_col,
            text_col=text_col,
            audit_version=_get_audit_version(),
            timestamp=datetime.datetime.utcnow().isoformat(),
            checks=checks,
        )