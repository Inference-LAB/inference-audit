"""Command-line interface for inference-audit."""

from pathlib import Path

import typer

from inference_audit.auditor import Auditor

app = typer.Typer(help="inference-audit: NLP dataset quality auditor.")


@app.command()
def run(
    path: str = typer.Argument(..., help="Path to the dataset file (.csv, .json, .parquet)."),
    label_col: str = typer.Option(..., "--label-col", help="Name of the label column."),
    text_col: str = typer.Option(..., "--text-col", help="Name of the text column."),
    output: str = typer.Option("audit_report.html", "--output", help="Output file path. Format inferred from extension (.html or .json)."),
    conf_col: str = typer.Option(None, "--conf-col", help="Optional confidence/agreement column."),
    language: str = typer.Option("auto", "--language", help="Expected language code, or 'auto' to detect."),
    fail_below: int = typer.Option(None, "--fail-below", help="Exit with code 1 if overall_score falls below this value."),
):
    """Run a full quality audit on a dataset and write a report."""
    auditor = Auditor()

    try:
        report = auditor.audit(
            path,
            label_col=label_col,
            text_col=text_col,
            conf_col=conf_col,
            language=language,
        )
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    output_path = Path(output)
    if output_path.suffix.lower() == ".json":
        report.to_json(output)
    else:
        report.save(output)

    typer.echo(f"Report written to {output}")
    typer.echo(f"Overall score: {report.overall_score}")

    if fail_below is not None:
        if report.overall_score is None:
            typer.echo("Warning: overall_score is None (all checks skipped) - cannot compare against --fail-below.")
        elif report.overall_score < fail_below:
            typer.echo(f"Score {report.overall_score} is below threshold {fail_below}.")
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()