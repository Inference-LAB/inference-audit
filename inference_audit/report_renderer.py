"""Renders an AuditReport as a self-contained HTML file.

Uses Jinja2 for the page structure and matplotlib for the score chart,
embedded directly as a base64 PNG -- no external files, so the output
stays a single, self-contained HTML document per the design doc's
output requirement.
"""

import base64
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = Path(__file__).parent / "templates"


def render_html(report, path):
    """Renders an AuditReport to a self-contained HTML file.

    Args:
        report: An AuditReport instance.
        path: Output file path for the HTML report.

    Raises:
        OSError: if the file can't be written (bad path, no permission).
    """
    chart_base64 = _build_score_chart(report.checks)

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("report.html.j2")

    html_output = template.render(
        dataset_path=report.dataset_path,
        rows=report.rows,
        label_col=report.label_col,
        text_col=report.text_col,
        audit_version=report.audit_version,
        timestamp=report.timestamp,
        overall_score=report.overall_score,
        checks=report.checks,
        chart_image_base64=chart_base64,
    )

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_output)
    except OSError as e:
        raise OSError(
            "Could not write HTML report to '" + str(path) + "'. "
            "Check that the folder exists and is writable."
        ) from e


def _build_score_chart(checks):
    """Builds a horizontal bar chart of per-check scores as a base64 PNG.

    Skipped checks (score=None) are shown as empty gray bars labeled
    "skipped" rather than omitted, so the chart always covers all five
    checks regardless of what ran.

    Returns:
        A base64-encoded PNG string, ready to drop into an
        <img src="data:image/png;base64,..."> tag.
    """
    names = [name.replace("_", " ").title() for name in checks.keys()]
    scores = [r.score if r.score is not None else 0 for r in checks.values()]
    colors = []
    for r in checks.values():
        if r.score is None:
            colors.append("#cccccc")
        elif r.score >= 80:
            colors.append("#1e7e34")
        elif r.score >= 50:
            colors.append("#b8860b")
        else:
            colors.append("#c0392b")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.barh(names, scores, color=colors)

    for bar, r in zip(bars, checks.values()):
        label = "skipped" if r.score is None else str(r.score)
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=9,
            color="#333333",
        )

    ax.set_xlim(0, 105)
    ax.set_xlabel("Score")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")