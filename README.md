# inference-audit

An NLP dataset quality auditor. Run one command, get a reproducible report covering five quality checks and an overall score — built so a dataset that scores well is something you can actually cite as evidence of quality, not just a number.

```
pip install inference-audit-pk
```

## Why this exists

Every stage of an NLP pipeline — publishing a dataset, choosing one for training — depends on quality that's usually checked with a one-off script written from scratch each time, producing results that aren't comparable across projects or papers. `inference-audit` replaces that with a standard, versioned, reproducible check anyone can run and cite.

## Quickstart

### Python

```python
from inference_audit import Auditor

auditor = Auditor()
report = auditor.audit("dataset.csv", label_col="emotion", text_col="text")

report.save("audit_report.html")     # human-readable, self-contained HTML
report.to_json("audit_report.json")  # machine-readable, for CI pipelines
```

### CLI

```bash
inference-audit run dataset.csv \
  --label-col emotion \
  --text-col  text \
  --output    audit_report.html
```

PowerShell:
```powershell
inference-audit run dataset.csv `
  --label-col emotion `
  --text-col  text `
  --output    audit_report.html
```

Use `--fail-below <score>` to make the command exit with a non-zero status if the dataset doesn't meet a quality bar — this is what makes it usable as a CI gate on a pull request:

```bash
inference-audit run dataset.csv --label-col emotion --text-col text --fail-below 70
```

## What gets checked

**Label distribution** — flags severe class imbalance. A dataset where one label dominates and another barely appears is harder to train on fairly and easier to get misleading results from.

**Near duplicates** — flags verbatim and near-verbatim repeated samples using MinHash/LSH similarity, including cases where the same text appears under two different labels (a common annotation error).

**Language contamination** — flags text confidently detected in a language your dataset isn't supposed to contain. Checks against a configurable list of "concern languages" (default: English and Hindi, reflecting realistic contamination risk for Roman Urdu corpora) rather than trying to guess the dataset's "true" language, since off-the-shelf language detection has no reliable profile for Roman Urdu.

**Missing values** — flags null, whitespace-only, and suspiciously short text samples that likely don't carry real content.

**Annotation consistency** — flags low-confidence or inconsistent annotations, when a confidence/agreement column is available. Skipped (not scored zero) if no such column is provided.

Each check returns a score (0–100), an optional warning, and detailed metrics. An overall score is computed as a weighted average — label distribution and near-duplicates count for more, reflecting which issues most often affect real published datasets.

## CLI Reference

| Flag | Required | Description |
|---|---|---|
| `path` | Yes | Dataset file (`.csv`, `.json`, or `.parquet`) |
| `--label-col` | Yes | Name of the label column |
| `--text-col` | Yes | Name of the text column |
| `--output` | No | Output path, `.html` or `.json` (default: `audit_report.html`) |
| `--conf-col` | No | Confidence/agreement column, enables the annotation consistency check |
| `--concern-languages` | No | Comma-separated ISO 639-1 codes to flag if detected (default: `en,hi`) |
| `--fail-below` | No | Exit with code 1 if the overall score is below this value (0–100) |

## Known Limitations

- **Language detection has no real signal for Roman Urdu.** The underlying detection library has no language profile for it, so `language_contamination` doesn't attempt to identify "the dataset's language" — it only flags confident detections of specific languages on a configurable concern list. Genuine Roman Urdu text can still occasionally be misdetected as a concern-list language (measured false-positive rate: roughly 10–12% on real test data) — this is a bounded, documented limitation, not a bug.
- **Near-duplicate detection performance** on very large datasets (100K+ rows) can approach the tool's own time budget; if you're auditing a very large corpus, expect this to be the slowest of the five checks.
- **`--concern-languages` defaults to English/Hindi**, chosen for the tool's primary use case (Roman Urdu corpora). If you're auditing a dataset with different realistic contamination risks, override this flag rather than relying on the default.

## Example Output

Running against a real 134K-row Roman Urdu emotion corpus:

```
Overall score: 74

label_distribution:       61  (severe class imbalance detected)
near_duplicates:          78.6
language_contamination:   71.1
missing_values:           88
annotation_consistency:   skipped (no confidence column provided)
```

See `report.html` in this repository for a full rendered example, including the score chart.

## Requirements

- Python 3.9+
- pandas, jinja2, matplotlib, datasketch, langdetect, typer (installed automatically)

## Contributing / Development Setup

```bash
git clone https://github.com/Inference-LAB/inference-audit.git
cd inference-audit
python -m venv venv
source venv/bin/activate   # or venv\Scripts\Activate.ps1 on Windows
pip install -e .
pytest --cov=inference_audit
```

## License

MIT — see [LICENSE](LICENSE).

## Authors

Built as part of the Inference Lab Engineering Fellowship, Cohort 01 — Project C.

- Khadija Faisal (Lead Engineer)
- Muhammad Shoaib Altaf (Research & Implementation Engineer)