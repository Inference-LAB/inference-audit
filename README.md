# inference-audit

An NLP dataset quality auditor. Run one command, get a reproducible report covering five quality checks and an overall score.This pip installable python library is built so a dataset that scores well is something you can actually cite as evidence of quality, not just a number.

```
pip install inference-audit-pk
```

## Why this exists

Every stage of an NLP pipeline from publishing a dataset, choosing one for training , depends on quality that's usually checked with a one-off script written from scratch each time, producing results that aren't comparable across projects or papers. `inference-audit` replaces that with a standard, versioned, reproducible check anyone can run and cite.

## Quickstart

### Default usage

The simplest case — audits a dataset using the five checks with default settings. `annotation_consistency` will be skipped (not scored zero) since no confidence column is provided.

```python
from inference_audit import Auditor

auditor = Auditor()
report = auditor.audit("dataset.csv", label_col="emotion", text_col="text")

report.save("audit_report.html")     # human-readable, self-contained HTML
report.to_json("audit_report.json")  # machine-readable, for CI pipelines
```

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

### With a confidence column

If your dataset includes an annotation confidence or agreement score per row, pass it in to enable the `annotation_consistency` check — otherwise this check is silently skipped rather than scored.

```python
report = auditor.audit(
    "dataset.csv", label_col="emotion", text_col="text",
    conf_col="confidence",
)
```

```bash
inference-audit run dataset.csv \
  --label-col emotion \
  --text-col  text \
  --conf-col  confidence \
  --output    audit_report.html
```

PowerShell:
```powershell
inference-audit run dataset.csv `
  --label-col emotion `
  --text-col  text `
  --conf-col  confidence `
  --output    audit_report.html
```

### With a custom concern-languages list

`language_contamination` defaults to flagging English/Hindi as contamination risks. If your dataset has a different realistic contamination risk (e.g. an English dataset where French or Spanish leakage is the actual concern), override the default rather than relying on it.

```python
report = auditor.audit(
    "dataset.csv", label_col="label", text_col="text",
    concern_languages=("fr", "es"),
)
```

```bash
inference-audit run dataset.csv \
  --label-col label \
  --text-col  text \
  --concern-languages fr,es \
  --output    audit_report.html
```

PowerShell:
```powershell
inference-audit run dataset.csv `
  --label-col label `
  --text-col  text `
  --concern-languages fr,es `
  --output    audit_report.html
```

### CI quality gate

Use `--fail-below <score>` to make the command exit with a non-zero status if the dataset doesn't meet a quality bar — this is what makes it usable as a CI gate on a pull request.

```bash
inference-audit run dataset.csv --label-col emotion --text-col text --fail-below 70
```

## What gets checked

**Label distribution** — flags severe class imbalance. A dataset where one label dominates and another barely appears is harder to train on fairly and easier to get misleading results from.

**Near duplicates** — flags verbatim and near-verbatim repeated samples using MinHash/LSH similarity, including cases where the same text appears under two different labels (a common annotation error).

**Language contamination** — flags text confidently detected in a language your dataset isn't supposed to contain. Checks against a configurable list of "concern languages" (default: English and Hindi) rather than trying to guess the dataset's "true" language, since off-the-shelf language detection has no reliable profile for every language — Roman Urdu being one notable example. Override `concern_languages`/`--concern-languages` for datasets with a different realistic contamination risk.

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

- **Language detection has no real signal for some languages.** The underlying detection library has no language profile for some languages (Roman Urdu being a notable example), so `language_contamination` doesn't attempt to identify "the dataset's language" — it only flags confident detections of specific languages on a configurable concern list. Text in an undetectable language can still occasionally be misdetected as a concern-list language (measured false-positive rate: roughly 10–12% on real Roman Urdu test data) — this is a bounded, documented limitation, not a bug.
- **Performance on very large datasets.** Recent optimizations (deduplication before comparison, parallelized language detection) significantly improved runtime on large datasets, but `near_duplicates` and `language_contamination` remain the two most compute-intensive checks. On datasets over 100K rows, a full audit can still take up to several minutes (up to ~5 minutes observed on a 100K+ row real-world dataset) depending on text diversity and duplication rate. If you're auditing a very large corpus, expect these two checks to dominate total runtime.
- **`--concern-languages` defaults to English/Hindi.** This is a reasonable general-purpose default, particularly for code-switched or Roman-script corpora where English/Hindi leakage is a common risk — but it is not universally correct. If you're auditing a dataset with a different realistic contamination risk, override this flag rather than relying on the default.

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

See `sample_report.html` and `sample_report.json` in this repository for full rendered examples, including the score chart.

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

- **Khadija Faisal** (Lead Engineer) — [GitHub](https://github.com/khadijja1) · [LinkedIn](https://www.linkedin.com/in/khadijjafaisal)
- **Muhammad Shoaib Altaf** (Research & Implementation Engineer) — [GitHub](https://github.com/Shoaib-Altaf) · [LinkedIn](https://www.linkedin.com/in/muhammad-shoaib-altaf-6ab3a8326)