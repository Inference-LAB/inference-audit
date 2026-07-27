# inference-audit — Design Document

**INFERENCE Lab Engineering Fellowship · Cohort 01 · Project C**

**Team:**
- Lead Engineer: Khadija Faisal
- Research and Implementation Engineer: Muhammad Shoaib Altaf


## 1. Project Summary
*(Owner: Lead Engineer)*

inference-audit is a Python library and command-line tool that audits the quality of an NLP dataset before it is used for research or model training. It runs five structured checks: label distribution, near-duplicate detection, language contamination, missing value analysis, and annotation consistency, and produces a single, reproducible report with a per-check score and an overall weighted quality score. The goal is to replace one-off, hand-written quality scripts with a standard, citable tool that any researcher can run with one command.

---

## 2. Problem Statement
Every stage of the NLP pipeline,from a researcher publishing a dataset to a practitioner selecting one for model training ,depends on data quality, yet there is no standard way to answer a simple question: *how do I know my dataset is actually good?* At present, labs and researchers rewrite quality-checking scripts from scratch for every corpus release, producing results that are inconsistent and not comparable across papers, even within the same lab. Reviewers have no common standard to point to when requesting quality evidence. inference-audit addresses this gap directly by becoming a standardised, pip-installable, reproducible tool that any NLP researcher can cite in a methods section — turning dataset quality from an ad hoc claim into a falsifiable, versioned report.

---
## 3. Technical Approach
*(Owner: Research/Implementation Engineer)*

### Near-Duplicate Detection Threshold (MinHash/LSH)

**Method:** Tested character-3-gram Jaccard similarity on 20 real pairs from RUEmoCorp (`Khubaib01/RUEmoCorp`, `ruemocorp-annotated` split) — 15 true near-duplicates spanning punctuation-only differences, word reordering, single- and multi-word spelling variants, minor additions, and a case-scrambled repost pattern found naturally in the corpus; 5 distinct pairs sharing surface vocabulary/tone but differing in actual meaning.

**Result:** With text lowercased before shingling, true duplicates scored 0.851–1.000 (avg 0.943); distinct pairs scored 0.053–0.164 (avg 0.107) — a clean separation with no overlap.

**Finding:** `threshold=0.8` (the brief's suggested default) correctly classifies all 20 pairs — but only when lowercasing is applied before fingerprinting. Without it, the case-scrambled duplicate pattern found in this corpus would score significantly lower and risk falling below threshold, producing false negatives.

**Recommendation:** `check_near_duplicates` must lowercase text as a preprocessing step before MinHash fingerprinting.

**Known limitation:** Sample size (20 pairs, ~5 pattern types) is a first-pass feasibility check. Recommend expanding during Week 3 implementation, and stress-testing with synthetically injected duplicates since natural near-duplicates are relatively rare in this dataset.

### Language Contamination Detection Reliability (langdetect)

**Method:** Tested `langdetect` on 12 real Roman Urdu messages at 3/5/10/20-token lengths (44 total detections). Ran a seed-determinism test and a sanity check against real English/French/Spanish text to rule out tool malfunction.

**Result:** `langdetect` never correctly identified Roman Urdu at any token length tested. Guesses scattered across 10+ unrelated languages with no improvement at longer lengths. Sanity check confirmed the tool works correctly on languages it supports (100% accuracy on English/French/Spanish). Seed-determinism test confirmed `DetectorFactory.seed = 0` is required: 2/5 texts returned different language labels across repeated runs without it; all stable once set.

**Finding:** There is no "minimum reliable length" for Roman Urdu with `langdetect` — it has no language profile for Roman Urdu at all, a structural limitation, not a short-text weakness.

**Recommendation:** `check_language_contamination` cannot use "not detected as Urdu" as its contamination signal, since that would flag the entire dataset as 100% contaminated. Instead, only flag samples confidently (>0.9 probability) identified as a real, unrelated language. `DetectorFactory.seed = 0` must be set at module init.

### Scoring Formulas (All 5 Checks)

| Check | Formula | Justification |
|---|---|---|
| `check_near_duplicates` | `100 × (1 - duplicate_rate)` | duplicate_rate = flagged pairs / total samples. Grounded in threshold finding above. |
| `check_label_distribution` | `100 × (1 - min(imbalance_ratio / max_acceptable_ratio, 1))` | max_acceptable_ratio configurable (proposed default: 20). 1:1 balance → 100; ratio ≥ 20:1 → 0, linear between. |
| `check_missing_values` | `100 × (1 - missing_rate)` | missing_rate = (null + whitespace-only + below-length-threshold) / total. |
| `check_language_contamination` | `100 × (1 - confident_contamination_rate)` | Only counts confident (>0.9 prob) detections of a real, unrelated language — per finding above. |
| `check_annotation_consistency` | `100 × (1 - low_confidence_rate)` | low_confidence_rate = samples below threshold (proposed default: 0.6) / total with a confidence column. Returns `None` if no confidence column exists. |

**Note:** Per the brief, `label_distribution` and `near_duplicates` are weighted 1.5× in the overall `AuditReport` score — flagging for team confirmation this still applies.

---



---
## 4. Evaluation Plan
*(Owner: Integration/Evaluation Engineer — role shared between Lead and Research/Implementation Engineer)*

**1. Synthetic fixture validation (primary pass/fail gate)**
For each of the five checks, the package will be run against paired synthetic fixtures one deliberately clean, one deliberately flawed ,and the resulting score must fall inside a pre-defined expected range, calculated independently of the code before the test is run. Example thresholds:
- Balanced, clean dataset → check score expected in the 90–100 range
- Severely imbalanced dataset (e.g. 950/50 class split) → check score expected below 30
- Dataset with seeded near-duplicates and cross-label duplicates → check score expected in the 40–65 range

A check only passes evaluation if its output falls inside the pre-calculated expected range on both the clean and the flawed fixture. A result outside this range fails evaluation and is treated as a defect in either the check or the fixture, to be investigated before merge — not silently accepted.

**2. Real-world dataset validation**
Once all five checks pass fixture validation, the package will be run end-to-end on 2–3 real public NLP datasets, including a subset of `rotten_tomatoes` (HuggingFace) and samples in multiple languages (English and Roman Urdu) to confirm language detection behaves correctly on real, non-synthetic text. This step validates that the checks generalise beyond controlled fixtures and surface genuine, interpretable issues on real data.

**3. Non-functional acceptance criteria**
- CLI `--fail-below` flag correctly exits with code 1 when `overall_score` falls below the given threshold, and code 0 otherwise.
- Full audit on a 100K-row dataset completes in under 60 seconds on CPU.
- `pytest` suite achieves 90%+ coverage on all check modules.

Only once all three layers above pass will the package be considered ready for PyPI publication.

---

## 5. Module Ownership Table
| File / Module | Owner | Depends On | Target Week |
|---|---|---|---|
| `inference_audit/report.py` (`AuditReport`, `CheckResult`) | Khadija (Lead) | — | Week 1 |
| `inference_audit/loader.py` | Khadija (Lead) | — | Week 2–3 |
| `inference_audit/auditor.py` | Khadija (Lead) | `loader.py`, `report.py`, all `checks/*.py` | Week 2–3 |
| `inference_audit/templates/report.html.j2` (Jinja2) | Khadija (Lead) | `report.py` | Week 2–3 (structure), Week 5 (charts) |
| `inference_audit/report_renderer.py` | Khadija (Lead) | `report.py`, Jinja2 template | Week 4–5 |
| `inference_audit/cli.py` (Typer) | Khadija (Lead) | `auditor.py` | Week 4–5 |
| `pyproject.toml` / PyPI packaging | Khadija (Lead) | all modules | Week 6 |
| `README.md` | Khadija (Lead) | all modules | Week 4–5 (draft), Week 6 (final) |
| `inference_audit/checks/missing_values.py` | Shoaib (Research/Implementation) | `report.py` (`CheckResult`) | Week 2 |
| `inference_audit/checks/label_distribution.py` | Shoaib (Research/Implementation) | `report.py` | Week 2 |
| `inference_audit/checks/near_duplicates.py` (MinHash/LSH via datasketch) | Shoaib (Research/Implementation) | `report.py`, `datasketch` | Week 3 |
| `inference_audit/checks/language_contamination.py` (langdetect) | Shoaib (Research/Implementation) | `report.py`, `langdetect` | Week 4 |
| `inference_audit/checks/annotation_consistency.py` | Shoaib (Research/Implementation) | `report.py` | Week 4 |
| Scoring formulas (all five checks) | Shoaib (Research/Implementation) | — | Week 1 (design), Week 2–4 (implementation) |
| `tests/generate_fixtures.py` (synthetic fixture generator) | Shoaib (build) + Khadija (final review) | — | Week 2 |
| `tests/test_<check>.py` (per-check unit tests, 90%+ coverage target) | Shoaib (writes, runs locally before PR) + Khadija (final review before merge) | fixtures, each check module | Week 3–4 |
| `tests/test_performance.py` (100K rows < 60s benchmark) | Shoaib (writes) + Khadija (final verification) | `auditor.py`, all checks | Week 4 |
| Coverage report (`pytest --cov`, 90%+ target) | Khadija (final run + verification) | full test suite | Week 5–6 |
| `pip install` clean-virtualenv verification | Khadija (Lead) | `pyproject.toml` | Week 6 |
| `sample_report.html` / `sample_report.json` (real public dataset run) | Shoaib (generates) + Khadija (verifies output against schema) | full pipeline | Week 6 |
| 500-word lab notes (individual) | Khadija + Shoaib (separately) | — | Week 6 |

---

## 6. Known Risks

1. **Package name conflict.** `inference-audit` is already registered on PyPI (an inactive placeholder redirecting to `aiir`). Verified alternatives available at time of writing: `inference-auditor`, `inference-check`, `inference-analysis`. The Python import name (`from inference_audit import Auditor`) is unaffected — only the PyPI distribution name needs to change. 

2. **Reduced team size.** Unlike Projects A and B, this project has two members rather than three ,there is no dedicated Integration/Evaluation Engineer. Fixture generation, test-suite ownership, and performance/coverage verification are split between the Lead and Research/Implementation Engineer. **Mitigation:** evaluation responsibilities are explicitly assigned per-file in the Module Ownership Table above, with the Lead performing an independent final verification pass rather than relying solely on self-review.

3. **langdetect cannot reliably detect Roman Urdu.** Testing on 12 real RUEmoCorp messages at 3/5/10/20-token lengths (44 total detections) found `langdetect` never correctly identified Roman Urdu at any length tested — guesses scattered across 10+ unrelated languages (Somali, Indonesian, Estonian, Tagalog, etc.) with no improvement at longer lengths. A sanity check confirmed the tool works correctly on languages it does support (100% accuracy on real English/French/Spanish text), ruling out a tool malfunction — the issue is that Roman Urdu simply has no language profile in `langdetect`, a structural limitation rather than a short-text weakness. **Mitigation:** `check_language_contamination` will not use "not detected as Urdu" as its contamination signal, since that would flag the entire dataset as 100% contaminated. Instead, the check will only flag samples confidently (>0.9 probability) identified as a real, unrelated language (e.g. genuine English text mixed into the corpus). `DetectorFactory.seed = 0` will be set at module init to ensure reproducible results, since testing also confirmed detection is non-deterministic without it (2/5 test texts returned different results across repeated runs on identical input before the seed was set).

---

## 7. Definition of Done
*(Owner: Lead Engineer)*

### Core Functionality
- [ ] `Auditor().audit(path, label_col, text_col)` returns a valid `AuditReport`
- [ ] All five checks implemented and wired into `Auditor.audit()`: `label_distribution`, `near_duplicates`, `language_contamination`, `missing_values`, `annotation_consistency`
- [ ] Every check returns a `CheckResult` (never a raw dict, never `None`, never raises unexpectedly)
- [ ] Per-check score (0–100) computed correctly; `overall_score` computed as documented weighted average (label_distribution & near_duplicates weighted 1.5x)
- [ ] Per-check `warning` string populated when issues are found; `None` when clean
- [ ] Graceful degradation confirmed: a check with missing required input (e.g. no `conf_col`) returns `score=None` with explanation — does not crash the whole audit

### Output Formats
- [ ] `report.save()` produces a self-contained single HTML file — no external CSS/JS dependencies, opens correctly offline
- [ ] `report.to_json()` produces valid, versioned JSON matching the schema in the Brief
- [ ] `ensure_ascii=False` confirmed in `to_json()` — verified with a real Roman Urdu / Urdu sample, not just tested in English
- [ ] JSON output round-trip tested: `json.load()` on the output file parses without error

### CLI
- [ ] `inference-audit run <path> --label-col <col> --text-col <col> --output <file>` works end-to-end
- [ ] `--fail-below <int>` flag implemented — exits with code 1 when `overall_score` is below threshold, exit code 0 otherwise
- [ ] CLI tested against a real GitHub Actions-style invocation (or simulated locally)

### Data Loading
- [ ] All three formats load correctly: `.csv`, `.json`, `.parquet`
- [ ] Correct exceptions raised: `FileNotFoundError` (missing file), `ValueError` (unsupported format, empty dataset, missing required columns)
- [ ] JSON parse failure produces a clear, custom error message — not a raw pandas traceback
- [ ] Column validation happens once, centrally in `loader.py` — not duplicated inside each check

### Testing & Quality
- [ ] pytest suite passes, 90%+ coverage on all check modules (`pytest --cov=inference_audit --cov-report=html`)
- [ ] Fixture datasets exist for all five checks with documented expected score ranges, and each check produces a score inside its fixture's expected range
- [ ] Performance test passes: full audit on a 100K-row synthetic dataset completes in under 60 seconds on CPU
- [ ] Edge cases verified: 1-row dataset, all-same-label dataset, numeric-only text column, trailing whitespace in column names — none crash

### Packaging & Docs
- [ ] Package name confirmed available (or reserved) on PyPI before publishing
- [ ] `pip install inference-audit` (or chosen name) installs cleanly in a fresh Python 3.9+ virtualenv
- [ ] All public functions/classes have complete docstrings (params, return type, known limitations)
- [ ] README includes: installation, quickstart, explanation of each check, known limitations, one worked example on a real public dataset

### Process / Submission
- [ ] `sample_report.html` and `sample_report.json` generated from a real public dataset run and committed to the repo
- [ ] 500-word lab note written and submitted per Handbook format
- [ ] All PRs for this project reviewed and merged to `main` (no outstanding unmerged work)

---

## Appendix: Public API Design

### Design Principle
The public surface must be small enough to use correctly without reading documentation. A developer should be able to run a full audit in three lines of Python, or one line of shell.

### 1. Python API — Primary Interface

```python
from inference_audit import Auditor

auditor = Auditor()
report  = auditor.audit("ruemocorp_v1.csv", label_col="emotion", text_col="text")

report.save("audit_report.html")     # human-readable, self-contained HTML
report.to_json("audit_report.json")  # machine-readable, for CI pipelines
```

**`Auditor` class**
- `Auditor()` — no required constructor arguments for v1.
- `Auditor.audit(path, label_col, text_col, conf_col=None, language="auto") -> AuditReport`
  - `path` (str) — dataset file path. Format detected by extension: `.csv`, `.json`, `.parquet`.
  - `label_col` (str) — required. Name of the label column.
  - `text_col` (str) — required. Name of the text column.
  - `conf_col` (str, optional) — annotation confidence column, if available. If omitted, `annotation_consistency` check returns `score=None` with an explanatory warning instead of failing.
  - `language` (str, default `"auto"`) — expected language for contamination check. Accepts `"auto"` or an ISO 639-1 code (e.g. `"ur"`).
  - Returns a fully populated `AuditReport`. Never returns a partially-run report — either the whole audit completes (with individual checks possibly skipped/scored `None`) or an exception is raised before any `AuditReport` is constructed.

**`AuditReport` — returned object**
- `report.overall_score` — property, weighted average across all non-skipped checks (0–100, or `None` if every check was skipped).
- `report.checks` — dict of five `CheckResult` objects, keyed by check name.
- `report.save(path: str) -> None` — writes self-contained HTML report.
- `report.to_json(path: str) -> None` — writes versioned JSON report (`ensure_ascii=False`).

**`CheckResult` — one per check, nested inside `AuditReport.checks`**
- `.score` — `int` 0–100, or `None` if the check was skipped.
- `.warning` — `str` describing an issue, or `None` if clean.
- `.details` — `dict` of check-specific metrics (e.g. `imbalance_ratio`, `duplicate_pct`).

### 2. CLI — Secondary Interface (wraps the same Python API)

```bash
inference-audit run <path> \
  --label-col <col> \
  --text-col  <col> \
  --output    <file> \
  [--conf-col <col>] \
  [--language <code>] \
  [--fail-below <int>]
```

PowerShell (Windows):
```powershell
inference-audit run <path> `
  --label-col <col> `
  --text-col  <col> `
  --output    <file> `
  --fail-below <int>
```

- `run` — the only subcommand for v1.
- `--fail-below <int>` — if `overall_score` is below this threshold, the CLI exits with code `1` (non-zero); otherwise exits `0`. This is what makes the tool usable as a CI quality gate.
- Output format (HTML vs JSON) is inferred from the `--output` file extension.
