"""
inference_audit/checks/near_duplicates.py

Detects verbatim and near-verbatim duplicate samples using MinHash + LSH
over character 3-grams. Threshold and preprocessing choices are backed
by the Week 1 design doc findings (20 real RUEmoCorp pairs tested).
"""

import pandas as pd
from datasketch import MinHash, MinHashLSH
from inference_audit.report import CheckResult


def _to_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    text = text.lower().strip()
    for i in range(len(text) - 2):
        m.update(text[i:i + 3].encode("utf-8"))
    return m


def check_near_duplicates(
    df: pd.DataFrame,
    text_col: str,
    threshold: float = 0.8,
    num_perm: int = 128,
) -> CheckResult:
    """
    Flags near-duplicate rows using MinHash + LSH over character 3-grams.

    Never raises. Returns CheckResult(score=None, ...) for edge cases:
    missing column, empty dataframe, fewer than 2 rows, or -- per Lead
    Engineer review -- when EVERY row is too short to compare (zero
    comparable rows after filtering is the same "nothing to measure"
    situation as the single-row case, just reached a different way, and
    must not silently report a perfect score).
    """
    if text_col not in df.columns:
        return CheckResult(
            score=None,
            warning=f"Column '{text_col}' not found in dataset.",
            details={"error": "missing_column"},
        )

    total_rows = len(df)
    if total_rows == 0:
        return CheckResult(
            score=None,
            warning="Dataset is empty (0 rows) — nothing to check.",
            details={"error": "empty_dataframe"},
        )

    if total_rows < 2:
        return CheckResult(
            score=None,
            warning="Fewer than 2 rows — no possible duplicate pairs to check.",
            details={"error": "insufficient_rows"},
        )

    MIN_SHINGLE_LENGTH = 3
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    texts = df[text_col].fillna("").astype(str)
    skipped_short_count = 0

    for idx, text in texts.items():
        if len(text.strip()) < MIN_SHINGLE_LENGTH:
            skipped_short_count += 1
            continue
        mh = _to_minhash(text, num_perm=num_perm)
        minhashes[idx] = mh
        lsh.insert(str(idx), mh)

    # Lead Engineer review (Week 4): if EVERY row was too short to build
    # a meaningful signature, there are zero comparable rows -- this is
    # the same "nothing to compare" situation as total_rows < 2, just
    # reached via filtering instead of raw row count. Before this fix,
    # falling through with an empty minhashes dict produced
    # duplicate_rate=0/total_rows=0 -> score=100.0, silently reporting
    # "perfectly clean" when nothing was actually measured. Caught via
    # Khadija's test case: ["hi","ok","no","hi","ok"] (5 rows, all
    # too short) incorrectly returned 100.0 instead of None.
    if len(minhashes) == 0:
        return CheckResult(
            score=None,
            warning=(
                f"All {total_rows} rows were too short to compare "
                f"(< {MIN_SHINGLE_LENGTH} characters) — nothing could be measured."
            ),
            details={"error": "all_rows_too_short", "total_rows": total_rows, "skipped_short_count": skipped_short_count},
        )

    candidate_pairs = set()
    for idx, mh in minhashes.items():
        matches = lsh.query(mh)
        for match_str in matches:
            pair = tuple(sorted((str(idx), str(match_str))))
            if pair[0] != pair[1]:
                candidate_pairs.add(pair)

    pair_count = len(candidate_pairs)

    flagged_row_ids = set()
    for a, b in candidate_pairs:
        flagged_row_ids.add(a)
        flagged_row_ids.add(b)

    duplicate_rate = len(flagged_row_ids) / total_rows
    score = round(max(0.0, 100 * (1 - duplicate_rate)), 2)

    warning_parts = []
    if pair_count > 0:
        warning_parts.append(
            f"{pair_count} candidate duplicate pair(s) found among {total_rows} rows "
            f"({len(flagged_row_ids)} rows involved)."
        )
    if skipped_short_count > 0:
        warning_parts.append(
            f"{skipped_short_count} row(s) skipped (text shorter than "
            f"{MIN_SHINGLE_LENGTH} characters -- too short to compare reliably)."
        )
    warning = " ".join(warning_parts) if warning_parts else None

    example_pairs = []
    for a, b in list(candidate_pairs)[:5]:
        example_pairs.append({
            "row_a": texts.loc[int(a) if a.isdigit() else a][:80],
            "row_b": texts.loc[int(b) if b.isdigit() else b][:80],
        })

    details = {
        "total_rows": total_rows,
        "candidate_pair_count": pair_count,
        "flagged_row_count": len(flagged_row_ids),
        "skipped_short_count": skipped_short_count,
        "duplicate_rate": round(duplicate_rate, 4),
        "threshold": threshold,
        "num_perm": num_perm,
        "example_pairs": example_pairs,
    }

    return CheckResult(score=score, warning=warning, details=details)
