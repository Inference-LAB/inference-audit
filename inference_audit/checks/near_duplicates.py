"""
inference_audit/checks/near_duplicates.py

Detects verbatim and near-verbatim duplicate samples using MinHash + LSH
over character 3-grams. Threshold and preprocessing choices are backed
by the Week 1 design doc findings (20 real RUEmoCorp pairs tested).

PERFORMANCE (Week 5, Lead Engineer): the original implementation built
one MinHash per ROW, even when many rows share exact identical text --
on a real 134K-row corpus this took ~6 minutes. This version builds one
MinHash per UNIQUE normalized text instead, and computes pair counts /
flagged rows arithmetically from group sizes rather than enumerating
every row pair. Parameters (threshold, num_perm) and output shape
(CheckResult schema) are unchanged -- this is a "do less redundant
work" optimization, not a behavior change. Verified via
tests/test_near_duplicates_regression.py against the original
row-by-row implementation on all existing fixtures before replacing it.
"""

import pandas as pd
from datasketch import MinHash, MinHashLSH
from inference_audit.report import CheckResult

MIN_SHINGLE_LENGTH = 3


def _to_minhash(text: str, num_perm: int = 128) -> MinHash:
    """
    Builds a MinHash signature from character 3-grams of lowercased text.

    Uses update_batch() instead of calling update() once per shingle --
    fewer individual Python-level calls for the same result.
    """
    m = MinHash(num_perm=num_perm)
    encoded = text.encode("utf-8")
    shingles = [encoded[i:i + 3] for i in range(len(encoded) - 2)]
    if shingles:
        m.update_batch(shingles)
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
    missing column, empty dataframe, fewer than 2 rows, or when every
    row is too short to compare (zero comparable rows after filtering
    is the same "nothing to measure" situation as the single-row case,
    just reached a different way -- must not silently report a perfect
    score).
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

    texts = df[text_col].fillna("").astype(str)
    normalized = texts.str.lower().str.strip()

    comparable_mask = normalized.str.len() >= MIN_SHINGLE_LENGTH
    skipped_short_count = int((~comparable_mask).sum())

    if not comparable_mask.any():
        return CheckResult(
            score=None,
            warning=(
                f"All {total_rows} rows were too short to compare "
                f"(< {MIN_SHINGLE_LENGTH} characters) — nothing could be measured."
            ),
            details={
                "error": "all_rows_too_short",
                "total_rows": total_rows,
                "skipped_short_count": skipped_short_count,
            },
        )

    # --- Group row indices by exact normalized text. Rows sharing exact
    # text are duplicates of each other by definition (Jaccard = 1.0),
    # so this avoids running MinHash/LSH between them at all. ---
    groups = {}
    for idx, norm_text in zip(df.index[comparable_mask], normalized[comparable_mask]):
        groups.setdefault(norm_text, []).append(str(idx))

    unique_texts = list(groups.keys())

    # --- One MinHash per unique text, not per row. ---
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhash_cache = {}
    for u_text in unique_texts:
        mh = _to_minhash(u_text, num_perm=num_perm)
        minhash_cache[u_text] = mh
        lsh.insert(u_text, mh)

    # --- Find near-duplicate matches BETWEEN distinct unique texts. ---
    unique_text_pairs = set()
    for u_text in unique_texts:
        for match in lsh.query(minhash_cache[u_text]):
            if match != u_text:
                unique_text_pairs.add(tuple(sorted((u_text, match))))

    # --- Compute pair_count and flagged rows arithmetically from group
    # sizes, rather than enumerating every individual row pair. This
    # reproduces the same counts the original row-by-row version would
    # have produced (exact-duplicate groups of size k contribute
    # C(k,2) pairs; a near-duplicate match between two unique texts
    # contributes one pair per row-combination between them). ---
    pair_count = 0
    flagged_unique_texts = set()

    for u_text, indices in groups.items():
        k = len(indices)
        if k > 1:
            pair_count += k * (k - 1) // 2
            flagged_unique_texts.add(u_text)

    for u1, u2 in unique_text_pairs:
        pair_count += len(groups[u1]) * len(groups[u2])
        flagged_unique_texts.add(u1)
        flagged_unique_texts.add(u2)

    flagged_row_ids = set()
    for u_text in flagged_unique_texts:
        flagged_row_ids.update(groups[u_text])

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

    # Example pairs for the report -- best-effort selection, capped at 5.
    # NOTE: not guaranteed to select the identical example rows the
    # original set-based implementation would have shown (that version
    # iterated an unordered Python set, so its own examples were never
    # deterministic either). See regression test for what IS guaranteed
    # to match exactly: score, counts, and rates.
    example_pairs = []
    for u_text, indices in groups.items():
        if len(example_pairs) >= 5:
            break
        if len(indices) > 1:
            a, b = indices[0], indices[1]
            example_pairs.append({
                "row_a": texts.loc[int(a) if a.isdigit() else a][:80],
                "row_b": texts.loc[int(b) if b.isdigit() else b][:80],
            })
    for u1, u2 in unique_text_pairs:
        if len(example_pairs) >= 5:
            break
        a, b = groups[u1][0], groups[u2][0]
        example_pairs.append({
            "row_a": texts.loc[int(a) if a.isdigit() else a][:80],
            "row_b": texts.loc[int(b) if b.isdigit() else b][:80],
        })

    details = {
        "total_rows": total_rows,
        "unique_text_count": len(unique_texts),
        "candidate_pair_count": pair_count,
        "flagged_row_count": len(flagged_row_ids),
        "skipped_short_count": skipped_short_count,
        "duplicate_rate": round(duplicate_rate, 4),
        "threshold": threshold,
        "num_perm": num_perm,
        "example_pairs": example_pairs,
    }

    return CheckResult(score=score, warning=warning, details=details)