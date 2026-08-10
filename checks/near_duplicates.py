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
    """
    Builds a MinHash signature from character 3-grams of lowercased text.

    Lowercasing BEFORE shingling is a hard requirement, not optional --
    per Week 1 finding, case-scrambled duplicates found in real RUEmoCorp
    data would be undercounted without it (e.g. "Him" and "him" produce
    completely different 3-grams if case-sensitive).
    """
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

    Args:
        df:         The dataset as a pandas DataFrame.
        text_col:   Name of the column containing text to check.
        threshold:  Jaccard similarity threshold for LSH bucketing.
                    Default 0.8, per Week 1 design doc finding (correctly
                    separates true duplicates 0.851-1.000 from distinct
                    pairs 0.053-0.164 on 20 real RUEmoCorp pairs, WITH
                    lowercasing applied).
        num_perm:   Number of hash permutations for MinHash signatures.
                    Higher = more accurate estimate, more compute cost.

    Returns:
        CheckResult with score (0-100), warning (str or None), and
        details (dict with candidate pair count, flagged row count,
        and a few example pairs for the report).

    Never raises. Returns CheckResult(score=None, ...) for edge cases:
    missing column, empty dataframe, or fewer than 2 rows (no possible
    pairs to compare).
    """
    # Defense-in-depth only: primary column validation happens once,
    # centrally, in loader.py before any check runs (per design doc's
    # Public API section). This guard exists so the check still honors
    # its "never raises" contract if ever called directly (e.g. in an
    # isolated unit test), not as the main validation path.
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

    # --- Build MinHash signatures for every row, index into LSH ---
    # NOTE: text shorter than 3 characters produces ZERO 3-grams, so
    # MinHash.update() is never called and the signature stays at its
    # default untouched state -- meaning ANY two short texts would
    # trivially look identical to LSH regardless of actual content
    # (found via live edge-case testing: "hi" and "no" were flagged as
    # a duplicate pair). These rows are excluded from comparison
    # entirely rather than silently mismatched.
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

    # --- Find candidate pairs (each pair counted once) ---
    candidate_pairs = set()
    for idx, mh in minhashes.items():
        matches = lsh.query(mh)
        for match_str in matches:
            match_idx = int(match_str) if str(idx).isdigit() else match_str
            other_idx = match_str if isinstance(idx, int) else match_str
            # Normalize to avoid (a,b) and (b,a) both being counted
            pair = tuple(sorted((str(idx), str(match_str))))
            if pair[0] != pair[1]:
                candidate_pairs.add(pair)

    pair_count = len(candidate_pairs)

    # Rows involved in at least one duplicate pair (for interpretability
    # in details -- the approved formula itself uses pair_count, not this)
    flagged_row_ids = set()
    for a, b in candidate_pairs:
        flagged_row_ids.add(a)
        flagged_row_ids.add(b)

    # --- Score formula, per design doc: 100 x (1 - duplicate_rate) ---
    # NOTE: using flagged_row_count, NOT pair_count, as the numerator.
    # A cluster of k identical rows produces C(k,2) pairs -- e.g. 50
    # identical rows alone produce 1225 pairs, which can exceed
    # total_rows and blow the formula past 0 for the wrong reason.
    # Row-based rate reflects "what fraction of the dataset is affected"
    # rather than "how many pairwise relationships exist," which is the
    # actually meaningful quantity here. (Found via live testing against
    # the with_duplicates.csv fixture -- original pair-based formula
    # produced duplicate_rate=2.6, clamping score to 0.)
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

    # A few example pairs for the report / debugging -- capped at 5
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
