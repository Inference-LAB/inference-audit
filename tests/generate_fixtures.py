"""
tests/generate_fixtures.py

Run once to generate all test fixtures. Committed fixtures are
regenerated from this script -- not edited by hand.

Per Role Guide Week 1 fixture specification table.
"""
import pandas as pd
import random
from pathlib import Path

random.seed(42)  # Reproducible fixture generation.
# Change this seed and all fixtures change -- do not change it.

FIXTURE_DIR = Path("tests/fixtures")
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# label_distribution fixtures
# ============================================================

def make_balanced(n_per_class=200, n_classes=5) -> pd.DataFrame:
    """1000 rows, 5 classes, 200 each. Expected score: 90-100."""
    classes = [f"class_{i}" for i in range(n_classes)]
    rows = []
    for cls in classes:
        for j in range(n_per_class):
            rows.append({
                "text": f"Sample text {cls} number {j} with enough length to detect",
                "label": cls,
            })
    random.shuffle(rows)
    return pd.DataFrame(rows)


def make_severe_imbalance() -> pd.DataFrame:
    """1000 rows: class_A has 950, class_B has 50. Expected score: 0-30."""
    rows = (
        [{"text": f"text {i} positive", "label": "class_A"} for i in range(950)] +
        [{"text": f"text {i} negative", "label": "class_B"} for i in range(50)]
    )
    random.shuffle(rows)
    return pd.DataFrame(rows)


# ============================================================
# near_duplicates fixtures
#
# NOTE (Week 3 fix): the original version of these two functions used
# a numeric template ("unique sample number {i}...") where only a
# number changed between rows. That template IS a near-duplicate
# pattern by our own Week 1 definition (minor-addition/variation type)
# -- so "no_duplicates.csv" wasn't actually duplicate-free, and
# check_near_duplicates correctly (but confusingly) flagged it as
# heavily duplicated. Fixed by generating genuinely varied sentence
# content from combined vocabulary pools instead of one template.
# ============================================================

_SUBJECTS = ["the weather", "my neighbor", "the football match", "this recipe", "the election results",
             "a new phone", "the movie plot", "traffic downtown", "the school exam", "my garden",
             "the stock market", "a wedding", "the internet outage", "local news", "a birthday party",
             "the airport delay", "a science experiment", "the concert", "a job interview", "the hospital visit",
             "the university lecture", "a family dinner", "the office meeting", "a road trip", "the art exhibit",
             "a swimming competition", "the courtroom hearing", "a cooking class", "the museum tour", "a charity event"]
_VERBS = ["surprised everyone", "changed completely", "was disappointing", "went as expected", "caused chaos",
          "impressed the crowd", "took longer than planned", "was cancelled suddenly", "improved this year", "confused most people",
          "exceeded expectations", "fell apart quickly", "sparked a debate", "ran smoothly", "left people speechless"]
_DETAILS = ["according to reports", "based on what I saw", "much to our surprise", "after several delays",
            "without any warning", "despite the forecast", "for the third time", "in record time",
            "against all odds", "as many predicted", "right before sunset", "during the holiday season",
            "in front of a large crowd", "earlier than scheduled", "for reasons still unclear"]
_EXTRAS = ["near downtown", "this past weekend", "in a small town", "on live television", "just yesterday",
           "according to witnesses", "in the northern district", "before the deadline", "under new management", "for the first time"]


def make_no_duplicates(n=500) -> pd.DataFrame:
    """
    500 genuinely varied samples built from 4 independent slots (not 3)
    over larger vocabulary pools, to reduce the chance of any two rows
    accidentally sharing enough substring content to look near-duplicate.
    Expected score: 90-100.
    """
    rows = []
    used = set()
    while len(rows) < n:
        s = (f"{random.choice(_SUBJECTS)} {random.choice(_VERBS)} "
             f"{random.choice(_DETAILS)} {random.choice(_EXTRAS)}")
        if s not in used:
            used.add(s)
            rows.append({"text": s, "label": "pos" if len(rows) % 2 == 0 else "neg"})
    return pd.DataFrame(rows)


def make_with_duplicates() -> pd.DataFrame:
    """
    500 rows: 440 genuinely varied + 50 exact duplicates seeded in +
    10 cross-label duplicates. Expected score: 40-65 (currently open --
    see PR discussion; approved formula produces 78.2).
    """
    base = make_no_duplicates(n=440)
    duplicates = [{"text": "this is a repeated sample text that appears multiple times",
                   "label": "pos"} for _ in range(50)]
    cross = [{"text": "this specific text has two different labels assigned to it",
              "label": "neg" if i % 2 == 0 else "pos"} for i in range(10)]
    all_rows = base.to_dict("records") + duplicates + cross
    random.shuffle(all_rows)
    return pd.DataFrame(all_rows)


# ============================================================
# language_contamination fixtures
# ============================================================

ROMAN_URDU_SAMPLES = [
    "yeh bohat acha din tha aj", "mera dil khush hai", "kya haal hai bhai",
    "bohat mazedar khana tha", "yeh drama bilkul boring hai", "aj mausam acha hai",
    "mujhe yeh pasand nahi aya", "sab theek hai alhamdulillah",
]
ENGLISH_SAMPLES = [
    "This was a great day today", "My heart is happy", "How are you brother",
    "The food was very delicious", "This drama is quite boring", "The weather is nice today",
]


def make_clean_urdu(n=500) -> pd.DataFrame:
    """500 Roman Urdu samples, 0 English. Expected score: 90-100."""
    rows = [{"text": f"{random.choice(ROMAN_URDU_SAMPLES)} number {i}", "label": "pos"}
            for i in range(n)]
    return pd.DataFrame(rows)


def make_contaminated(n_urdu=400, n_english=100) -> pd.DataFrame:
    """400 Roman Urdu + 100 English (20% contamination). Expected score: 0-30."""
    rows = (
        [{"text": f"{random.choice(ROMAN_URDU_SAMPLES)} number {i}", "label": "pos"}
         for i in range(n_urdu)] +
        [{"text": f"{random.choice(ENGLISH_SAMPLES)} number {i}", "label": "pos"}
         for i in range(n_english)]
    )
    random.shuffle(rows)
    return pd.DataFrame(rows)


# ============================================================
# missing_values fixtures
# ============================================================

def make_clean_dataset_mv(n=500) -> pd.DataFrame:
    """500 rows, no nulls, no whitespace, all text > 10 chars. Expected score: 95-100."""
    rows = [{"text": f"This is a valid sample sentence number {i} with real content",
             "label": "pos"} for i in range(n)]
    return pd.DataFrame(rows)


def make_dirty_dataset_mv(n=500, n_null=25, n_whitespace=15, n_short=20) -> pd.DataFrame:
    """
    500 rows: 25 null texts, 15 whitespace-only, 20 very short.
    Expected score: 30-50 (currently open -- see PR discussion;
    approved formula produces 88).
    """
    n_clean = n - n_null - n_whitespace - n_short
    rows = [{"text": f"Valid sentence number {i} with real content here", "label": "pos"}
            for i in range(n_clean)]
    rows += [{"text": None, "label": "pos"} for _ in range(n_null)]
    rows += [{"text": "   ", "label": "pos"} for _ in range(n_whitespace)]
    rows += [{"text": "hi", "label": "pos"} for _ in range(n_short)]
    random.shuffle(rows)
    return pd.DataFrame(rows)


# ============================================================
# annotation_consistency fixtures
# ============================================================

def make_high_confidence(n=500) -> pd.DataFrame:
    """500 rows with conf column, all values > 0.7. Expected score: 90-100."""
    rows = [{"text": f"sample {i}", "label": "pos", "confidence": round(random.uniform(0.71, 1.0), 2)}
            for i in range(n)]
    return pd.DataFrame(rows)


def make_low_confidence(n=500, pct_low=0.3) -> pd.DataFrame:
    """500 rows with conf column, 30% of values < 0.4. Expected score: 30-50."""
    n_low = int(n * pct_low)
    n_high = n - n_low
    rows = [{"text": f"sample {i}", "label": "pos", "confidence": round(random.uniform(0.0, 0.39), 2)}
            for i in range(n_low)]
    rows += [{"text": f"sample {i}", "label": "pos", "confidence": round(random.uniform(0.4, 1.0), 2)}
             for i in range(n_high)]
    random.shuffle(rows)
    return pd.DataFrame(rows)


# ============================================================
# Generate all fixtures
# ============================================================

if __name__ == "__main__":
    datasets = {
        "balanced.csv": make_balanced(),
        "severe_imbalance.csv": make_severe_imbalance(),
        "no_duplicates.csv": make_no_duplicates(),
        "with_duplicates.csv": make_with_duplicates(),
        "clean_urdu.csv": make_clean_urdu(),
        "contaminated.csv": make_contaminated(),
        "clean_dataset.csv": make_clean_dataset_mv(),
        "dirty_dataset.csv": make_dirty_dataset_mv(),
        "high_confidence.csv": make_high_confidence(),
        "low_confidence.csv": make_low_confidence(),
    }

    for filename, df in datasets.items():
        path = FIXTURE_DIR / filename
        df.to_csv(path, index=False)
        print(f"Generated {path}: {len(df)} rows")
