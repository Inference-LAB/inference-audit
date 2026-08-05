"""
inference_audit/loader.py

Turns a file path into a clean, validated pandas DataFrame.

This is the only place in the codebase that should check whether a
file exists, whether it's a supported format, or whether the required
columns are present. Every check function downstream is allowed to
assume the DataFrame it receives is already valid — if that assumption
is ever wrong, the bug belongs here, not in five different places.
"""

import pandas as pd
from pathlib import Path

SUPPORTED_FORMATS = {".csv", ".json", ".parquet"}


def load_dataset(path: str, label_col: str, text_col: str) -> pd.DataFrame:
    """
    Loads a dataset file and confirms it's usable before returning it.

    Args:
        path:      Path to the dataset file. Format is detected from
                   the file extension, not by inspecting the content.
        label_col: Name of the column that should contain class labels.
        text_col:  Name of the column that should contain the text
                   being audited.

    Returns:
        A pandas DataFrame with at least one row and both required
        columns present. Column names have leading/trailing whitespace
        stripped, so a CSV exported with " label" instead of "label"
        still works.

    Raises:
        FileNotFoundError: the path doesn't point to a real file.
        ValueError: the file extension isn't one of .csv/.json/.parquet,
                    the file has 0 rows, the file can't be parsed, or
                    label_col/text_col aren't present in the data.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format: '{extension}'. "
            f"Supported formats are: {sorted(SUPPORTED_FORMATS)}"
        )

    df = _read_by_format(file_path, extension)

    # Column names sometimes carry stray whitespace from a spreadsheet
    # export ("label " instead of "label"). Stripping here means a
    # user doesn't get a confusing "column not found" error over a
    # single invisible space.
    df.columns = df.columns.str.strip()

    if len(df) == 0:
        raise ValueError(f"Dataset is empty (0 rows): {path}")

    missing_columns = [c for c in (label_col, text_col) if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Column(s) not found in dataset: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


def _read_by_format(file_path: Path, extension: str) -> pd.DataFrame:
    """
    Reads the file with the pandas function matching its format.

    Kept separate from load_dataset() so the format-specific parsing
    logic — and its error handling — doesn't clutter the main
    validation flow.
    """
    if extension == ".csv":
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(
                f"Could not parse '{file_path}' as CSV. Original error: {e}"
            ) from e

    if extension == ".json":
        try:
            return pd.read_json(file_path, orient="records")
        except ValueError as e:
            raise ValueError(
                f"Could not parse '{file_path}' as JSON.\n"
                f"Expected records-oriented JSON, e.g.:\n"
                f'[{{"text": "...", "label": "..."}}, {{"text": "...", "label": "..."}}]\n'
                f"Original error: {e}"
            ) from e

    if extension == ".parquet":
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            raise ValueError(
                f"Could not parse '{file_path}' as Parquet. Original error: {e}"
            ) from e

    # This line should be unreachable — load_dataset() already checked
    # the extension against SUPPORTED_FORMATS before calling this
    # function. It's here as a safety net in case a new format is
    # added to SUPPORTED_FORMATS without adding a matching branch here.
    raise ValueError(f"No reader implemented for format: '{extension}'")