"""Tests for inference_audit/loader.py"""

import pandas as pd
import pytest

from inference_audit.loader import load_dataset


def _write_csv(tmp_path, filename, df):
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return str(path)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset("does_not_exist.csv", "label", "text")


def test_unsupported_extension(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("not a real dataset")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_dataset(str(path), "label", "text")


def test_empty_dataset(tmp_path):
    df = pd.DataFrame({"label": [], "text": []})
    path = _write_csv(tmp_path, "empty.csv", df)
    with pytest.raises(ValueError, match="empty"):
        load_dataset(path, "label", "text")


def test_missing_required_columns(tmp_path):
    df = pd.DataFrame({"label": ["a", "b"], "other": ["x", "y"]})
    path = _write_csv(tmp_path, "missing_col.csv", df)
    with pytest.raises(ValueError, match="not found in dataset"):
        load_dataset(path, "label", "text")


def test_malformed_json_gives_clear_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json at all")
    with pytest.raises(ValueError, match="Could not parse"):
        load_dataset(str(path), "label", "text")


def test_valid_csv_loads_correctly(tmp_path):
    df = pd.DataFrame({"label": ["a", "b", "c"], "text": ["one", "two", "three"]})
    path = _write_csv(tmp_path, "valid.csv", df)
    result = load_dataset(path, "label", "text")
    assert len(result) == 3
    assert list(result["label"]) == ["a", "b", "c"]


def test_valid_json_loads_correctly(tmp_path):
    path = tmp_path / "valid.json"
    df = pd.DataFrame({"label": ["a", "b"], "text": ["one", "two"]})
    df.to_json(path, orient="records")
    result = load_dataset(str(path), "label", "text")
    assert len(result) == 2


def test_valid_parquet_loads_correctly(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "valid.parquet"
    df = pd.DataFrame({"label": ["a", "b"], "text": ["one", "two"]})
    df.to_parquet(path)
    result = load_dataset(str(path), "label", "text")
    assert len(result) == 2


def test_column_names_with_whitespace_are_stripped(tmp_path):
    df = pd.DataFrame({" label ": ["a", "b"], "text": ["one", "two"]})
    path = _write_csv(tmp_path, "whitespace_cols.csv", df)
    result = load_dataset(path, "label", "text")
    assert "label" in result.columns
    assert " label " not in result.columns