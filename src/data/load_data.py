"""
load_data.py
------------
Task 1, Step 1: Load the raw complaint datasets and print a first
"sanity check" of what we're working with.

Why this file exists on its own:
Every other script (EDA, training, the Streamlit apps) needs the same
loading logic. If we wrote `pd.read_csv(...)` in five different places
and then changed a column name, we'd have to fix it in five places.
Instead, everything imports `load_complaints()` from here.
"""

import pandas as pd
from pathlib import Path

#  so it works no matter where you run it from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/data -> src -> project root
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_csv(filepath) -> pd.DataFrame:
    """
    General-purpose CSV loader. Parses 'timestamp' as a real datetime
    if that column exists in the file.

    Parameters
    ----------
    filepath : str or Path
        Full path to the CSV file to load.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(filepath, parse_dates=["timestamp"])


def load_complaints(split: str = "train") -> pd.DataFrame:
    """
    Load either the train or test complaints CSV, from this project's
    known data/raw/ folder and complaints_{split}.csv naming pattern.

    Parameters
    ----------
    split : str
        Either "train" or "test".

    Returns
    -------
    pd.DataFrame
        The raw complaints data, with timestamp parsed as a real datetime.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    file_path = RAW_DATA_DIR / f"complaints_{split}.csv"
    return load_csv(file_path)


if __name__ == "__main__":
    # This block only runs when you execute `python load_data.py` directly,
    # NOT when another file does `from load_data import load_complaints`.
    # It's a common Python pattern for "quick manual test of this file".
    train_df = load_complaints("train")
    test_df = load_complaints("test")

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print("\nColumn dtypes:")
    print(train_df.dtypes)
    print("\nFirst 3 rows:")
    print(train_df.head(3))
