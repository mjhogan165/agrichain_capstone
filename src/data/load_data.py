import pandas as pd
from pathlib import Path

#  So it works no matter where its run from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_csv(filepath) -> pd.DataFrame:
    return pd.read_csv(filepath, parse_dates=["timestamp"])


def load_complaints(split: str = "train") -> pd.DataFrame:
    """
    Load the train or test complaints CSV.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    file_path = RAW_DATA_DIR / f"complaints_{split}.csv"
    return load_csv(file_path)
