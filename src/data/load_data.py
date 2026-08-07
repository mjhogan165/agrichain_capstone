import pandas as pd
from pathlib import Path

#  So it works no matter where its run from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/data -> src -> project root
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_csv(filepath) -> pd.DataFrame:
    """
    CSV loader. Parses 'timestamp' as a Datetime object, if it exists.
    """
    return pd.read_csv(filepath, parse_dates=["timestamp"])


def load_complaints(split: str = "train") -> pd.DataFrame:
    """
    Load the train or test complaints CSV.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    file_path = RAW_DATA_DIR / f"complaints_{split}.csv"
    return load_csv(file_path)


if __name__ == "__main__":
    # This block only runs when you execute `python load_data.py` directly,
    train_df = load_complaints("train")
    test_df = load_complaints("test")

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print("\nColumn dtypes:")
    print(train_df.dtypes)
    print("\nFirst 3 rows:")
    print(train_df.head(3))
