import pandas as pd
import urllib.request
from pathlib import Path
from loguru import logger

RAW_DATA_PATH = Path("data/raw/telco_churn.csv")
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def ensure_data_exists(path: Path = RAW_DATA_PATH) -> None:
    if not path.exists():
        logger.info(f"Data not found at {path} — downloading from source...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
        logger.info(f"Downloaded successfully to {path}")
    else:
        logger.info(f"Data found at {path}")


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    ensure_data_exists(path)
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
    return df


def basic_info(df: pd.DataFrame) -> None:
    print("\n=== Shape ===")
    print(df.shape)

    print("\n=== Churn Distribution ===")
    churn_counts = df["Churn"].value_counts()
    churn_pct = df["Churn"].value_counts(normalize=True).mul(100).round(1)
    print(churn_counts)
    print(churn_pct.astype(str) + "%")

    print("\n=== Missing Values ===")
    missing = df.isnull().sum()
    if missing.any():
        print(missing[missing > 0])
    else:
        print("No missing values detected")

    print("\n=== Column Types ===")
    print(df.dtypes)

    print("\n=== Sample Rows ===")
    print(df.head(3).to_string())


if __name__ == "__main__":
    df = load_raw_data()
    basic_info(df)