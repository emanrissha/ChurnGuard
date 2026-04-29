import pandas as pd
import numpy as np
from loguru import logger
from src.data.loader import load_raw_data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting data cleaning...")
    df = df.copy()

    # Fix TotalCharges — spaces become NaN, then drop
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["TotalCharges"])
    dropped = before - len(df)
    logger.info(f"Dropped {dropped} rows with blank TotalCharges")

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop customerID — not a feature
    df = df.drop(columns=["customerID"])

    # Encode binary Yes/No columns
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # Encode gender
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    # One-hot encode multi-class columns
    multi_cols = ["InternetService", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=False)

    # Convert all bool columns from get_dummies to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    logger.info(f"Cleaned data shape: {df.shape}")
    logger.info(f"Churn rate: {df['Churn'].mean():.1%}")
    return df


def get_features_and_target(df: pd.DataFrame):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    logger.info(f"Features: {X.shape[1]} columns | Target: {y.value_counts().to_dict()}")
    return X, y


if __name__ == "__main__":
    df = load_raw_data()
    df_clean = clean_data(df)
    print("\n=== Cleaned Columns ===")
    print(df_clean.columns.tolist())
    print("\n=== Sample ===")
    print(df_clean.head(3).to_string())
    X, y = get_features_and_target(df_clean)
    print(f"\nX shape: {X.shape}")
    print(f"y distribution:\n{y.value_counts()}")