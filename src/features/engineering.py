import pandas as pd
from loguru import logger


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering...")
    df = df.copy()

    # --- Tenure cohorts ---
    df["tenure_cohort"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=[1, 2, 3, 4]
    ).astype(int)

    # --- Revenue features ---
    df["avg_monthly_revenue"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["revenue_per_product"] = df["MonthlyCharges"] / (
        df[["PhoneService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies"]].sum(axis=1).replace(0, 1)
    )

    # --- Usage velocity ---
    df["charge_increase_rate"] = (
        df["MonthlyCharges"] - df["avg_monthly_revenue"]
    ) / df["avg_monthly_revenue"].replace(0, 1)

    # --- Product adoption score (0-7) ---
    df["product_count"] = df[[
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]].sum(axis=1)

    # --- Risk flags ---
    df["is_month_to_month"] = df["Contract_Month-to-month"].astype(int)
    df["is_electronic_check"] = df["PaymentMethod_Electronic check"].astype(int)
    df["is_fiber"] = df["InternetService_Fiber optic"].astype(int)
    df["is_senior_alone"] = (
        (df["SeniorCitizen"] == 1) &
        (df["Partner"] == 0) &
        (df["Dependents"] == 0)
    ).astype(int)

    # --- High risk score (composite) ---
    df["risk_score"] = (
        df["is_month_to_month"] +
        df["is_electronic_check"] +
        df["is_fiber"] +
        df["is_senior_alone"] +
        (df["tenure"] < 12).astype(int) +
        (df["product_count"] <= 1).astype(int)
    )

    # --- Loyalty flag ---
    df["is_loyal"] = (df["tenure"] >= 24).astype(int)

    # --- High spender flag ---
    df["is_high_spender"] = (
        df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)
    ).astype(int)

    logger.info(f"Feature engineering complete — shape: {df.shape}")
    logger.info(f"New features added: {df.shape[1] - 26}")
    return df


if __name__ == "__main__":
    from src.data.loader import load_raw_data
    from src.data.preprocessor import clean_data

    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)

    print("\n=== All Features ===")
    print(df.columns.tolist())
    print(f"\nTotal features: {df.shape[1]}")

    print("\n=== New Feature Stats ===")
    new_features = [
        "tenure_cohort", "avg_monthly_revenue", "revenue_per_product",
        "charge_increase_rate", "product_count", "risk_score",
        "is_loyal", "is_high_spender", "is_senior_alone"
    ]
    print(df[new_features].describe().round(2).to_string())