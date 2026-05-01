import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

MODEL_PATH = Path("models/xgb_v1.pkl")
SHAP_PATH = Path("models/shap_explainer_v1.pkl")


def build_explainer(model, X_train: pd.DataFrame):
    logger.info("Building SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, SHAP_PATH)
    logger.info(f"SHAP explainer saved to {SHAP_PATH}")
    return explainer


def get_shap_values(explainer, X: pd.DataFrame):
    logger.info(f"Computing SHAP values for {len(X)} samples...")
    shap_values = explainer.shap_values(X)
    return shap_values


def explain_customer(explainer, X: pd.DataFrame, customer_idx: int) -> dict:
    customer = X.iloc[[customer_idx]]
    shap_vals = explainer.shap_values(customer)[0]

    explanation = pd.Series(shap_vals, index=X.columns)
    top_risk = explanation.nlargest(5)
    top_protective = explanation.nsmallest(5)

    result = {
        "customer_index": customer_idx,
        "top_risk_factors": top_risk.round(4).to_dict(),
        "top_protective_factors": top_protective.round(4).to_dict(),
    }
    return result


def plot_global_importance(shap_values, X: pd.DataFrame, save_path: str = None):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X,
        plot_type="bar",
        max_display=15,
        show=False
    )
    plt.title("Global Feature Importance (SHAP)", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved global SHAP plot to {save_path}")
    plt.show()


def plot_customer_waterfall(explainer, X: pd.DataFrame, customer_idx: int, save_path: str = None):
    shap_explanation = explainer(X.iloc[[customer_idx]])
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_explanation[0], max_display=12, show=False)
    plt.title(f"Customer {customer_idx} — Churn Risk Breakdown", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved waterfall plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    from src.data.loader import load_raw_data
    from src.data.preprocessor import clean_data, get_features_and_target
    from src.features.engineering import engineer_features
    from sklearn.model_selection import train_test_split

    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}")

    explainer = build_explainer(model, X_train)
    shap_values = get_shap_values(explainer, X_test)

    print("\n=== Global Top 5 Features (mean |SHAP|) ===")
    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns
    ).sort_values(ascending=False).head(5)
    print(mean_shap.round(4).to_string())

    print("\n=== Customer 0 Explanation ===")
    explanation = explain_customer(explainer, X_test, customer_idx=0)
    print("Top risk factors:")
    for feat, val in explanation["top_risk_factors"].items():
        print(f"  {feat}: +{val}")
    print("Top protective factors:")
    for feat, val in explanation["top_protective_factors"].items():
        print(f"  {feat}: {val}")