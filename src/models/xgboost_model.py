import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, RandomizedSearchCV
)
from src.models.evaluator import evaluate_model

MODEL_PATH = Path("models/xgb_v1.pkl")


def train_xgboost(X, y) -> tuple:
    logger.info("Training XGBoost champion model with hyperparameter tuning...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

    param_grid = {
        "n_estimators": [300, 500, 700],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
    }

    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    logger.info("Running RandomizedSearchCV (20 iterations, 5-fold CV)...")
    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=20,
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)

    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best CV F1: {search.best_score_:.4f}")

    model = search.best_estimator_
    results = evaluate_model(model, X_test, y_test, "XGBoost (tuned)")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    return model, X_test, y_test, results


if __name__ == "__main__":
    from src.data.loader import load_raw_data
    from src.data.preprocessor import clean_data, get_features_and_target
    from src.features.engineering import engineer_features

    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    X, y = get_features_and_target(df)

    model, X_test, y_test, results = train_xgboost(X, y)

    print("\n=== XGBoost Tuned Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    print("\n=== Top 10 Features by Importance ===")
    feat_imp = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False).head(10)
    print(feat_imp.round(4).to_string())