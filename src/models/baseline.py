import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.models.evaluator import evaluate_model


def train_baseline(X, y) -> tuple:
    logger.info("Training Logistic Regression baseline...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
    logger.info(f"CV F1 scores: {cv_scores.round(4)}")
    logger.info(f"Mean CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipeline.fit(X_train, y_train)
    results = evaluate_model(pipeline, X_test, y_test, "Logistic Regression")
    return pipeline, X_test, y_test, results


if __name__ == "__main__":
    from src.data.loader import load_raw_data
    from src.data.preprocessor import clean_data, get_features_and_target
    from src.features.engineering import engineer_features

    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    X, y = get_features_and_target(df)

    model, X_test, y_test, results = train_baseline(X, y)
    print("\n=== Baseline Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")