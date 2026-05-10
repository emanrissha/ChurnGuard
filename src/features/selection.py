import pandas as pd
import numpy as np
from loguru import logger


def get_top_features(model, feature_names: list, top_n: int = 20) -> list:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top = [feature_names[i] for i in indices]
    logger.info(f"Top {top_n} features selected")
    return top


def drop_low_importance_features(
    X: pd.DataFrame, model, threshold: float = 0.001
) -> pd.DataFrame:
    importances = pd.Series(model.feature_importances_, index=X.columns)
    keep = importances[importances >= threshold].index.tolist()
    dropped = len(X.columns) - len(keep)
    logger.info(f"Dropped {dropped} low-importance features (threshold={threshold})")
    return X[keep]


def get_feature_importance_df(model, feature_names: list) -> pd.DataFrame:
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df