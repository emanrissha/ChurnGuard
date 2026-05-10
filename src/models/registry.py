import joblib
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

MODELS_DIR = Path("models")


def save_model(model, model_name: str, metrics: dict) -> Path:
    MODELS_DIR.mkdir(exist_ok=True)
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"{model_name}_{version}.pkl"
    meta_path = MODELS_DIR / f"{model_name}_{version}_metadata.json"

    joblib.dump(model, model_path)
    metadata = {
        "model_name": model_name,
        "version": version,
        "saved_at": datetime.now().isoformat(),
        "metrics": metrics,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model saved: {model_path}")
    logger.info(f"Metadata saved: {meta_path}")
    return model_path


def load_latest_model(model_name: str):
    models = sorted(MODELS_DIR.glob(f"{model_name}_*.pkl"))
    if not models:
        raise FileNotFoundError(f"No model found for {model_name}")
    latest = models[-1]
    logger.info(f"Loading model: {latest}")
    return joblib.load(latest)


def list_models() -> list:
    meta_files = sorted(MODELS_DIR.glob("*_metadata.json"))
    models = []
    for f in meta_files:
        with open(f) as m:
            models.append(json.load(m))
    return models