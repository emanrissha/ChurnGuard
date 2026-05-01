import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pathlib import Path
from loguru import logger
from api.schemas import CustomerFeatures, PredictionResponse
from src.features.engineering import engineer_features

router = APIRouter()

MODEL_PATH = Path("models/xgb_v1.pkl")
SHAP_PATH = Path("models/shap_explainer_v1.pkl")

try:
    model = joblib.load(MODEL_PATH)
    explainer = joblib.load(SHAP_PATH)
    logger.info("Model and SHAP explainer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    explainer = None

COLUMN_MAP = {
    "InternetService_Fiber_optic": "InternetService_Fiber optic",
    "Contract_Month_to_month": "Contract_Month-to-month",
    "Contract_One_year": "Contract_One year",
    "Contract_Two_year": "Contract_Two year",
    "PaymentMethod_Bank_transfer": "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit_card": "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic_check": "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed_check": "PaymentMethod_Mailed check",
}

REVENUE_PER_CUSTOMER = 8000


def input_to_dataframe(features: CustomerFeatures) -> pd.DataFrame:
    data = features.model_dump()
    df = pd.DataFrame([data])
    df = df.rename(columns=COLUMN_MAP)
    return df


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = input_to_dataframe(features)
    df = engineer_features(df)

    churn_prob = float(model.predict_proba(df)[:, 1][0])
    churn_pred = int(churn_prob >= 0.5)

    if churn_prob >= 0.7:
        risk_level = "HIGH"
    elif churn_prob >= 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    shap_vals = explainer.shap_values(df)[0]
    shap_series = pd.Series(shap_vals, index=df.columns)
    top_risk = shap_series.nlargest(5).round(4).to_dict()
    top_protective = shap_series.nsmallest(5).round(4).to_dict()

    return PredictionResponse(
        churn_probability=round(churn_prob, 4),
        churn_prediction=churn_pred,
        risk_level=risk_level,
        top_risk_factors=top_risk,
        top_protective_factors=top_protective,
        estimated_revenue_at_risk_ils=round(churn_prob * REVENUE_PER_CUSTOMER, 2)
    )