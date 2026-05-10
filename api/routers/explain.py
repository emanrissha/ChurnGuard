import joblib
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException
from src.data.loader import load_raw_data
from src.data.preprocessor import clean_data, get_features_and_target
from src.features.engineering import engineer_features

router = APIRouter()

MODEL_PATH = Path("models/xgb_v1.pkl")
SHAP_PATH = Path("models/shap_explainer_v1.pkl")


@router.get("/explain/{customer_id}", tags=["Explanation"])
def explain_customer(customer_id: str):
    try:
        model = joblib.load(MODEL_PATH)
        explainer = joblib.load(SHAP_PATH)
    except Exception:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df_raw = load_raw_data()
    matches = df_raw[df_raw["customerID"] == customer_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    df = clean_data(df_raw)
    df = engineer_features(df)
    X, y = get_features_and_target(df)

    idx = matches.index[0]
    prob = float(model.predict_proba(X.iloc[[idx]])[:, 1][0])
    risk = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW"

    shap_vals = explainer.shap_values(X.iloc[[idx]])[0]
    shap_series = pd.Series(shap_vals, index=X.columns)

    profile = df_raw.iloc[idx]

    return {
        "customer_id": customer_id,
        "churn_probability": round(prob, 4),
        "risk_level": risk,
        "profile": {
            "tenure": int(profile["tenure"]),
            "contract": profile["Contract"],
            "monthly_charges": float(profile["MonthlyCharges"]),
            "internet_service": profile["InternetService"],
            "payment_method": profile["PaymentMethod"],
        },
        "top_risk_factors": shap_series.nlargest(5).round(4).to_dict(),
        "top_protective_factors": shap_series.nsmallest(5).round(4).to_dict(),
        "estimated_revenue_at_risk_ils": round(prob * 8000, 2)
    }