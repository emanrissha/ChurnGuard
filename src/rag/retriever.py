import pandas as pd
import joblib
from pathlib import Path
from src.data.loader import load_raw_data
from src.data.preprocessor import clean_data, get_features_and_target
from src.features.engineering import engineer_features


def build_customer_context(customer_id: str) -> str | None:
    df_raw = load_raw_data()
    df = clean_data(df_raw)
    df = engineer_features(df)
    X, y = get_features_and_target(df)

    model = joblib.load(Path("models/xgb_v1.pkl"))
    explainer = joblib.load(Path("models/shap_explainer_v1.pkl"))

    matches = df_raw[df_raw["customerID"] == customer_id]
    if matches.empty:
        return None

    idx = matches.index[0]
    prob = model.predict_proba(X.iloc[[idx]])[:, 1][0]
    risk = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW"

    shap_vals = explainer.shap_values(X.iloc[[idx]])[0]
    shap_series = pd.Series(shap_vals, index=X.columns)
    top_risk = shap_series.nlargest(5)
    top_protect = shap_series.nsmallest(3)

    profile = df_raw.iloc[idx]

    context = f"""
CUSTOMER PROFILE
----------------
Customer ID: {customer_id}
Tenure: {profile['tenure']} months
Contract: {profile['Contract']}
Monthly Charges: ₪{profile['MonthlyCharges']}
Total Charges: ₪{profile['TotalCharges']}
Internet Service: {profile['InternetService']}
Payment Method: {profile['PaymentMethod']}
Senior Citizen: {"Yes" if profile['SeniorCitizen'] == 1 else "No"}
Has Partner: {profile['Partner']}
Has Dependents: {profile['Dependents']}

CHURN PREDICTION
----------------
Churn Probability: {prob:.1%}
Risk Level: {risk}
Predicted Action: {"WILL CHURN" if prob >= 0.5 else "WILL STAY"}

TOP CHURN RISK FACTORS (SHAP)
------------------------------
{chr(10).join([f"- {feat}: +{val:.3f} (increases churn risk)" for feat, val in top_risk.items()])}

TOP PROTECTIVE FACTORS (SHAP)
------------------------------
{chr(10).join([f"- {feat}: {val:.3f} (reduces churn risk)" for feat, val in top_protect.items()])}
"""
    return context