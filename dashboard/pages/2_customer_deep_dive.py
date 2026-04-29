import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from src.data.loader import load_raw_data
from src.data.preprocessor import clean_data, get_features_and_target
from src.features.engineering import engineer_features

st.set_page_config(page_title="Customer Deep Dive", page_icon="🔍", layout="wide")
st.title("🔍 Customer Deep Dive")

@st.cache_resource
def load_artifacts():
    model = joblib.load(Path("models/xgb_v1.pkl"))
    explainer = joblib.load(Path("models/shap_explainer_v1.pkl"))
    return model, explainer

@st.cache_data
def load_data():
    df_raw = load_raw_data()
    df = clean_data(df_raw)
    df = engineer_features(df)
    X, y = get_features_and_target(df)
    return df_raw, X, y

model, explainer = load_artifacts()
df_raw, X, y = load_data()

probs = model.predict_proba(X)[:, 1]
customer_ids = df_raw["customerID"].values

selected_id = st.selectbox(
    "Select Customer ID",
    options=customer_ids,
    index=0
)

idx = list(customer_ids).index(selected_id)
prob = probs[idx]
actual = y.iloc[idx]

col1, col2, col3 = st.columns(3)
col1.metric("Churn Probability", f"{prob:.1%}")
col2.metric("Risk Level", "🔴 HIGH" if prob >= 0.7 else "🟡 MEDIUM" if prob >= 0.4 else "🟢 LOW")
col3.metric("Actual Outcome", "Churned ✗" if actual == 1 else "Retained ✓")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Customer Profile")
    profile = df_raw.iloc[idx][["tenure", "Contract", "MonthlyCharges",
                                 "TotalCharges", "InternetService",
                                 "PaymentMethod", "SeniorCitizen"]]
    for k, v in profile.items():
        st.write(f"**{k}:** {v}")

with col_right:
    st.subheader("SHAP Risk Factors")
    customer_df = X.iloc[[idx]]
    shap_vals = explainer.shap_values(customer_df)[0]
    shap_series = pd.Series(shap_vals, index=X.columns).sort_values(key=abs, ascending=False).head(10)

    colors = ["🔴" if v > 0 else "🟢" for v in shap_series.values]
    for color, (feat, val) in zip(colors, shap_series.items()):
        st.write(f"{color} **{feat}**: {val:+.4f}")

st.divider()
st.subheader("SHAP Waterfall Chart")
fig, ax = plt.subplots(figsize=(10, 5))
shap_explanation = explainer(X.iloc[[idx]])
shap.plots.waterfall(shap_explanation[0], max_display=12, show=False)
st.pyplot(plt.gcf())
plt.close()