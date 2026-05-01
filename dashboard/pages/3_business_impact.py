import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path
from src.data.loader import load_raw_data
from src.data.preprocessor import clean_data, get_features_and_target
from src.features.engineering import engineer_features

st.set_page_config(page_title="Business Impact", page_icon="💰", layout="wide")
st.title("💰 Business Impact Calculator")

@st.cache_resource
def load_model():
    return joblib.load(Path("models/xgb_v1.pkl"))

@st.cache_data
def load_data():
    df = clean_data(load_raw_data())
    df = engineer_features(df)
    X, y = get_features_and_target(df)
    return X, y

X, y = load_data()
model = load_model()
probs = model.predict_proba(X)[:, 1]

st.sidebar.header("Business Parameters")
arr_per_customer = st.sidebar.number_input("ARR per customer (₪)", value=8000, step=500)
retention_cost = st.sidebar.number_input("Retention outreach cost (₪)", value=500, step=100)
retention_rate = st.sidebar.slider("Success rate of retention effort", 0.1, 0.9, 0.4)
threshold = st.sidebar.slider("Churn probability threshold", 0.3, 0.8, 0.5, 0.05)

high_risk = (probs >= threshold).sum()
predicted_churners = probs[probs >= threshold]
revenue_at_risk = int(high_risk * arr_per_customer)
cost_to_intervene = int(high_risk * retention_cost)
saved_customers = int(high_risk * retention_rate)
saved_revenue = int(saved_customers * arr_per_customer)
net_benefit = saved_revenue - cost_to_intervene

col1, col2, col3, col4 = st.columns(4)
col1.metric("High Risk Customers", f"{high_risk:,}")
col2.metric("Revenue at Risk", f"₪{revenue_at_risk:,}")
col3.metric("Cost to Intervene", f"₪{cost_to_intervene:,}")
col4.metric("Net Benefit", f"₪{net_benefit:,}", delta=f"₪{saved_revenue:,} saved")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Risk Distribution")
    risk_df = pd.DataFrame({
        "Risk Level": ["🟢 LOW", "🟡 MEDIUM", "🔴 HIGH"],
        "Customers": [
            (probs < 0.4).sum(),
            ((probs >= 0.4) & (probs < 0.7)).sum(),
            (probs >= 0.7).sum()
        ]
    })
    fig = px.pie(risk_df, values="Customers", names="Risk Level",
                 color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"])
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Revenue at Risk by Contract Type")
    df_raw = load_raw_data().iloc[:len(probs)]
    df_raw = df_raw.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    df_raw["churn_prob"] = probs
    df_raw["revenue_at_risk"] = (probs * arr_per_customer).round(0)
    contract_risk = df_raw.groupby("Contract")["revenue_at_risk"].sum().reset_index()
    fig2 = px.bar(contract_risk, x="Contract", y="revenue_at_risk",
                  color="Contract", labels={"revenue_at_risk": "Revenue at Risk (₪)"},
                  color_discrete_sequence=["#e74c3c", "#f39c12", "#2ecc71"])
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("ROI Summary")
st.markdown(f"""
| Metric | Value |
|--------|-------|
| Customers flagged for intervention | {high_risk:,} |
| Total cost of outreach | ₪{cost_to_intervene:,} |
| Expected customers saved | {saved_customers:,} |
| Revenue saved | ₪{saved_revenue:,} |
| **Net ROI** | **₪{net_benefit:,}** |
| **ROI multiple** | **{saved_revenue/max(cost_to_intervene,1):.1f}x** |
""")