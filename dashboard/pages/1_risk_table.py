import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from src.data.loader import load_raw_data
from src.data.preprocessor import clean_data, get_features_and_target
from src.features.engineering import engineer_features

st.set_page_config(page_title="Risk Table", page_icon="📋", layout="wide")
st.title("📋 Customer Churn Risk Table")

@st.cache_resource
def load_model():
    return joblib.load(Path("models/xgb_v1.pkl"))

@st.cache_data
def load_scored_data():
    df_raw = load_raw_data()
    df = clean_data(df_raw)
    df = engineer_features(df)
    X, y = get_features_and_target(df)
    model = load_model()
    probs = model.predict_proba(X)[:, 1]
    result = df_raw.iloc[:len(probs)][["customerID", "tenure", "Contract",
                                       "MonthlyCharges", "InternetService",
                                       "PaymentMethod"]].copy()
    result["churn_probability"] = probs.round(4)
    result["risk_level"] = pd.cut(
        result["churn_probability"],
        bins=[-0.01, 0.4, 0.7, 1.01],
        labels=["🟢 LOW", "🟡 MEDIUM", "🔴 HIGH"]
    )
    result["revenue_at_risk_ils"] = (result["churn_probability"] * 8000).round(0).astype(int)
    return result.sort_values("churn_probability", ascending=False).reset_index(drop=True)

df = load_scored_data()

# Sidebar filters
st.sidebar.header("Filters")
risk_filter = st.sidebar.multiselect(
    "Risk Level",
    options=["🔴 HIGH", "🟡 MEDIUM", "🟢 LOW"],
    default=["🔴 HIGH", "🟡 MEDIUM"]
)
contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df["Contract"].unique().tolist(),
    default=df["Contract"].unique().tolist()
)
min_prob = st.sidebar.slider("Min Churn Probability", 0.0, 1.0, 0.0, 0.05)

filtered = df[
    (df["risk_level"].isin(risk_filter)) &
    (df["Contract"].isin(contract_filter)) &
    (df["churn_probability"] >= min_prob)
]

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", len(filtered))
col2.metric("High Risk", len(filtered[filtered["risk_level"] == "🔴 HIGH"]))
col3.metric("Total Revenue at Risk", f"₪{filtered['revenue_at_risk_ils'].sum():,}")
col4.metric("Avg Churn Probability", f"{filtered['churn_probability'].mean():.1%}")

st.divider()
st.dataframe(
    filtered,
    use_container_width=True,
    column_config={
        "churn_probability": st.column_config.ProgressColumn(
            "Churn Probability", min_value=0, max_value=1, format="%.2f"
        ),
        "revenue_at_risk_ils": st.column_config.NumberColumn(
            "Revenue at Risk (₪)", format="₪%d"
        ),
    }
)

csv = filtered.to_csv(index=False)
st.download_button("📥 Export to CSV", csv, "churn_risk_report.csv", "text/csv")