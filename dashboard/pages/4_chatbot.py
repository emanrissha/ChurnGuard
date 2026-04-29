import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from src.data.loader import load_raw_data
from src.data.preprocessor import clean_data, get_features_and_target
from src.features.engineering import engineer_features

st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 ChurnGuard Chatbot")
st.caption("Ask about any customer in Hebrew or English")

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

def build_context(customer_id: str) -> str:
    matches = df_raw[df_raw["customerID"] == customer_id]
    if matches.empty:
        return None
    idx = matches.index[0]
    prob = probs[idx]
    profile = df_raw.iloc[idx]
    customer_X = X.iloc[[idx]]
    shap_vals = explainer.shap_values(customer_X)[0]
    shap_series = pd.Series(shap_vals, index=X.columns)
    top_risk = shap_series.nlargest(3)
    top_protect = shap_series.nsmallest(3)

    context = f"""
Customer ID: {customer_id}
Churn Probability: {prob:.1%}
Risk Level: {"HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW"}
Tenure: {profile['tenure']} months
Contract: {profile['Contract']}
Monthly Charges: ₪{profile['MonthlyCharges']}
Internet Service: {profile['InternetService']}
Payment Method: {profile['PaymentMethod']}

Top risk factors (SHAP):
{chr(10).join([f'- {f}: +{v:.3f}' for f, v in top_risk.items()])}

Top protective factors (SHAP):
{chr(10).join([f'- {f}: {v:.3f}' for f, v in top_protect.items()])}
"""
    return context


def generate_response(question: str, context: str) -> str:
    risk_level = "HIGH" if "HIGH" in context else "MEDIUM" if "MEDIUM" in context else "LOW"
    prob_line = [l for l in context.split("\n") if "Churn Probability" in l][0]
    prob = prob_line.split(": ")[1]

    response = f"""Based on the ChurnGuard analysis:

**{prob_line}** | Risk Level: {risk_level}

**Why is this customer at risk?**
The top churn signals are:
"""
    for line in context.split("\n"):
        if line.startswith("- ") and "+" in line:
            response += f"\n{line}"

    response += f"""

**What protects them?**
"""
    for line in context.split("\n"):
        if line.startswith("- ") and "+" not in line and "-" in line.split(": ")[-1]:
            response += f"\n{line}"

    response += f"""

**Recommended action:** {"Immediate outreach — offer contract upgrade or discount." if risk_level == "HIGH" else "Monitor closely and send check-in email." if risk_level == "MEDIUM" else "No action needed — healthy customer."}
"""
    return response


st.info("💡 This chatbot works offline. To enable full AI responses, add your OpenAI API key to `.env`")

customer_ids = df_raw["customerID"].tolist()
selected_id = st.selectbox("Select a customer to ask about", customer_ids)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about this customer... / שאל על הלקוח..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context = build_context(selected_id)
    if context:
        response = generate_response(prompt, context)
    else:
        response = f"Customer {selected_id} not found in the database."

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)