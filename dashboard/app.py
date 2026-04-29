import streamlit as st

st.set_page_config(
    page_title="ChurnGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ ChurnGuard")
st.subheader("B2B SaaS Churn Prediction Engine")

st.markdown("""
Welcome to **ChurnGuard** — an end-to-end ML system that identifies 
customers at risk of churning **before they cancel**.

---

### How to use this dashboard

| Page | What it shows |
|---|---|
| 📋 Risk Table | All customers ranked by churn probability |
| 🔍 Customer Deep Dive | Per-customer SHAP explanation |
| 💰 Business Impact | Revenue at risk + cost calculator |
| 🤖 Chatbot | Ask questions in Hebrew or English |

---
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model", "XGBoost (tuned)")
with col2:
    st.metric("AUC Score", "0.837")
with col3:
    st.metric("Potential Saved Revenue", "₪2.37M")