import streamlit as st
import joblib
from pathlib import Path
from src.data.loader import load_raw_data
from src.rag.chatbot import ask_churnguard

st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 ChurnGuard AI Chatbot")
st.caption("Powered by Claude AI — ask about any customer in Hebrew or English")

@st.cache_data
def get_customer_ids():
    df = load_raw_data()
    return df["customerID"].tolist()

customer_ids = get_customer_ids()
selected_id = st.selectbox("Select a customer", customer_ids)

st.info(f"💡 Selected: **{selected_id}** — ask anything about this customer")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask in Hebrew or English... / שאל בעברית או אנגלית..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = ask_churnguard(selected_id, prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.rerun()