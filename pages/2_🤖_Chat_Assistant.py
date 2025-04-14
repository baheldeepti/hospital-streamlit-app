# 📘 IMPORTS
import streamlit as st
import pandas as pd
import os
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import altair as alt
from fpdf import FPDF
from statsmodels.tsa.seasonal import seasonal_decompose
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 📘 Setup API Key
import openai
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# 📊 Fullscreen Chat Page
st.set_page_config(layout="wide")
st.title("🧠 Hospital Chat Assistant")

# 🧾 Sidebar
with st.sidebar.expander("ℹ️ About This App", expanded=False):
    st.markdown("Built for exploring hospital data with chat + charts.")

with st.sidebar.expander("📁 Load or Upload Dataset", expanded=True):
    if st.button("Load Sample Hospital Data"):
        try:
            sample_url = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"
            df = pd.read_csv(sample_url)
            st.session_state["main_df"] = df
            st.success("✅ Sample data loaded.")
        except Exception as e:
            st.error(f"❌ Could not load sample data: {e}")

    uploaded_file = st.file_uploader("Upload your CSV", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["main_df"] = df
            st.success("✅ File uploaded successfully.")
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")

# 🧠 Session State Setup
for key in ["chat_history", "query_log", "fallback_log"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "query_log" else {}

if "main_df" not in st.session_state:
    st.warning("⚠️ Please upload or load a dataset.")
    st.stop()

df = st.session_state["main_df"]

# Data Cleanup
df["Billing Amount"] = pd.to_numeric(df["Billing Amount"].replace('[\$,]', '', regex=True), errors="coerce")
df["Length of Stay"] = pd.to_numeric(df.get("Length of Stay", pd.Series(dtype=float)), errors="coerce")

# 💬 Chat UI
st.markdown("### 💬 Chat with Assistant")
for i, (q, a) in enumerate(st.session_state.chat_history):
    message(q, is_user=True, key=f"user_{i}")
    message(a, key=f"bot_{i}")

# Suggestion Buttons
suggestions = [
    "Show billing trend by hospital",
    "Patient count by gender",
    "Top conditions by test results",
    "Total billing by insurance provider",
    "Average age of patients by condition"
]

cols = st.columns(len(suggestions))
for i, s in enumerate(suggestions):
    if cols[i].button(s):
        st.session_state.query_log[s] = st.session_state.query_log.get(s, 0) + 1
        st.session_state.chat_input = s

user_input = st.text_input("", key="chat_input", placeholder="Ask your question...")

# 🔍 Agent Handler
def safe_chat_response(query):
    try:
        agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0), df, verbose=False)
        response = agent.run(query)
        return response + "\n\n🔍 Powered by LangChain Agent"
    except Exception as e:
        return f"⚠️ Error: {e}"

if user_input:
    with st.spinner("🤖 Thinking..."):
        response = safe_chat_response(user_input)
        st.session_state.chat_history.append((user_input, response))

# 📊 Chart Generator
st.markdown("### 📊 Auto Chart Preview")
if "chat_input" in st.session_state:
    query = st.session_state.chat_input.lower()
    if "billing" in query and "hospital" in query:
        chart_data = df.groupby("Hospital")["Billing Amount"].sum().reset_index()
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("Hospital:N", sort="-y"),
            y="Billing Amount:Q"
        )
        st.altair_chart(chart, use_container_width=True)

# 📈 Leaderboard
if st.session_state.query_log:
    leaderboard = pd.DataFrame(st.session_state.query_log.items(), columns=["Query", "Clicks"])
    st.markdown("### 🏆 Leaderboard")
    st.dataframe(leaderboard)

# 📦 Download Options
chat_df = pd.DataFrame(st.session_state.chat_history, columns=["User", "Assistant"])
st.download_button("🗎 Download Chat Log", data=chat_df.to_csv(index=False), file_name="chat_log.csv")

# 🔗 Page Navigation
st.page_link("pages/1_📊_Dashboard.py", label="Dashboard", icon="📊")
st.page_link("pages/4_Dashboard_Feature_Overview.py", label="Dashboard Feature Overview")
st.page_link("pages/3__Chat_Assistant_Feature_Overview.py", label="Chat Assistant Feature Overview")
