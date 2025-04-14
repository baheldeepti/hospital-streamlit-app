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

# 🔐 OpenAI Key Setup
import openai
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# 📊 Page Config
st.set_page_config(page_title="🤖 Chat Assistant", layout="wide")
st.title("🤖 Hospital Chat Assistant")

# 🧾 Dataset Required
if "main_df" not in st.session_state:
    st.warning("⚠️ Please upload or load a dataset from the sidebar before using the chat assistant.")
    st.stop()

df = st.session_state["main_df"]

# 🔧 Data Cleanup
df["Billing Amount"] = pd.to_numeric(df["Billing Amount"].replace('[\$,]', '', regex=True), errors="coerce")
df["Length of Stay"] = pd.to_numeric(df.get("Length of Stay", pd.Series(dtype=float)), errors="coerce")

# 📚 Sidebar: Data Glossary
with st.sidebar.expander("📚 Data Glossary", expanded=False):
    st.markdown("""
    **Column Descriptions:**
    - **Name**: Patient name
    - **Age**: Patient's age
    - **Gender**: Male/Female
    - **Medical Condition**: Primary diagnosis
    - **Date of Admission**: Admission date
    - **Doctor**: Attending doctor
    - **Hospital**: Hospital name
    - **Billing Amount**: Total charges
    - **Length of Stay**: Days admitted
    """)

# 💾 Session Initialization
for key in ["chat_history", "query_log", "fallback_log"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "query_log" else {}

# 💬 Chat Section
st.markdown("### 💬 Chat with Assistant")
for i, (q, a) in enumerate(st.session_state.chat_history):
    message(q, is_user=True, key=f"user_{i}")
    message(a, key=f"bot_{i}")

# 💡 Suggested Prompts
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
        st.session_state["chat_input"] = s
        st.session_state["query_log"][s] = st.session_state["query_log"].get(s, 0) + 1

# 🧠 User Input
user_input = st.text_input("Ask a question", key="chat_input", placeholder="E.g. Average stay by condition")

# 🛡️ Safe Agent Execution
def respond_to_query(query):
    try:
        agent = create_pandas_dataframe_agent(
            llm=ChatOpenAI(temperature=0),
            df=df,
            verbose=False
        )
        return agent.run(query)
    except Exception as e:
        st.session_state["fallback_log"].append(query)
        return (
            "⚠️ This query relies on code execution tools that are not supported on this platform.\n\n"
            "Please explore the dashboard or try uploading different data."
        )

# 📘 Keyword Tooltips
tooltips = {
    "billing": "Total amount charged to the patient",
    "stay": "Length of stay in days",
    "gender":
