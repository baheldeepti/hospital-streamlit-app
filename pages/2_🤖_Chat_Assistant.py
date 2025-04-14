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
from streamlit_chat import message

from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 🔐 OpenAI Key Setup
import openai
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# 📊 Page Config
st.set_page_config(page_title="🤖 Chat Assistant", layout="wide")
st.title("🤖 Hospital Chat Assistant")

# ℹ️ Sidebar Info
with st.sidebar.expander("ℹ️ About This App", expanded=False):
    st.markdown("""
    This app helps you explore hospital datasets through a conversational AI assistant.
    Ask about billing, conditions, patient trends, and more!
    """)

# 📁 Sidebar: Dataset Loader
with st.sidebar.expander("📥 Load or Upload Dataset", expanded=True):
    if st.button("Load Sample Hospital Data"):
        try:
            sample_url = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"
            df = pd.read_csv(sample_url)
            st.session_state["main_df"] = df
            st.success("✅ Sample dataset loaded.")
        except Exception as e:
            st.error(f"❌ Failed to load sample dataset: {e}")

    uploaded_file = st.file_uploader("Upload your CSV", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["main_df"] = df
            st.success("✅ File uploaded successfully.")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# 🧠 Session State Initialization
for key in ["chat_history", "query_log", "fallback_log"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "query_log" else {}

# 🧾 Dataset Validation
if "main_df" not in st.session_state:
    st.warning("⚠️ Please load or upload a dataset to begin.")
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

# 💬 Chat Section
st.markdown("### 💬 Chat with Assistant")
for i, (q, a) in enumerate(st.session_state.chat_history):
    message(q, is_user=True, key=f"user_{i}")
    message(a, key=f"bot_{i}")

# 💡 Suggested Questions
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

user_input = st.text_input("Ask a question", key="chat_input", placeholder="E.g. Average stay by condition")

# 🤖 Query Processor
def respond_to_query(query):
    try:
        agent = create_pandas_dataframe_agent(
            llm=ChatOpenAI(temperature=0),
            df=df,
            verbose=False
            # Not passing allow_dangerous_code=True ensures safety
        )
        return agent.run(query)
    except Exception as e:
        return f"⚠️ Error: {str(e)}\n\nThis platform doesn't support unsafe code execution."

if user_input:
    with st.spinner("🤖 Thinking..."):
        response = respond_to_query(user_input)
        st.session_state.chat_history.append((user_input, response))

# 📊 Auto Chart Preview
st.markdown("### 📊 Auto Chart Preview")
if "chat_input" in st.session_state:
    query = st.session_state.chat_input.lower()

    if "billing" in query and "hospital" in query:
        chart_data = df.groupby("Hospital")["Billing Amount"].sum().reset_index()
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("Hospital:N", sort="-y"),
            y="Billing Amount:Q"
        ).properties(title="Billing by Hospital")
        st.altair_chart(chart, use_container_width=True)

# 🏆 Leaderboard
if st.session_state.query_log:
    leaderboard = pd.DataFrame(st.session_state.query_log.items(), columns=["Query", "Clicks"])
    st.markdown("### 🏆 Most Popular Queries")
    st.dataframe(leaderboard)

# ⬇️ Chat History Download
if st.session_state.chat_history:
    chat_df = pd.DataFrame(st.session_state.chat_history, columns=["User", "Assistant"])
    st.download_button("📥 Download Chat Log", data=chat_df.to_csv(index=False), file_name="chat_history.csv")

# 🔗 Page Navigation
st.page_link("pages/1_📊_Dashboard.py", label="📊 Dashboard")
st.page_link("pages/4_Dashboard_Feature_Overview.py", label="📘 Dashboard Feature Overview")
st.page_link("pages/3__Chat_Assistant_Feature_Overview.py", label="📄 Chat Assistant Feature Overview")
