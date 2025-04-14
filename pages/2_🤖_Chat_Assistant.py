# 📘 IMPORTS
import shutil
# 📘 IMPORTS
import streamlit as st
import os
import pandas as pd
import openai
import numpy as np
import altair as alt
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from fpdf import FPDF
from statsmodels.tsa.seasonal import seasonal_decompose
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents import create_pandas_dataframe_agent

openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# 📊 Fullscreen Chat Page
st.set_page_config(layout="wide")
st.title("🧠 Hospital Chat Assistant")

# ℹ️ App Information
with st.sidebar.expander("ℹ️ About This App", expanded=False):
    st.markdown("""
    **🧠 Hospital Chat Assistant** is an interactive Streamlit application designed to help users explore and analyze hospital patient data with ease.

    #### 🔍 Key Features:
    - **Conversational AI Chatbot**: Ask natural language questions about the dataset (e.g., billing, conditions, admissions).
    - **Data Loading Options**: Upload your own CSV or load a sample dataset.
    - **Auto Chart Previews**: Get visual answers with auto-generated bar charts and downloadable PNG/PDF files.
    - **Data Glossary**: Sidebar reference for column definitions to support data understanding.
    - **Dynamic Suggestions**: Smart question prompts based on available columns in your dataset.
    - **RAG and Fallback Agents**: Uses LangChain RAG (Retrieval-Augmented Generation) and Pandas Data Agent for insights.
    - **Tooltip-Enhanced Responses**: Explanations for technical terms directly in chat.
    - **Query Leaderboard**: See which prompts are used most often.
    - **Full Logs & Downloads**: Export your chat history, query logs, and fallback inputs.

    #### 🛠️ Powered By:
    - **Streamlit** for UI
    - **LangChain + FAISS** for conversational AI
    - **Altair & Matplotlib** for charts
    - **Statsmodels** for seasonal analysis (extendable)
    - **OpenAI API** for LLM responses

    👩‍⚕️ Built for data analysts, hospital admins, and curious users who want to make sense of medical data — easily and intuitively.
    """)

# 🧾 Sidebar: Dataset Loader
with st.sidebar.expander("📁 Load or Upload Dataset", expanded=True):
    if st.button("Load Sample Hospital Data"):
        sample_url = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/data/modified_healthcare_sample.csv"
        try:
            st.session_state.main_df = pd.read_csv(sample_url)
            st.success("✅ Sample data loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load sample dataset: {e}")

    uploaded_file = st.file_uploader("Or drop your CSV file here", type="csv")
    if uploaded_file is not None:
        try:
            st.session_state.main_df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading uploaded file: {e}")

# 📚 Sidebar: Data Glossary
with st.sidebar.expander("📚 Data Glossary", expanded=False):
    st.markdown("""
    **Column Descriptions:**
    - **Name**: Name of the patient.
    - **Age**: Patient's age at the time of admission.
    - **Gender**: "Male" or "Female".
    - **Blood Type**: (e.g., "A+", "O-").
    - **Medical Condition**: Primary diagnosis like "Diabetes".
    - **Date of Admission**: Admission date.
    - **Doctor**: Attending doctor name.
    - **Hospital**: Name of hospital.
    - **Insurance Provider**: (e.g., "Aetna", "Medicare").
    - **Billing Amount**: Amount charged.
    - **Room Number**: Room number.
    - **Admission Type**: ("Emergency", "Elective").
    - **Discharge Date**: Date of discharge.
    - **Medication**: Administered drugs.
    - **Test Results**: ("Normal", "Abnormal").
    """)

# 🧠 Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_log" not in st.session_state:
    st.session_state.query_log = {}
if "fallback_log" not in st.session_state:
    st.session_state.fallback_log = []

if "main_df" not in st.session_state:
    st.warning("⚠️ Please upload or select a dataset from the sidebar to begin chatting.")
    st.stop()

# 🔧 Data Cleanup
st.session_state.main_df["Billing Amount"] = pd.to_numeric(
    st.session_state.main_df["Billing Amount"].replace('[\$,]', '', regex=True), errors="coerce")
st.session_state.main_df["Length of Stay"] = pd.to_numeric(
    st.session_state.main_df.get("Length of Stay", pd.Series(dtype=float)), errors="coerce")

# 🏁 Getting Started
with st.expander("📘 Getting Started", expanded=False):
    st.markdown("""
    Welcome to the Hospital Chat Assistant! Upload a dataset or use the sample. Ask about billing, admissions, or health trends.
    """)

st.sidebar.info("✅ Dataset loaded: " + str(st.session_state.main_df.shape))

# Layout
col1, col2 = st.columns([2, 1])

# ─── Chat Column ──────────────────────────────────────────────
with col1:
    st.markdown("### 💬 Chat with the Assistant")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        message(q, is_user=True, key=f"user_{i}")
        message(a, key=f"bot_{i}")

    # Dynamic Suggestions
    df_columns = st.session_state.main_df.columns
    suggestions = []
    if "Hospital" in df_columns and "Billing Amount" in df_columns:
        suggestions.append("Show billing trend by hospital")
    if "Gender" in df_columns:
        suggestions.append("Patient count by gender")
    if "Medical Condition" in df_columns and "Test Results" in df_columns:
        suggestions.append("Top conditions by test results")
    if "Insurance Provider" in df_columns and "Billing Amount" in df_columns:
        suggestions.append("Total billing by insurance provider")
    if "Age" in df_columns and "Medical Condition" in df_columns:
        suggestions.append("Average age of patients by condition")

    cols = st.columns(len(suggestions))
    for i, s in enumerate(suggestions):
        if cols[i].button(s):
            st.session_state.query_log[s] = st.session_state.query_log.get(s, 0) + 1
            st.session_state.chat_input = s

    user_input = st.text_input("", key="chat_input", placeholder="Ask your question here...")

    tooltips = {
        "billing": "Total amount charged to the patient",
        "stay": "Length of stay in days",
        "gender": "Gender breakdown of patients",
        "condition": "Primary medical condition during admission"
    }

    def add_tooltip(response, terms):
        for word, tip in terms.items():
            if word in response.lower():
                response += f"\n\n🛈 *{word.capitalize()}* refers to: {tip}"
        return response

    if user_input:
        with st.spinner("🤖 Assistant is typing..."):
            try:
                rag_qa = st.session_state.get("rag_qa_chain")
                if rag_qa:
                    response = rag_qa.run(user_input)
                else:
                    agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0), st.session_state.main_df, verbose=False)
                    response = agent.run(user_input) + "\n\n🔍 Powered by Data Agent"
                response = add_tooltip(response, tooltips)
                st.session_state.chat_history.append((user_input, response))
            except Exception as e:
                st.session_state.chat_history.append((user_input, f"⚠️ Error: {e}"))
                st.session_state.fallback_log.append(user_input)

# ─── Chart Column ─────────────────────────────────────────────
with col2:
    st.markdown("### 📊 Auto Chart Preview")
    df = st.session_state.main_df.copy()

    def export_chart(chart, filename):
        buf = BytesIO()
        chart.save(buf, format="png")
        b64 = base64.b64encode(buf.getvalue()).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">📩 Download PNG</a>'
        st.markdown(href, unsafe_allow_html=True)

    def export_chart_pdf(dataframe, filename):
        buffer = BytesIO()
        fig, ax = plt.subplots()
        dataframe.plot(kind='barh', x=dataframe.columns[0], y=dataframe.columns[1], ax=ax)
        fig.tight_layout()
        plt.savefig(buffer, format="pdf")
        buffer.seek(0)
        b64_pdf = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}.pdf">📄 Download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

    if "chat_input" in st.session_state:
        query = st.session_state.chat_input.lower()

        if "billing" in query and "hospital" in query:
            chart_data = df.groupby("Hospital")["Billing Amount"].sum().reset_index()
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X("Hospital:N", sort="-y"),
                y="Billing Amount:Q"
            ).properties(title="Billing by Hospital")
            st.altair_chart(chart, use_container_width=True)
            export_chart(chart, "billing_by_hospital")
            export_chart_pdf(chart_data, "billing_by_hospital")

        elif "gender" in query:
            chart_data = df["Gender"].value_counts().reset_index()
            chart_data.columns = ["Gender", "Count"]
            chart = alt.Chart(chart_data).mark_bar().encode(
                x="Gender:N",
                y="Count:Q"
            ).properties(title="Patient Count by Gender")
            st.altair_chart(chart, use_container_width=True)
            export_chart(chart, "patient_count_by_gender")
            export_chart_pdf(chart_data, "patient_count_by_gender")

# 📈 Leaderboard and Logs
if st.session_state.query_log:
    st.markdown("### 🏆 Most Clicked Suggestions")
    leaderboard_df = pd.DataFrame(sorted(st.session_state.query_log.items(), key=lambda x: x[1], reverse=True), columns=["Query", "Clicks"])
    st.dataframe(leaderboard_df, use_container_width=True)
    st.download_button("⬇️ Download Query Log (CSV)", data=leaderboard_df.to_csv(index=False), file_name="query_log.csv")

if st.session_state.fallback_log:
    fallback_df = pd.DataFrame(st.session_state.fallback_log, columns=["Query"])
    st.markdown("### 🧾 Fallback Queries")
    st.dataframe(fallback_df, use_container_width=True)
    st.download_button("⬇️ Download Fallback Queries", data=fallback_df.to_csv(index=False), file_name="fallback_queries.csv")

if st.session_state.chat_history:
    chat_df = pd.DataFrame(st.session_state.chat_history, columns=["User", "Assistant"])
    st.download_button("🗎 Download Chat History (CSV)", data=chat_df.to_csv(index=False), file_name="chat_history.csv")
