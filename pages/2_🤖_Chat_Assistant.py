# 📘 IMPORTS
import streamlit as st
import os
import shutil
import pandas as pd
import openai
import numpy as np
import altair as alt
from hashlib import md5
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
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

openai.api_key = st.secrets["OPENAI_API_KEY"]

# 📊 Fullscreen Chat Page
st.set_page_config(layout="wide")
st.title("🧠 Hospital Chat Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "main_df" not in st.session_state:
    st.warning("⚠️ Please upload or select a dataset from the sidebar to begin chatting.")
    st.stop()

# Convert data types
st.session_state.main_df["Billing Amount"] = pd.to_numeric(
    st.session_state.main_df["Billing Amount"].replace('[\$,]', '', regex=True), errors="coerce"
)
st.session_state.main_df["Length of Stay"] = pd.to_numeric(
    st.session_state.main_df["Length of Stay"], errors="coerce"
)

# Layout: Chat + Chart Tabs side-by-side
col1, col2 = st.columns([2, 1])

# ─── Chat Column ──────────────────────────────────────────────
with col1:
    st.markdown("### 💬 Chat with the Assistant")
    st.caption("Ask questions like: 'Show billing trend by hospital', 'Top conditions by stay length'.")

    for i, (q, a) in enumerate(st.session_state.chat_history):
        message(q, is_user=True, key=f"user_{i}")
        message(a, key=f"bot_{i}")

    example_queries = [
        "Show billing trend by hospital",
        "Top diagnoses by average stay",
        "Patient count by gender",
        "Total billing by insurance"
    ]
    st.markdown("**Quick Suggestions:**")
    cols = st.columns(len(example_queries))
    for i, query in enumerate(example_queries):
        if cols[i].button(query):
            st.session_state.chat_input = query

    user_input = st.text_input("", key="chat_input", placeholder="Ask about billing, admissions, stay length, etc.")

    if user_input:
        df = st.session_state.main_df
        with st.spinner("🤖 Assistant is typing..."):
            try:
                if len(st.session_state.chat_history) > 20:
                    st.session_state.chat_history = st.session_state.chat_history[-20:]

                rag_qa = st.session_state.get("rag_qa_chain")
                if rag_qa:
                    response = rag_qa.run(user_input)
                else:
                    response = "Please ensure the embedding pipeline is active."
                st.session_state.chat_history.append((user_input, response))
            except Exception as e:
                st.session_state.chat_history.append((user_input, f"⚠️ Error: {e}"))

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

    if "chat_input" in st.session_state:
        user_input = st.session_state.chat_input.lower()

        if "billing" in user_input and "hospital" in user_input:
            chart_data = df.groupby("Hospital")["Billing Amount"].sum().reset_index()
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X("Hospital:N", sort='-y'),
                y="Billing Amount:Q"
            ).properties(title="Billing by Hospital")
            st.altair_chart(chart, use_container_width=True)
            export_chart(chart, "billing_by_hospital")

        elif "stay" in user_input and "condition" in user_input:
            if "Medical Condition" in df.columns and "Length of Stay" in df.columns:
                chart_data = df.groupby("Medical Condition")["Length of Stay"].mean().reset_index()
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x="Length of Stay:Q",
                    y=alt.Y("Medical Condition:N", sort='-x')
                ).properties(title="Avg Stay by Condition")
                st.altair_chart(chart, use_container_width=True)
                export_chart(chart, "avg_stay_by_condition")

        elif "gender" in user_input:
            chart_data = df["Gender"].value_counts().reset_index()
            chart_data.columns = ["Gender", "Count"]
            chart = alt.Chart(chart_data).mark_bar().encode(
                x="Gender:N",
                y="Count:Q"
            ).properties(title="Patient Count by Gender")
            st.altair_chart(chart, use_container_width=True)
            export_chart(chart, "patient_count_by_gender")

# Chat CSV Export
if st.session_state.chat_history:
    chat_df = pd.DataFrame(st.session_state.chat_history, columns=["User", "Assistant"])
    csv = chat_df.to_csv(index=False)
    st.download_button("📅 Download Chat (CSV)", data=csv, file_name="chat_history.csv", mime="text/csv")
