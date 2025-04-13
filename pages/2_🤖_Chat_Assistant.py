# 📘 IMPORTS
import streamlit as st
import os
import shutil
import pandas as pd
import openai
import numpy as np
import altair as alt
from hashlib import md5

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from streamlit_chat import message

openai.api_key = st.secrets["OPENAI_API_KEY"]

# 🎨 Collapsible UI Sections
with st.sidebar.expander("📊 Summary Stats", expanded=False):
    if 'main_df' in st.session_state:
        df = st.session_state.main_df
        st.markdown("### 🔢 Key Dataset Stats")
        st.metric("Total Records", len(df))
        if "Billing Amount" in df.columns:
            st.metric("Total Billing", f"${df['Billing Amount'].sum():,.2f}")
        if "Length of Stay" in df.columns:
            st.metric("Avg Stay (days)", f"{df['Length of Stay'].mean():.1f}")
        if "Medical Condition" in df.columns:
            top_cond = df['Medical Condition'].value_counts().idxmax()
            st.metric("Top Condition", top_cond)
    else:
        st.info("No dataset loaded yet.")

with st.sidebar.expander("📂 Dataset Configuration", expanded=True):
    use_sample_data = st.toggle("Use Sample Data Instead of Upload", value=True)
    uploaded_file = st.file_uploader("📁 Or upload your hospital dataset", type=["csv"])

if use_sample_data:
    sample_url = "https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv"
    df = pd.read_csv(sample_url)
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    st.session_state.main_df = df
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
    st.session_state.main_df = df
else:
    df = None

# 🔍 Data Glossary
with st.sidebar.expander("📘 Data Glossary", expanded=False):
    st.markdown("### 🗂️ Column Descriptions")
    glossary = {
        "Name": "Full name of the patient.",
        "Age": "Age of the patient in years.",
        "Gender": "Biological sex of the patient.",
        "Blood Type": "Patient's blood group (A, B, AB, O).",
        "Medical Condition": "Primary medical condition or diagnosis.",
        "Date of Admission": "Date the patient was admitted to the hospital.",
        "Doctor": "Primary physician assigned.",
        "Hospital": "Name of the hospital facility.",
        "Insurance Provider": "Health insurance provider name.",
        "Billing Amount": "Total bill generated for the patient.",
        "Room Number": "Assigned hospital room.",
        "Admission Type": "Emergency, Routine, etc.",
        "Discharge Date": "Date of discharge from hospital.",
        "Medication": "Prescribed medication(s).",
        "Test Results": "Lab or diagnostic test results.",
        "Length of Stay": "Duration of hospitalization in days."
    }
    for col, desc in glossary.items():
        st.markdown(f"- **{col}**: {desc}")

# 🔍 Required Column Defaults
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        (
            "👋 Welcome! I'm your Hospital Data Assistant.",
            "Upload a dataset or use the sample data to ask questions like:\n"
            "- What is the average length of stay by condition?\n"
            "- Show billing trend for January\n"
            "- How many patients were admitted last week?"
        )
    ]

st.subheader("💬 Ask Questions About the Data")
st.caption("You can chat with the assistant like: 'Show top conditions by stay', or 'Average billing per hospital'.")
st.markdown("Use the chat box below to ask natural language questions about your dataset.")
# 📘 IMPORTS
import streamlit as st
import os
import shutil
import pandas as pd
import openai
import numpy as np
import altair as alt
import random
sample_qas = []
if df is not None:
    if 'Length of Stay' in df.columns and 'Medical Condition' in df.columns:
        sample_qas.append((
            "What is the average length of stay by condition?",
            "Here is a breakdown of the average stay for each condition."
        ))
    if 'Date of Admission' in df.columns:
        sample_qas.append((
            "How many patients were admitted in the last month?",
            "Let's look at the admissions over the past 30 days."
        ))
    if 'Billing Amount' in df.columns:
        sample_qas.append((
            "Show billing trend for January.",
            "Here's the billing trend for January."
        ))
if not sample_qas:
    sample_qas = [
        ("How do I get started?", "Please upload your hospital dataset or enable sample data to begin analysis.")
    ]



st.markdown("### 💬 Chat with the Assistant")
with st.container():
    for i, (q, a) in enumerate(st.session_state.chat_history):
        if q.startswith("👋"):
            message(q, is_user=False, key=f"sys_{i}", avatar_style="thumbs")
        else:
            message(q, is_user=True, key=f"user_{i}")
        message(a, key=f"bot_{i}", avatar_style="fun-emoji")

st.markdown("---")
st.markdown("#### ✏️ Type your question below")
user_input = st.text_input("", placeholder="Ask about billing, admissions, stay length, etc.", key="chat_input")
if not user_input.strip():
    st.warning("Please enter a question above to receive insights.")

# Generate response
if user_input:
    if df is None or 'rag_qa_chain' not in st.session_state:
        st.error("⚠️ Please upload a dataset or enable sample data before asking questions.")
    else:
        with st.spinner("🤖 Typing response... Please wait..."):
            try:
                if len(st.session_state.chat_history) > max_history:
                    st.warning(f"🧠 Chat history truncated to last {max_history} entries.")
                    st.session_state.chat_history = st.session_state.chat_history[-max_history:]

                import tiktoken
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                history_text = "\n".join([q + "\n" + a for q, a in st.session_state.chat_history])
                all_context = history_text + "\n" + user_input
                context_tokens_only = len(encoding.encode(history_text))
                user_tokens = len(encoding.encode(user_input))
                context_tokens = len(encoding.encode(all_context))
                if user_tokens > 3000:
                    st.error("❌ Your input is too long.")
                    st.stop()
                response = st.session_state.rag_qa_chain.run(user_input)
                st.session_state.chat_history.append((user_input, response))
            except Exception as e:
                st.error("⚠️ Something went wrong.")
                st.write(f"**Error:** {e}")
                st.stop()

        # 📊 Auto Charting Based on Keywords
        with st.expander("📊 Auto Insights", expanded=True):
            if "billing trend" in user_input.lower():
                if 'Medical Condition' in df.columns and 'Billing Amount' in df.columns:
                    trend_df = df.groupby("diagnosis")["charges"].sum().reset_index().sort_values(by="charges", ascending=False)
                    st.altair_chart(
                        alt.Chart(trend_df).mark_line().encode(
                            x="Date of Admission:T", y="Billing Amount:Q"
                        ).properties(title="Monthly Billing Trend", width=600), use_container_width=True
                    )

            elif "average length of stay" in user_input.lower():
                if 'Medical Condition' in df.columns and 'Length of Stay' in df.columns:
                    avg_stay = df.groupby("diagnosis")["length_of_stay"].mean().reset_index().sort_values(by="length_of_stay", ascending=False)
                    st.altair_chart(
                        alt.Chart(avg_stay).mark_bar().encode(
                            x="Length of Stay:Q", y=alt.Y("Medical Condition:N", sort='-x')
                        ).properties(title="Average Length of Stay by Condition", width=600), use_container_width=True
                    )

            elif "admitted" in user_input.lower():
                if 'Date of Admission' in df.columns:
                    admission_df = df.groupby(df['Date of Admission'].dt.to_period("W")).size().reset_index(name="Admissions")
                    admission_df['Date of Admission'] = admission_df['Date of Admission'].dt.start_time
                    st.altair_chart(
                        alt.Chart(admission_df).mark_area().encode(
                            x="Date of Admission:T", y="Admissions:Q"
                        ).properties(title="Weekly Patient Admissions", width=600), use_container_width=True
                    )

# Render chat
for i, (q, a) in enumerate(st.session_state.chat_history):
    if q.startswith("👋"):
        message(q, is_user=False, key=f"sys_{i}", avatar_style="thumbs")
    else:
        message(q, is_user=True, key=f"user_{i}")
    message(a, key=f"bot_{i}", avatar_style="fun-emoji")

# Downloads
if st.session_state.chat_history:
    chat_df = pd.DataFrame(st.session_state.chat_history, columns=["User", "Assistant"])
    st.download_button("📅 Download Chat as CSV", data=chat_df.to_csv(index=False), file_name="chat_history.csv", mime="text/csv")
