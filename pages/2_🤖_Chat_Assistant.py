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

# 🔍 Data Preview
with st.sidebar.expander("📘 Data Glossary", expanded=False):
    if df is not None:
        glossary = {
            "Billing Amount": "Total charge billed to the patient or insurance.",
            "Length of Stay": "Number of days a patient stayed in the hospital.",
            "Medical Condition": "Primary diagnosis or condition for admission.",
            "Date of Admission": "Date when the patient was admitted.",
            "Discharge Date": "Date when the patient was discharged."
        }
        for col in df.columns:
            if col in glossary:
                st.markdown(f"- **{col}**: {glossary[col]}")
            else:
                st.markdown(f"- **{col}**: _(No description available)_")
with st.sidebar.expander("🔍 Data Preview & Stats"):
    if 'main_df' in st.session_state:
        required_cols = st.multiselect("✅ Required Columns", ["Billing Amount", "Length of Stay", "Medical Condition"], default=["Billing Amount", "Length of Stay"])
        df = st.session_state.main_df
        if set(required_cols).issubset(df.columns):
            st.success("✅ Dataset loaded successfully!")
            st.dataframe(df.head())
            col_stats = df[required_cols].describe(include='all').T
            col_stats['missing'] = df[required_cols].isnull().sum()
            col_stats['missing_pct'] = (col_stats['missing'] / len(df) * 100).round(2)
            high_missing_cols = col_stats[col_stats['missing_pct'] > 25].index.tolist()
            if high_missing_cols:
                st.warning(f"⚠️ High missing data in: {', '.join(high_missing_cols)}")
            st.dataframe(col_stats[['count', 'mean', 'min', 'max', 'missing']].fillna("-").astype(str))
        else:
            st.error("❌ Required columns missing.")
            st.stop()

# ⚙️ Embedding Settings
with st.sidebar.expander("⚙️ Embedding Settings"):
    clear_cache = st.button("🗑️ Clear Embedding Cache")
    if clear_cache and os.path.exists(".embedding_cache"):
        shutil.rmtree(".embedding_cache")
        st.success("Embedding cache cleared.")
    max_chunks = st.slider("Max Chunks for Embedding", min_value=50, max_value=500, value=150, step=50)
    estimated_tokens = max_chunks * 500
    st.markdown(f"🧠 Estimated tokens for embedding: **{estimated_tokens}**")
    if estimated_tokens > 900000:
        st.warning("⚠️ Estimated tokens exceed 900,000.")

# 🧠 Chat Settings
with st.sidebar.expander("🧠 Chat Settings"):
    max_history = st.slider("Max Chat History Length", min_value=5, max_value=20, value=10, step=1)
    if st.button("🔁 Reset App State"):
    st.session_state.clear()
    st.success("✅ App state cleared. Restarting...")
    st.rerun()

# Document Embedding
if df is not None:
    csv_path = "filtered_data.txt"
    df.to_csv(csv_path, index=False)
    loader = TextLoader(csv_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc_chunks = splitter.split_documents(docs)

    @st.cache_resource(show_spinner="🔄 Embedding in progress...")
    def safe_embed(_chunks):
        text = " ".join([str(doc.page_content) for doc in _chunks])
        cache_key = md5(text.encode()).hexdigest()
        cache_path = f".embedding_cache/{cache_key}.faiss"
        if os.path.exists(cache_path):
            return FAISS.load_local(cache_path, OpenAIEmbeddings())
        else:
            vs = FAISS.from_documents(_chunks, OpenAIEmbeddings())
            os.makedirs(".embedding_cache", exist_ok=True)
            vs.save_local(cache_path)
            return vs

    try:
        vectorstore_doc = safe_embed(doc_chunks[:max_chunks])
    except Exception as e:
        st.error("⚠️ Token limit exceeded or embedding failed.")
        st.stop()

    rag_qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        retriever=vectorstore_doc.as_retriever(),
        return_source_documents=False
    )
    st.session_state.rag_qa_chain = rag_qa
else:
    st.warning("⚠️ No dataset selected. Please upload a file or toggle sample data.")

@st.cache_resource(show_spinner="🔄 Embedding in progress...")
def safe_embed(_chunks):
    text = " ".join([str(doc.page_content) for doc in _chunks])
    cache_key = md5(text.encode()).hexdigest()
    cache_path = f".embedding_cache/{cache_key}.faiss"
    if os.path.exists(cache_path):
        return FAISS.load_local(cache_path, OpenAIEmbeddings())
    else:
        vs = FAISS.from_documents(_chunks, OpenAIEmbeddings())
        os.makedirs(".embedding_cache", exist_ok=True)
        vs.save_local(cache_path)
        return vs

try:
    vectorstore_doc = safe_embed(doc_chunks[:max_chunks])
except Exception as e:
    st.error("⚠️ Token limit exceeded or embedding failed.")
    st.stop()

rag_qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    retriever=vectorstore_doc.as_retriever(),
    return_source_documents=False
)
st.session_state.rag_qa_chain = rag_qa

# Chat Interface
st.subheader("💬 Ask Questions About the Data")
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [random.choice(sample_qas)]
example_queries = [
    "Show billing trend for January",
    "What is the average length of stay by condition?",
    "How many patients were admitted last week?"
]
selected_example = st.selectbox("💡 Click an example to auto-fill the question box", [q for q, _ in sample_qas], index=0, key="example_prompt")
user_input = st.text_input("💬 Ask your question:", value=selected_example)
if not user_input.strip():
    st.warning("Please enter a question.")
    st.stop()

# Generate response
if user_input:
    if df is None or 'rag_qa_chain' not in st.session_state:
        st.error("⚠️ Please upload a dataset or enable sample data before asking questions.")
    else:
        with st.spinner("Thinking..."):
            try:
                if len(st.session_state.chat_history) > max_history:
                    st.warning(f"🧠 Chat history truncated to last {max_history} entries.")
                    st.session_state.chat_history = st.session_state.chat_history[-max_history:]

                import tiktoken
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                history_text = "".join([q + "" + a for q, a in st.session_state.chat_history])
                all_context = history_text + "" + user_input
                context_tokens_only = len(encoding.encode(history_text))
                user_tokens = len(encoding.encode(user_input))
                context_tokens = len(encoding.encode(all_context))
                if user_tokens > 3000:
                    st.error("❌ Your input is too long.")
                    st.stop()
                response = st.session_state.rag_qa_chain.run(user_input)
            except Exception as e:
                st.error("⚠️ Something went wrong.")
                st.write(f"**Error:** {e}")
                st.stop()

            st.session_state.chat_history.append((user_input, response))
            response = st.session_state.rag_qa_chain.run(user_input)
        except Exception as e:
            st.error("⚠️ Something went wrong.")
            st.write(f"**Error:** {e}")
            st.stop()

        st.session_state.chat_history.append((user_input, response))

        # 📊 Auto Charting Based on Keywords
        with st.expander("📊 Auto Insights", expanded=True):
            if "billing trend" in user_input.lower():
                if 'Date of Admission' in df.columns and 'Billing Amount' in df.columns:
                    trend_df = df.groupby(df['Date of Admission'].dt.to_period("M"))['Billing Amount'].sum().reset_index()
                    trend_df['Date of Admission'] = trend_df['Date of Admission'].dt.to_timestamp()
                    st.altair_chart(
                        alt.Chart(trend_df).mark_line().encode(
                            x="Date of Admission:T", y="Billing Amount:Q"
                        ).properties(title="Monthly Billing Trend", width=600), use_container_width=True
                    )

            elif "average length of stay" in user_input.lower():
                if 'Medical Condition' in df.columns and 'Length of Stay' in df.columns:
                    avg_stay = df.groupby("Medical Condition")["Length of Stay"].mean().reset_index().sort_values(by="Length of Stay", ascending=False)
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
    message(q, is_user=True, key=f"user_{i}")
    message(a, key=f"bot_{i}")

# Downloads
if st.session_state.chat_history:
    chat_df = pd.DataFrame(st.session_state.chat_history, columns=["User", "Assistant"])
    st.download_button("📅 Download Chat as CSV", data=chat_df.to_csv(index=False), file_name="chat_history.csv", mime="text/csv")
