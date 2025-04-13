# 📘 Introduction
import streamlit as st

# 💬 CHATBOT SECTION
import os
import shutil
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from streamlit_chat import message
import openai
import altair as alt
import numpy as np
from hashlib import md5

openai.api_key = st.secrets["OPENAI_API_KEY"]

uploaded_file = st.sidebar.file_uploader("📁 Upload a CSV file to analyze", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')

st.sidebar.markdown("### 🔍 Preview & Validate")

required_cols = st.sidebar.multiselect("✅ Required Columns", ["Billing Amount", "Length of Stay", "Medical Condition"], default=["Billing Amount", "Length of Stay"])
if uploaded_file is not None and not set(required_cols).issubset(df.columns):
    st.sidebar.error("❌ Uploaded file is missing required columns: Billing Amount and Length of Stay.")
elif uploaded_file is not None:
    st.session_state.main_df = df
    st.sidebar.success("✅ Custom dataset loaded successfully!")
    st.sidebar.markdown("### 📊 Sample Preview")
    st.sidebar.dataframe(df.head())
    st.sidebar.markdown("### 📈 Column Stats")
    col_stats = df[required_cols].describe(include='all').T
    col_stats['missing'] = df[required_cols].isnull().sum()
    col_stats['missing_pct'] = (col_stats['missing'] / len(df) * 100).round(2)
    high_missing_cols = col_stats[col_stats['missing_pct'] > 25].index.tolist()
    if high_missing_cols:
        st.sidebar.warning(f"⚠️ High missing data in: {', '.join(high_missing_cols)}")
    st.sidebar.dataframe(col_stats[['count', 'mean', 'min', 'max', 'missing']].fillna("-").astype(str))

st.sidebar.markdown("## 🤖 Chat Assistant")
clear_cache = st.sidebar.button("🗑️ Clear Embedding Cache")
if clear_cache:
    if os.path.exists(".embedding_cache"):
        shutil.rmtree(".embedding_cache")
        st.sidebar.success("Embedding cache cleared.")

if 'main_df' not in st.session_state:
    sample_url = "https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv"
    df = pd.read_csv(sample_url)
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    st.session_state.main_df = df
else:
    df = st.session_state.main_df

filtered_df = df.copy()
csv_path = "filtered_data.txt"
filtered_df.to_csv(csv_path, index=False)

loader = TextLoader(csv_path)
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
doc_chunks = splitter.split_documents(docs)

max_chunks = st.sidebar.slider("Max Chunks for Embedding", min_value=50, max_value=500, value=150, step=50)
estimated_tokens = max_chunks * 500
st.sidebar.markdown(f"🧠 Estimated tokens for embedding: **{estimated_tokens}**")
if estimated_tokens > 900000:
    st.sidebar.warning("⚠️ Estimated tokens exceed 900,000.")

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
    limited_doc_chunks = doc_chunks[:max_chunks]
    vectorstore_doc = safe_embed(limited_doc_chunks)
except Exception as e:
    st.error("⚠️ Token limit exceeded.")
    st.stop()

rag_qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    retriever=vectorstore_doc.as_retriever(),
    return_source_documents=False
)
st.session_state.rag_qa_chain = rag_qa

st.subheader("💬 Ask Questions About the Data")
max_history = st.sidebar.slider("Max Chat History Length", min_value=5, max_value=20, value=10, step=1)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

example_queries = [
    "Show billing trend for January",
    "What is the average length of stay by condition?",
    "How many patients were admitted last week?"
]

st.markdown("### 💡 Example Queries")
st.selectbox("Need inspiration?", example_queries, index=0, key="example_prompt")

user_input = st.text_input("Ask a question like 'How many ICU admissions last month?'", value=st.session_state.example_prompt)

if not user_input.strip():
    st.warning("Please enter a question.")
    st.stop()

if user_input:
    with st.spinner("Thinking..."):
        try:
            if len(st.session_state.chat_history) > max_history:
                st.warning(f"🧠 Chat history truncated to the last {max_history} entries.")
                st.session_state.chat_history = st.session_state.chat_history[-max_history:]

            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

            history_text = "\n".join([q + "\n" + a for q, a in st.session_state.chat_history])
            all_context = history_text + "\n" + user_input

            context_tokens_only = len(encoding.encode(history_text))
            user_tokens = len(encoding.encode(user_input))
            context_tokens = len(encoding.encode(all_context))

            st.sidebar.markdown(f"🧾 Prompt tokens: **{user_tokens}**, Context tokens: **{context_tokens_only}**, Total: **{context_tokens}**")

            if "token_log" not in st.session_state:
                st.session_state.token_log = []
            st.session_state.token_log.append({"prompt": user_input, "tokens": context_tokens})

            if user_tokens > 3000:
                st.error("❌ Your input is too long.")
                st.stop()

            response = st.session_state.rag_qa_chain.run(user_input)
        except Exception as e:
            st.error("⚠️ Your question is too long or the context is too large.")
            st.write(f"**Error:** {e}")
            st.stop()

        st.session_state.chat_history.append((user_input, response))

        # Tagging and visualization logic here (billing trends, length of stay, pie charts, etc.)
        # ... your existing charts remain unchanged ...

for i, (q, a) in enumerate(st.session_state.chat_history):
    message(q, is_user=True, key=f"user_{i}")
    message(a, key=f"bot_{i}")

# Download chat and token usage
if "token_log" in st.session_state:
    token_df = pd.DataFrame(st.session_state.token_log)
    st.download_button("📥 Download Token Usage Log", data=token_df.to_csv(index=False), file_name="token_usage_log.csv", mime="text/csv")

if st.session_state.chat_history:
    chat_df = pd.DataFrame(st.session_state.chat_history, columns=["User", "Assistant"])
    st.download_button("📥 Download Chat as CSV", data=chat_df.to_csv(index=False), file_name="chat_history.csv", mime="text/csv")
