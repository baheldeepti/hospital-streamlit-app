# 📘 Introduction

st.set_page_config(page_title="Hospital Data Assistant", layout="wide")

# Animated header
st.image("https://lottie.host/5e5aa1c7-1781-4ed5-b05f-00584e963b48/oBHF65wAbI.json", caption="Interactive Hospital Insights", use_column_width=True)

st.title("🏥 Hospital Data Assistant")

# Collapsible section
with st.expander("ℹ️ How it works"):
    st.markdown("""
    - 📁 Upload your hospital dataset or use a sample
    - 🧠 Ask questions like “What was the average billing last month?”
    - 📊 See auto-generated visualizations
    - 💬 Download chatbot conversations and usage logs
    """)

# Links
st.markdown("""
### 👩‍💻 About the Developer
This app was built by [Deepti Bahel](https://www.linkedin.com/in/deepti-bahel/) to help turn complex hospital data into insights using AI.

Check out the full source code and contribute on GitHub:
""")
col1, col2 = st.columns(2)
with col1:
    st.link_button("🌐 GitHub Repo", "https://github.com/baheldeepti/hospital-streamlit-app/tree/main")
with col2:
    st.link_button("👤 LinkedIn Profile", "https://www.linkedin.com/in/deepti-bahel/")

st.markdown("""
Welcome to the **Hospital Data Assistant** 👋

This tool lets you:
- 🔍 Upload and explore hospital datasets
- 🤖 Ask data-driven questions using a chatbot
- 📊 Visualize insights (like trends and patient stats)
- 📁 Export chat logs and token usage

Upload your data or use the sample to get started!
""")

# 💬 CHATBOT SECTION: Integrated with Main App using a safer structured agent

import os
import shutil
import streamlit as st
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

# Set API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Sidebar UI
uploaded_file = st.sidebar.file_uploader("📁 Upload a CSV file to analyze", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')

    # Preview & Validate Section
st.sidebar.markdown("### 🔍 Preview & Validate")

# Configurable required fields
required_cols = st.sidebar.multiselect("✅ Required Columns", ["Billing Amount", "Length of Stay", "Medical Condition"], default=["Billing Amount", "Length of Stay"])
if not set(required_cols).issubset(df.columns):
    st.sidebar.error("❌ Uploaded file is missing required columns: Billing Amount and Length of Stay.")
else:
        st.session_state.main_df = df
        st.sidebar.success("✅ Custom dataset loaded successfully!")
        st.sidebar.markdown("### 📊 Sample Preview")
        st.sidebar.dataframe(df.head())

        # Column stats
        st.sidebar.markdown("### 📈 Column Stats")
        col_stats = df[required_cols].describe(include='all').T
        col_stats['missing'] = df[required_cols].isnull().sum()
        col_stats['missing_pct'] = (col_stats['missing'] / len(df) * 100).round(2)

        # Alert if missing % exceeds threshold
        high_missing_cols = col_stats[col_stats['missing_pct'] > 25].index.tolist()
        if high_missing_cols:
            st.sidebar.warning(f"⚠️ High missing data in: {', '.join(high_missing_cols)}")
        st.sidebar.dataframe(col_stats[['count', 'mean', 'min', 'max', 'missing']].fillna("-").astype(str))
st.sidebar.markdown("## 🤖 Chat Assistant")
st.sidebar.markdown("Ask questions based on filtered hospital data.")
clear_cache = st.sidebar.button("🗑️ Clear Embedding Cache")
if clear_cache:
    import shutil
    if os.path.exists(".embedding_cache"):
        shutil.rmtree(".embedding_cache")
        st.sidebar.success("Embedding cache cleared.")

# Load hospital dataset
if 'main_df' not in st.session_state:
    sample_url = "https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv"
    df = pd.read_csv(sample_url)
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    st.session_state.main_df = df
else:
    df = st.session_state.main_df

# Filters
# Skipping department and date filters as those fields are not present
filtered_df = df.copy()

# Turn filtered data into a .txt for retrieval
csv_path = "filtered_data.txt"
filtered_df.to_csv(csv_path, index=False)
loader = TextLoader(csv_path)
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
doc_chunks = splitter.split_documents(docs)

max_chunks = st.sidebar.slider("Max Chunks for Embedding", min_value=50, max_value=500, value=150, step=50)
estimated_tokens = max_chunks * 500  # rough estimate, 500 tokens per chunk
st.sidebar.markdown(f"🧠 Estimated tokens for embedding: **{estimated_tokens}**")
if estimated_tokens > 900000:
    st.sidebar.warning("⚠️ Estimated tokens exceed 900,000. You may hit your OpenAI plan's TPM limit.")

from hashlib import md5
import os

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
    st.error("⚠️ Token limit exceeded. Try reducing the chunk limit from the sidebar.")
    st.stop()

rag_qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    retriever=vectorstore_doc.as_retriever(),
    return_source_documents=False
)
st.session_state.rag_qa_chain = rag_qa

# Chat interface
st.subheader("💬 Ask Questions About the Data")

# Sidebar control for adjustable memory limit
max_history = st.sidebar.slider("Max Chat History Length", min_value=5, max_value=20, value=10, step=1)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question like 'How many ICU admissions last month?'")

if not user_input.strip():
    st.warning("Please enter a question.")
    st.stop()

if user_input:
    with st.spinner("Thinking..."):
        try:
            # Truncate long history to stay within 16K token limit
            if len(st.session_state.chat_history) > max_history:
                st.warning(f"🧠 Chat history truncated to the last {max_history} entries to prevent context overflow.")
                st.session_state.chat_history = st.session_state.chat_history[-max_history:]

            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            all_context = "
".join([q + "
" + a for q, a in st.session_state.chat_history]) + user_input
            context_tokens = len(encoding.encode(all_context))
st.sidebar.markdown(f"🧾 Estimated context tokens: **{context_tokens}**")
            if "token_log" not in st.session_state:
                st.session_state.token_log = []
            st.session_state.token_log.append({"prompt": user_input, "tokens": context_tokens})

            response = st.session_state.rag_qa_chain.run(user_input)
        except Exception as e:
            st.error("⚠️ Your question is too long or the context is too large. Please simplify your input or reduce the document size.")
            st.write(f"**Error logged:** {e}")
            st.stop()
        st.session_state.chat_history.append((user_input, response))

        # Auto-tagging
        tags = []
        if any(word in user_input.lower() for word in ["bill", "cost"]):
            tags.append("Billing")
        if "stay" in user_input.lower():
            tags.append("Length of Stay")
        if tags:
            st.markdown(f"**Tags:** {', '.join(tags)}")

        # Visual summaries
        if "billing trend" in user_input.lower():
            if "Billing Amount" in filtered_df.columns and "Date of Admission" in filtered_df.columns:
                trend_df = filtered_df.groupby(pd.Grouper(key="Date of Admission", freq="M"))[["Billing Amount"]].sum().reset_index()
                chart = alt.Chart(trend_df).mark_line(point=True).encode(
                    x="Date of Admission",
                    y="Billing Amount",
                    tooltip=["Date of Admission", "Billing Amount"]
                ).properties(title="Monthly Billing Trend")
                st.altair_chart(chart, use_container_width=True)

        if "length of stay" in user_input.lower() and "by condition" in user_input.lower():
            if "Medical Condition" in filtered_df.columns and "Length of Stay" in filtered_df.columns:
                los_df = filtered_df.groupby("Medical Condition")["Length of Stay"].mean().reset_index()
                chart = alt.Chart(los_df).mark_bar().encode(
                    x="Medical Condition",
                    y="Length of Stay",
                    tooltip=["Medical Condition", "Length of Stay"]
                ).properties(title="Avg Length of Stay by Medical Condition")
                st.altair_chart(chart, use_container_width=True)

        if "patients by condition" in user_input.lower():
            if "Medical Condition" in filtered_df.columns:
                pie_df = filtered_df["Medical Condition"].value_counts().reset_index()
                pie_df.columns = ["Medical Condition", "Count"]
                chart = alt.Chart(pie_df).mark_arc().encode(
                    theta="Count",
                    color="Medical Condition",
                    tooltip=["Medical Condition", "Count"]
                ).properties(title="Patient Distribution by Medical Condition")
                st.altair_chart(chart, use_container_width=True)
        

        

        

for i, (q, a) in enumerate(st.session_state.chat_history):
    message(q, is_user=True, key=f"user_{i}")
    message(a, key=f"bot_{i}")

# CSV/Excel download of chat

# Export token usage log
if "token_log" in st.session_state:
    token_df = pd.DataFrame(st.session_state.token_log)
    token_csv = token_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Token Usage Log", data=token_csv, file_name="token_usage_log.csv", mime="text/csv")
if st.session_state.chat_history:
    chat_df = pd.DataFrame(st.session_state.chat_history, columns=["User", "Assistant"])
    csv_data = chat_df.to_csv(index=False).encode("utf-8")
    excel_buffer = pd.ExcelWriter("chat_history.xlsx", engine='xlsxwriter')
    chat_df.to_excel(excel_buffer, index=False, sheet_name='Chat')
    excel_buffer.close()
    st.download_button("📥 Download Chat as CSV", data=csv_data, file_name="chat_history.csv", mime="text/csv")
