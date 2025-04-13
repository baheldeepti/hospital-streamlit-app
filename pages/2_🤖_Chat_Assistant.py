
import os  
import shutil
# 💬 CHATBOT SECTION: Integrated with Main App using a safer structured agent

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
dept_options = df["Department"].unique().tolist() if "Department" in df else []
dept_filter = st.sidebar.multiselect("Filter by Department", options=dept_options, default=dept_options)
date_range = st.sidebar.date_input("Filter by Admission Date Range", [])

filtered_df = df.copy()
if dept_filter:
    filtered_df = filtered_df[filtered_df["Department"].isin(dept_filter)]
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["Date of Admission"] >= pd.to_datetime(date_range[0])) &
        (filtered_df["Date of Admission"] <= pd.to_datetime(date_range[1]))
    ]

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
from sklearn.feature_extraction.text import CountVectorizer
st.subheader("💬 Ask Questions About the Data")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question like 'How many ICU admissions last month?'")

if not user_input.strip():
    st.warning("Please enter a question.")
    st.stop()

if user_input:
    with st.spinner("Thinking..."):
        if "rag_qa_chain" in st.session_state:
            response = st.session_state.rag_qa_chain.run(user_input)
        else:
            response = qa_chain.run(user_input)
        st.session_state.chat_history.append((user_input, response))

        # Auto-tagging
        tags = []
        if any(word in user_input.lower() for word in ["bill", "cost"]):
            tags.append("Billing")
        if "stay" in user_input.lower():
            tags.append("Length of Stay")
        if "department" in user_input.lower():
            tags.append("Department")
        if tags:
            st.markdown(f"**Tags:** {', '.join(tags)}")

        # Visual summaries
        if "billing by department" in user_input.lower():
            chart_df = filtered_df.groupby("Department")["Billing Amount"].sum().reset_index()
            chart = alt.Chart(chart_df).mark_bar().encode(
                x="Department",
                y="Billing Amount",
                tooltip=["Department", "Billing Amount"]
            ).properties(title="Total Billing by Department")
            st.altair_chart(chart, use_container_width=True)

        if "length of stay by department" in user_input.lower():
            chart_df = filtered_df.groupby("Department")["Length of Stay"].mean().reset_index()
            chart = alt.Chart(chart_df).mark_bar().encode(
                x="Department",
                y="Length of Stay",
                tooltip=["Department", "Length of Stay"]
            ).properties(title="Average Length of Stay by Department")
            st.altair_chart(chart, use_container_width=True)

        if "patient distribution" in user_input.lower():
            pie_df = filtered_df["Department"].value_counts().reset_index()
            pie_df.columns = ["Department", "Count"]
            chart = alt.Chart(pie_df).mark_arc().encode(
                theta="Count",
                color="Department",
                tooltip=["Department", "Count"]
            ).properties(title="Patient Distribution by Department")
            st.altair_chart(chart, use_container_width=True)

for i, (q, a) in enumerate(st.session_state.chat_history):
    message(q, is_user=True, key=f"user_{i}")
    message(a, key=f"bot_{i}")

# CSV/Excel download of chat
if st.session_state.chat_history:
    chat_df = pd.DataFrame(st.session_state.chat_history, columns=["User", "Assistant"])
    csv_data = chat_df.to_csv(index=False).encode("utf-8")
    excel_buffer = pd.ExcelWriter("chat_history.xlsx", engine='xlsxwriter')
    chat_df.to_excel(excel_buffer, index=False, sheet_name='Chat')
    excel_buffer.close()
    st.download_button("📥 Download Chat as CSV", data=csv_data, file_name="chat_history.csv", mime="text/csv")

# Set API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Sidebar UI
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
dept_options = df["Department"].unique().tolist() if "Department" in df else []
dept_filter = st.sidebar.multiselect("Filter by Department", options=dept_options, default=dept_options)
date_range = st.sidebar.date_input("Filter by Admission Date Range", [])

filtered_df = df.copy()
if dept_filter:
    filtered_df = filtered_df[filtered_df["Department"].isin(dept_filter)]
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["Date of Admission"] >= pd.to_datetime(date_range[0])) &
        (filtered_df["Date of Admission"] <= pd.to_datetime(date_range[1]))
    ]

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

limited_doc_chunks = doc_chunks[:max_chunks]
vectorstore_doc = safe_embed(limited_doc_chunks)

rag_qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    retriever=vectorstore_doc.as_retriever(),
    return_source_documents=False
)
st.session_state.rag_qa_chain = rag_qa

# Chat interface
from sklearn.feature_extraction.text import CountVectorizer
st.subheader("💬 Ask Questions About the Data")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question like 'How many ICU admissions last month?'")

if not user_input.strip():
    st.warning("Please enter a question.")
    st.stop()

if user_input:
    with st.spinner("Thinking..."):
        if "rag_qa_chain" in st.session_state:
            response = st.session_state.rag_qa_chain.run(user_input)
        else:
            response = qa_chain.run(user_input)
        st.session_state.chat_history.append((user_input, response))

        # Auto-tagging
        tags = []
        if any(word in user_input.lower() for word in ["bill", "cost"]):
            tags.append("Billing")
        if "stay" in user_input.lower():
            tags.append("Length of Stay")
        if "department" in user_input.lower():
            tags.append("Department")
        if tags:
            st.markdown(f"**Tags:** {', '.join(tags)}")

        # Visual summaries
        if "billing by department" in user_input.lower():
            chart_df = filtered_df.groupby("Department")["Billing Amount"].sum().reset_index()
            chart = alt.Chart(chart_df).mark_bar().encode(
                x="Department",
                y="Billing Amount",
                tooltip=["Department", "Billing Amount"]
            ).properties(title="Total Billing by Department")
            st.altair_chart(chart, use_container_width=True)

        if "length of stay by department" in user_input.lower():
            chart_df = filtered_df.groupby("Department")["Length of Stay"].mean().reset_index()
            chart = alt.Chart(chart_df).mark_bar().encode(
                x="Department",
                y="Length of Stay",
                tooltip=["Department", "Length of Stay"]
            ).properties(title="Average Length of Stay by Department")
            st.altair_chart(chart, use_container_width=True)

        if "patient distribution" in user_input.lower():
            pie_df = filtered_df["Department"].value_counts().reset_index()
            pie_df.columns = ["Department", "Count"]
            chart = alt.Chart(pie_df).mark_arc().encode(
                theta="Count",
                color="Department",
                tooltip=["Department", "Count"]
            ).properties(title="Patient Distribution by Department")
            st.altair_chart(chart, use_container_width=True)

for i, (q, a) in enumerate(st.session_state.chat_history):
    message(q, is_user=True, key=f"user_{i}")
    message(a, key=f"bot_{i}")

# CSV/Excel download of chat
if st.session_state.chat_history:
    chat_df = pd.DataFrame(st.session_state.chat_history, columns=["User", "Assistant"])
    csv_data = chat_df.to_csv(index=False).encode("utf-8")
    excel_buffer = pd.ExcelWriter("chat_history.xlsx", engine='xlsxwriter')
    chat_df.to_excel(excel_buffer, index=False, sheet_name='Chat')
    excel_buffer.close()
    st.download_button("📥 Download Chat as CSV", data=csv_data, file_name="chat_history.csv", mime="text/csv")
