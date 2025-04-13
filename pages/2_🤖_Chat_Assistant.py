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

# Set API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Sidebar UI
st.sidebar.markdown("## 🤖 Chat Assistant")
st.sidebar.markdown("Ask questions based on filtered hospital data or a policy document.")

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
vectorstore = FAISS.from_documents(doc_chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    retriever=retriever,
    return_source_documents=False
)

# Document RAG upload
uploaded_doc = st.sidebar.file_uploader("Upload Policy Doc (.txt)", type=["txt"])
if uploaded_doc:
    with open(uploaded_doc.name, "w", encoding="utf-8") as f:
        f.write(uploaded_doc.getvalue().decode("utf-8"))

    doc_loader = TextLoader(uploaded_doc.name)
    doc_chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(doc_loader.load())
    vectorstore_doc = FAISS.from_documents(doc_chunks, OpenAIEmbeddings())

    rag_qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        retriever=vectorstore_doc.as_retriever(),
        return_source_documents=False
    )
    st.session_state.rag_qa_chain = rag_qa

# Chat interface with speech and metrics
import altair as alt
import numpy as np
import pyttsx3
import tempfile
import os

# Optionally enable voice
st.sidebar.markdown("### 🔈 Voice Output")
speak_enabled = st.sidebar.checkbox("Enable Text-to-Speech")
st.subheader("💬 Ask Questions About the Data or Policies")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question like 'How many ICU admissions last month?'")

if user_input:
    if speak_enabled:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
    with st.spinner("Thinking..."):
        if "rag_qa_chain" in st.session_state:
            response = st.session_state.rag_qa_chain.run(user_input)
        else:
            response = qa_chain.run(user_input)
                st.session_state.chat_history.append((user_input, response))
        engine.say(response)
        engine.runAndWait()

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
