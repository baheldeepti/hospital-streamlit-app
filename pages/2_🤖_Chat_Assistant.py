# 💬 CHATBOT SECTION: Integrated with Main App

# Import all required packages for chatbot
import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import openai

# Set API key for OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Add chatbot UI section to the sidebar
st.sidebar.markdown("## 🤖 Chat Assistant")
st.sidebar.markdown("Interact with hospital data or upload a policy document to enhance responses.")

# Add department/date filter inputs
# Add department/date filter inputs
if 'main_df' not in st.session_state:
    sample_url = "https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv"
    df = pd.read_csv(sample_url)
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    st.session_state.main_df = df
else:
    df = st.session_state.main_df


dept_options = df["Department"].unique().tolist() if "Department" in df else []
dept_filter = st.sidebar.multiselect("Filter by Department", options=dept_options, default=dept_options)
date_range = st.sidebar.date_input("Filter by Admission Date Range", [])

# Apply filters to context-aware dataframe
filtered_df = df.copy()
if dept_filter:
    filtered_df = filtered_df[filtered_df["Department"].isin(dept_filter)]
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["Date of Admission"] >= pd.to_datetime(date_range[0])) &
        (filtered_df["Date of Admission"] <= pd.to_datetime(date_range[1]))
    ]

# Create chatbot agent with caching for performance
@st.cache_resource
def get_chat_agent(data):
    return create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        data,
        verbose=False,
        memory=ConversationBufferMemory()
    )

agent = get_chat_agent(filtered_df)

# Add optional RAG (doc-based retrieval) section
uploaded_doc = st.sidebar.file_uploader("Upload Policy Doc (.txt)", type=["txt"])
if uploaded_doc:
    with open(uploaded_doc.name, "w", encoding="utf-8") as f:
        f.write(uploaded_doc.getvalue().decode("utf-8"))

    loader = TextLoader(uploaded_doc.name)
    docs = loader.load()
    chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )

    st.session_state.rag_qa_chain = qa_chain

# Chat interface in main section
st.subheader("💬 Ask Questions About the Data or Policies")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question like 'What’s the average billing for ICU patients last month?'")

if user_input:
    with st.spinner("Thinking..."):
        if "rag_qa_chain" in st.session_state:
            response = st.session_state.rag_qa_chain.run(user_input)
        else:
            response = agent.run(user_input)
        st.session_state.chat_history.append((user_input, response))

for i, (q, a) in enumerate(st.session_state.chat_history):
    message(q, is_user=True, key=f"user_{i}")
    message(a, key=f"bot_{i}")
