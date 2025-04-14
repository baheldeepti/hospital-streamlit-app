
# 📘 Hospital Chat Assistant - PR-Ready Version with Logging and Enhancements

import streamlit as st
import pandas as pd
import os
import numpy as np
import base64
import altair as alt
from io import BytesIO
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import openai
from datetime import datetime
import logging



# 🔐 OpenAI Key Setup
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# 📊 Page Config
st.set_page_config(page_title="🤖 Chat Assistant", layout="wide")
st.title("🤖 Hospital Chat Assistant")

# 📋 Usage Logging Setup
if "usage_log" not in st.session_state:
    st.session_state["usage_log"] = []

def log_event(event_type, detail):
    timestamp = datetime.now().isoformat()
    st.session_state["usage_log"].append({
        "timestamp": timestamp,
        "type": event_type,
        "detail": detail
    })



# ℹ️ About the App – Sidebar
with st.sidebar.expander("ℹ️ About This App", expanded=False):
    st.markdown("""
    **🧠 Hospital Chat Assistant** is a smart dashboard and AI assistant built to help hospitals explore their data effortlessly.

    #### 🛠️ Powered By:
    - **Streamlit** for UI
    - **LangChain + OpenAI** for conversational logic
    - **Altair** for interactive visualizations

    👩‍⚕️ Created for healthcare analysts, data teams, and curious users to make insights more accessible.
    """)

# 📁 Sidebar: Dataset Loader + Sample Link
with st.sidebar.expander("📁 Load or Upload Dataset", expanded=True):
    st.markdown("""
    If you don't have your own data yet, you can use our sample hospital dataset to try out the dashboard.  
    🔗 [**Download Sample CSV**](https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv)
    """)

    if st.button("Load Sample Hospital Data"):
        try:
            sample_url = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"
            df = pd.read_csv(sample_url)
            st.session_state["main_df"] = df
            st.success("✅ Sample dataset loaded.")
            log_event("dataset_loaded", "Sample dataset")
        except Exception as e:
            st.error(f"❌ Could not load sample dataset: {e}")

    uploaded_file = st.file_uploader("Upload your CSV", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["main_df"] = df
            st.success("✅ File uploaded successfully.")
            log_event("dataset_loaded", "User uploaded dataset")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")



# 📥 Download Usage Log
if st.session_state["usage_log"]:
    log_df = pd.DataFrame(st.session_state["usage_log"])
    st.download_button("📥 Download Usage Log", log_df.to_csv(index=False), file_name="usage_log.csv")

# 🔗 Page Navigation
st.page_link("pages/1_📊_Dashboard.py", label="📊 Dashboard", icon="📊")
st.page_link("pages/4_Dashboard_Feature_Overview.py", label="📘 Dashboard Feature Overview", icon="📘")
st.page_link("pages/3__Chat_Assistant_Feature_Overview.py", label="📄 Chat Assistant Feature Overview", icon="📄")

# 👣 Footer Branding
st.markdown("---")
st.markdown("Made with ❤️ by Deepti Bahel | Powered by Streamlit + LangChain + Altair")
