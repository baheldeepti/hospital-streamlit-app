import streamlit as st

st.set_page_config(page_title="📊 Dashboard Feature Overview", layout="wide")
st.title("📊 Hospital Dashboard - Feature Overview")

st.markdown("""
This document outlines the current capabilities of the **Hospital Dashboard** page within the Hospital Streamlit App.

with open("docs/Dashboard_Feature_Overview.md", "r") as f:
    md_content = f.read()

st.markdown(md_content, unsafe_allow_html=True)

st.page_link("pages/1_📊_Dashboard.py", label="📊Dashboard", icon="📊")
st.page_link("pages/2_🤖_Chat_Assistant.py", label="🤖Chat Assistant", icon="📊")
st.page_link("pages/4_Dashboard_Feature_Overview.py", label="📊Dashboard Feature Overview", icon="📊")
st.page_link("pages/3__ChatAssistantFeature_Overview.py", label="📄Chat Assistant Feature Overview", icon="📄")
