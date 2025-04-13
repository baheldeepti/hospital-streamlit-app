# 📘 Introduction
import streamlit as st
from streamlit_lottie import st_lottie
import requests

# Page setup
st.set_page_config(
    page_title="Hospital Data Assistant",
    layout="wide",
    page_icon="🏥"
)

# Load Lottie animation from URL
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://lottie.host/5e5aa1c7-1781-4ed5-b05f-00584e963b48/oBHF65wAbI.json"
lottie_animation = load_lottie_url(lottie_url)

# Display animated header
st_lottie(
    lottie_animation,
    speed=1,
    reverse=False,
    loop=True,
    quality="high",
    height=300,
    key="hospital-animation"
)

# Title
st.title("🏥 Hospital Data Assistant")

# ℹ️ How It Works
with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
    Welcome to the **Hospital Data Assistant** 👋  
    This tool helps you uncover powerful insights from hospital datasets using AI.  

    **Steps to get started:**
    - 📁 Upload your hospital dataset or use the built-in sample
    - 🤖 Ask questions like “What was the average billing last month?”
    - 📊 View auto-generated visualizations instantly
    - 💬 Download chat history and usage logs
    """)

# 👩‍💻 About the Developer
st.markdown("### 👩‍💻 About the Developer")
st.markdown("""
Built with ❤️ by [Deepti Bahel](https://www.linkedin.com/in/deepti-bahel/), this app turns hospital data into conversational insights with the help of AI and interactive charts.

Explore the code, contribute, or connect!
""")

col1, col2 = st.columns(2)
with col1:
    st.link_button("🌐 GitHub Repo", "https://github.com/baheldeepti/hospital-streamlit-app/tree/main")
with col2:
    st.link_button("👤 LinkedIn Profile", "https://www.linkedin.com/in/deepti-bahel/")

# Intro Section Call-to-Action
st.markdown("""
---

### 🚀 What You Can Do Here:
- 🔍 Upload and explore hospital datasets
- 🧠 Ask data-related questions in plain English
- 📊 View trends and statistical charts instantly
- 📥 Export chat logs and prompt token usage for audit or reference

👉 **Get started by uploading your file from the sidebar**, or use the sample dataset!
""")
