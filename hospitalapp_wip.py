# 📘 Introduction
import streamlit as st
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
