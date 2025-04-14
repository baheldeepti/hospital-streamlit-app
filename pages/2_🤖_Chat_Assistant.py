import streamlit as st

# ========================================
# 🔧 Page Configuration
# ========================================
st.set_page_config(
    page_title="🏥 Hospital BI Reporting Tool",
    page_icon="🏥",
    layout="wide"
)

# ========================================
# 🏠 App Title & Welcome Message
# ========================================
st.title("🏥 Hospital BI Reporting Tool")

st.markdown("""
Welcome to the **Hospital Business Intelligence Reporting Tool**!

This app empowers users to:
- Analyze hospital metrics
- Explore interactive dashboards
- Use conversational AI for data exploration
""")

# ========================================
# ℹ️ How It Works (Collapsible Instructions)
# ========================================
with st.expander("ℹ️ How It Works", expanded=False):
    st.markdown("""
    **1. 📊 Dashboard:**  
    Visualize KPIs, patient trends, and billing performance.

    **2. 🤖 Chat Assistant:**  
    Ask natural language questions and receive data-driven insights.

    **3. 📄 Feature Overview Pages:**  
    Understand how each tool works.

    **4. 📁 Data:**  
    Upload your own hospital dataset or use the provided sample.

    ---
    💡 **Tip:** Download charts and logs for reporting!
    """)

# ========================================
# 🔗 Navigation to App Pages
# ========================================
st.markdown("## 🔗 Navigate to App Modules")

st.page_link("pages/1_📊_Dashboard.py", label="📊 Dashboard", icon="📊")
st.page_link("pages/2_🤖_Chat_Assistant.py", label="🤖 Chat Assistant", icon="🤖")
st.page_link("pages/3__Chat_Assistant_Feature_Overview.py", label="📄 Chat Assistant Features", icon="📄")
st.page_link("pages/4_Dashboard_Feature_Overview.py", label="📘 Dashboard Features", icon="📘")

# ========================================
# 👩‍💻 About the Developer
# ========================================
st.markdown("---")
st.markdown("### 👩‍💻 About the Developer")

st.markdown("""
Built by **Deepti Bahel**, a Senior BI Engineer who blends data engineering, AI, and design to build impactful tools for healthcare analytics.

👉 [Connect on LinkedIn](https://www.linkedin.com/in/deepti-bahel/)
""")

# ========================================
# 🦶 Footer Section
# ========================================
st.markdown("---")
st.markdown("#### 🔻 Quick Access Links & Credits")

st.markdown("Made with ❤️ by Deepti Bahel | Powered by Streamlit + LangChain + Altair")

st.page_link("pages/1_📊_Dashboard.py", label="📊 Dashboard", icon="📊")
st.page_link("pages/3__Chat_Assistant_Feature_Overview.py", label="📄 Chat Features", icon="📄")
st.page_link("pages/4_Dashboard_Feature_Overview.py", label="📘 Dashboard Features", icon="📘")

# Optional: Add light styling at the end
st.caption("© 2025 Hospital BI Tool. All rights reserved.")
