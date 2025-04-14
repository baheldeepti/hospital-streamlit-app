# 📘 IMPORTS
import streamlit as st
import pandas as pd
import os
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import altair as alt
from fpdf import FPDF
from statsmodels.tsa.seasonal import seasonal_decompose
from streamlit_chat import message

from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 🔐 OpenAI Key Setup
import openai
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# 📊 Page Config
st.set_page_config(page_title="🤖 Chat Assistant", layout="wide")
st.title("🤖 Hospital Chat Assistant")

# 🆕 Sample Dataset Download Prompt
st.markdown("""
If you don't have your own data yet, you can use our sample hospital dataset to try out the dashboard.

🔗 [**Download Sample CSV**](https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv)
""")

# 🧾 Dataset Required
if "main_df" not in st.session_state:
    st.warning("⚠️ Please upload or load a dataset from the sidebar before using the chat assistant.")
    st.stop()

df = st.session_state["main_df"]

# 🔧 Data Cleanup
df["Billing Amount"] = pd.to_numeric(df["Billing Amount"].replace('[\$,]', '', regex=True), errors="coerce")
df["Length of Stay"] = pd.to_numeric(df.get("Length of Stay", pd.Series(dtype=float)), errors="coerce")

# 📚 Sidebar: Data Glossary
with st.sidebar.expander("📚 Data Glossary", expanded=False):
    st.markdown("""
    **Column Descriptions:**
    - **Name**: Patient name
    - **Age**: Patient's age
    - **Gender**: Male/Female
    - **Medical Condition**: Primary diagnosis
    - **Date of Admission**: Admission date
    - **Doctor**: Attending doctor
    - **Hospital**: Hospital name
    - **Billing Amount**: Total charges
    - **Length of Stay**: Days admitted
    """)

# 💾 Session Initialization
for key in ["chat_history", "query_log", "fallback_log"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "query_log" else {}

# 💬 Chat Section
st.markdown("### 💬 Chat with Assistant")
for i, (q, a) in enumerate(st.session_state.chat_history):
    message(q, is_user=True, key=f"user_{i}")
    message(a, key=f"bot_{i}")

# 💡 Suggested Prompts
suggestions = [
    "Show billing trend by hospital",
    "Patient count by gender",
    "Top conditions by test results",
    "Total billing by insurance provider",
    "Average age of patients by condition"
]

cols = st.columns(len(suggestions))
for i, s in enumerate(suggestions):
    if cols[i].button(s):
        st.session_state["chat_input"] = s
        st.session_state["query_log"][s] = st.session_state["query_log"].get(s, 0) + 1

# 🧠 User Input
user_input = st.text_input("Ask a question", key="chat_input", placeholder="E.g. Average stay by condition")

# 🛡️ Safe Agent Execution
def respond_to_query(query):
    try:
        agent = create_pandas_dataframe_agent(
            llm=ChatOpenAI(temperature=0),
            df=df,
            verbose=False
        )
        return agent.run(query)
    except Exception as e:
        st.session_state["fallback_log"].append(query)
        return (
            "⚠️ This query relies on code execution tools that are not supported on this platform.\n\n"
            "Please explore the dashboard or try uploading different data."
        )

# 📘 Keyword Tooltips
tooltips = {
    "billing": "Total amount charged to the patient",
    "stay": "Length of stay in days",
    "gender": "Gender breakdown of patients",
    "condition": "Primary medical condition during admission"
}

def add_tooltip(response, terms):
    for word, tip in terms.items():
        if word in response.lower():
            response += f"\n\n🛈 *{word.capitalize()}* refers to: {tip}"
    return response

# 🧠 Handle Chat
if user_input:
    with st.spinner("🤖 Assistant is thinking..."):
        response = respond_to_query(user_input)
        response = add_tooltip(response, tooltips)
        st.session_state.chat_history.append((user_input, response))

# 📊 Auto Chart Preview
st.markdown("### 📊 Auto Chart Preview")
if "chat_input" in st.session_state:
    query = st.session_state.chat_input.lower()

    def export_chart(chart, filename):
        buf = BytesIO()
        chart.save(buf, format="png")
        b64 = base64.b64encode(buf.getvalue()).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">📩 Download PNG</a>'
        st.markdown(href, unsafe_allow_html=True)

    def export_chart_pdf(dataframe, filename):
        buffer = BytesIO()
        fig, ax = plt.subplots()
        dataframe.plot(kind='barh', x=dataframe.columns[0], y=dataframe.columns[1], ax=ax)
        fig.tight_layout()
        plt.savefig(buffer, format="pdf")
        buffer.seek(0)
        b64_pdf = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}.pdf">📄 Download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

    if "billing" in query and "hospital" in query:
        chart_data = df.groupby("Hospital")["Billing Amount"].sum().reset_index()
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("Hospital:N", sort="-y"),
            y="Billing Amount:Q"
        ).properties(title="Billing by Hospital")
        st.altair_chart(chart, use_container_width=True)
        export_chart(chart, "billing_by_hospital")
        export_chart_pdf(chart_data, "billing_by_hospital")

    elif "gender" in query:
        chart_data = df["Gender"].value_counts().reset_index()
        chart_data.columns = ["Gender", "Count"]
        chart = alt.Chart(chart_data).mark_bar().encode(
            x="Gender:N",
            y="Count:Q"
        ).properties(title="Patient Count by Gender")
        st.altair_chart(chart, use_container_width=True)
        export_chart(chart, "patient_count_by_gender")
        export_chart_pdf(chart_data, "patient_count_by_gender")

# 🏆 Leaderboard
if st.session_state.query_log:
    leaderboard_df = pd.DataFrame(
        sorted(st.session_state.query_log.items(), key=lambda x: x[1], reverse=True),
        columns=["Query", "Clicks"]
    )
    st.markdown("### 🏆 Most Clicked Suggestions")
    st.dataframe(leaderboard_df, use_container_width=True)
    st.download_button("⬇️ Download Query Log (CSV)", data=leaderboard_df.to_csv(index=False), file_name="query_log.csv")

# 🧾 Fallback Queries
if st.session_state.fallback_log:
    fallback_df = pd.DataFrame(st.session_state.fallback_log, columns=["Query"])
    st.markdown("### 🧾 Fallback Queries")
    st.dataframe(fallback_df, use_container_width=True)
    st.download_button("⬇️ Download Fallback Queries", data=fallback_df.to_csv(index=False), file_name="fallback_queries.csv")

# 💬 Chat Download
if st.session_state.chat_history:
    chat_df = pd.DataFrame(st.session_state.chat_history, columns=["User", "Assistant"])
    st.download_button("🗎 Download Chat History (CSV)", data=chat_df.to_csv(index=False), file_name="chat_history.csv")

# 🔗 Page Navigation
st.page_link("pages/1_📊_Dashboard.py", label="📊 Dashboard", icon="📊")
st.page_link("pages/4_Dashboard_Feature_Overview.py", label="📘 Dashboard Feature Overview", icon="📘")
st.page_link("pages/3__Chat_Assistant_Feature_Overview.py", label="📄 Chat Assistant Feature Overview", icon="📄")
