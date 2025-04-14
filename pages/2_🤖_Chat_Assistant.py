
# 📘 Hospital Chat Assistant - v1.3.6 FINAL

import streamlit as st
import pandas as pd
import os
import numpy as np
import altair as alt
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import openai
from datetime import datetime
import time

# 🔐 OpenAI Key Setup
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# 📊 Page Config
st.set_page_config(page_title="🤖 Chat Assistant", layout="wide")
st.title("🤖 Hospital Chat Assistant")

# 🐞 Debug Mode
DEBUG_MODE = st.sidebar.checkbox("🐞 Enable Debug Mode")
def debug_log(msg):
    if DEBUG_MODE:
        st.sidebar.markdown(f"🔍 **Debug**: {msg}")

# 📦 Export CSV Helper
def export_csv(dataframe, filename):
    csv = dataframe.to_csv(index=False).encode()
    st.download_button("📩 Download CSV", csv, file_name=f"{filename}.csv", mime="text/csv")

# 📋 Usage Logging
if "usage_log" not in st.session_state:
    st.session_state["usage_log"] = []
def log_event(event_type, detail):
    st.session_state["usage_log"].append({
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "detail": detail
    })

# ℹ️ About
with st.sidebar.expander("ℹ️ About This App", expanded=False):
    st.markdown("""
    **🧠 Hospital Chat Assistant** helps hospitals explore data interactively.

    - 🤖 Chat with an AI agent
    - 📊 Create charts from prompts or dropdowns
    - 📋 View KPIs and session summary
    - 🔍 Search glossary for help

    Created by Deepti Bahel
    """)

# 📁 Load Data
with st.sidebar.expander("📁 Load or Upload Dataset", expanded=True):
    st.markdown("""Try with your own CSV or use a sample dataset:  
    🔗 [**Download Sample CSV**](https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv)""")
    if st.button("Load Sample Hospital Data"):
        df = pd.read_csv("https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv")
        st.session_state["main_df"] = df
        st.success("✅ Sample dataset loaded.")
        log_event("dataset_loaded", "Sample")
    uploaded_file = st.file_uploader("Upload your CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["main_df"] = df
        st.success("✅ File uploaded successfully.")
        log_event("dataset_loaded", "User CSV")

if "main_df" not in st.session_state:
    st.warning("⚠️ Please load or upload a dataset.")
    st.stop()

df = st.session_state["main_df"]
df["Billing Amount"] = pd.to_numeric(df["Billing Amount"].replace('[\$,]', '', regex=True), errors="coerce")
df["Length of Stay"] = pd.to_numeric(df.get("Length of Stay", pd.Series(dtype=float)), errors="coerce")
df["Billing Formatted"] = df["Billing Amount"].apply(lambda x: f"${x/1000:.1f}K" if pd.notnull(x) else "N/A")

# 🔍 Glossary (Updated Based on Kaggle Dataset)
with st.sidebar.expander("🔍 Data Glossary", expanded=False):
    st.text_input("Search term", key="glossary_search")
    glossary = {
        "Name": "Patient’s name associated with the record.",
        "Age": "Age of the patient at the time of admission (in years).",
        "Gender": "Indicates the gender of the patient: “Male” or “Female.”",
        "Blood Type": "Patient’s blood type, e.g., “A+”, “O-”.",
        "Medical Condition": "Primary diagnosis, e.g., “Diabetes,” “Hypertension.”",
        "Date of Admission": "Date the patient was admitted to the hospital.",
        "Doctor": "Doctor responsible for the patient during admission.",
        "Hospital": "Name of the healthcare facility/hospital.",
        "Insurance Provider": "Patient’s insurer: Aetna, Blue Cross, Cigna, etc.",
        "Billing Amount": "Total bill amount (float).",
        "Room Number": "Room assigned to the patient.",
        "Admission Type": "Type of admission: “Emergency”, “Elective”, or “Urgent.”",
        "Discharge Date": "Date the patient was discharged.",
        "Medication": "Prescribed or administered medication (e.g., “Aspirin”).",
        "Test Results": "Outcome of a test: “Normal”, “Abnormal”, or “Inconclusive.”"
    }
    for term, desc in glossary.items():
        if st.session_state.glossary_search.lower() in term.lower():
            st.markdown(f"- **{term}**: {desc}")

# 💾 Init State
for key in ["chat_history", "query_log", "fallback_log"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "query_log" else {}

# 🧩 Filters
st.sidebar.markdown("### 🔎 Apply Filters")
hospitals = st.sidebar.multiselect("Filter by Hospital", df["Hospital"].dropna().unique())
conditions = st.sidebar.multiselect("Filter by Condition", df["Medical Condition"].dropna().unique())
filtered_df = df.copy()
if hospitals:
    filtered_df = filtered_df[filtered_df["Hospital"].isin(hospitals)]
if conditions:
    filtered_df = filtered_df[filtered_df["Medical Condition"].isin(conditions)]

# 📈 KPIs
st.markdown("## 📈 Summary KPIs")
col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Billing", f"${filtered_df['Billing Amount'].sum():,.2f}")
col2.metric("🛏️ Avg Stay", f"{filtered_df['Length of Stay'].mean():.1f} days")
col3.metric("👥 Total Patients", f"{filtered_df['Name'].nunique()}")

# 📈 Trend Chart
if "Date of Admission" in filtered_df.columns:
    filtered_df["Date of Admission"] = pd.to_datetime(filtered_df["Date of Admission"])
    trend_data = filtered_df.groupby("Date of Admission")["Billing Amount"].sum().reset_index()
    trend_chart = alt.Chart(trend_data).mark_line(point=True).encode(
        x="Date of Admission:T",
        y="Billing Amount:Q"
    ).properties(title="📉 Billing Trend Over Time")
    st.altair_chart(trend_chart, use_container_width=True)

# 🤖 Respond Function
def respond_to_query(query):
    try:
        agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0), df=filtered_df, verbose=False)
        return agent.run(query)
    except Exception:
        return "🤖 I’m currently unable to answer that question. Try rephrasing or ask about another metric!"

# 💬 Chat Section
st.markdown("### 💬 Chat with Assistant")
for i, (q, a) in enumerate(st.session_state["chat_history"]):
    message(q, is_user=True, key=f"user_{i}")
    message(a, key=f"bot_{i}")

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
        response = respond_to_query(s)
        st.session_state.chat_history.append((s, response))
        st.session_state["query_log"][s] = st.session_state["query_log"].get(s, 0) + 1

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question", placeholder="E.g. Average stay by condition")
    submitted = st.form_submit_button("Send")
    if submitted and user_input:
        response = respond_to_query(user_input)
        st.session_state.chat_history.append((user_input, response))
        st.session_state["query_log"][user_input] = st.session_state["query_log"].get(user_input, 0) + 1
        with st.expander("📋 Copy Response"):
            st.code(response, language="markdown")

# 📊 Advanced Insights (Added)
st.markdown("### 📊 Advanced Insights")
chart_type = st.selectbox("Chart type", ["Bar Chart", "Line Chart", "Pie Chart"])
dimension = st.selectbox("Dimension", ["Gender", "Insurance Provider", "Hospital", "Medical Condition", "Date of Admission"])

if chart_type == "Bar Chart":
    data = filtered_df[dimension].dropna().value_counts().reset_index()
    data.columns = [dimension, "Count"]
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X(f"{dimension}:N", sort="-y"),
        y="Count:Q",
        tooltip=[dimension, "Count"]
    )
    labels = alt.Chart(data).mark_text(
        align="center", baseline="bottom", dy=-5, fontSize=12
    ).encode(
        x=f"{dimension}:N",
        y="Count:Q",
        text="Count:Q"
    )
    st.altair_chart(chart + labels, use_container_width=True)
    export_csv(data, f"bar_chart_{dimension.lower().replace(' ', '_')}")

elif chart_type == "Line Chart" and dimension == "Date of Admission":
    filtered_df["Date of Admission"] = pd.to_datetime(filtered_df["Date of Admission"])
    data = filtered_df.groupby("Date of Admission")["Billing Amount"].mean().reset_index()
    chart = alt.Chart(data).mark_line(point=True).encode(
        x="Date of Admission:T",
        y="Billing Amount:Q"
    ).properties(title="Average Billing Over Time")
    st.altair_chart(chart, use_container_width=True)
    export_csv(data, "line_chart_billing_trend")

elif chart_type == "Pie Chart":
    data = filtered_df[dimension].dropna().value_counts().reset_index()
    data.columns = [dimension, "Count"]
    chart = alt.Chart(data).mark_arc(innerRadius=50).encode(
        theta="Count:Q",
        color=alt.Color(f"{dimension}:N"),
        tooltip=[dimension, "Count"]
    ).properties(title=f"{dimension} Distribution")
    st.altair_chart(chart, use_container_width=True)
    export_csv(data, f"pie_chart_{dimension.lower().replace(' ', '_')}")

# 📖 Narrative Insights
st.markdown("### 📖 Narrative Insights")
if st.button("Generate Narrative Summary"):
    with st.spinner("Generating insights..."):
        try:
            from langchain.prompts import PromptTemplate
            from langchain.llms import OpenAI
            summary_text = filtered_df.describe(include='all').to_string()
            prompt = PromptTemplate.from_template(
                "You are a healthcare analyst. Summarize the following dataset insightfully:

{summary}"
            )
            llm = OpenAI(temperature=0)
            summary = llm(prompt.format(summary=summary_text))
            st.success("🔍 GPT Insight:")
            st.markdown(summary)
        except Exception as e:
            st.warning(f"GPT Summary unavailable: {e}")

# 🏆 Leaderboard
if st.session_state["query_log"]:
    leaderboard_df = pd.DataFrame(
        sorted(st.session_state["query_log"].items(), key=lambda x: x[1], reverse=True),
        columns=["Query", "Clicks"]
    )
    st.markdown("### 🏆 Most Clicked Suggestions")
    st.dataframe(leaderboard_df, use_container_width=True)
    st.download_button("⬇️ Download Query Log (CSV)", data=leaderboard_df.to_csv(index=False), file_name="query_log.csv")

# 📥 Usage Log
if st.session_state["usage_log"]:
    log_df = pd.DataFrame(st.session_state["usage_log"])
    st.download_button("📥 Download Usage Log", log_df.to_csv(index=False), file_name="usage_log.csv")

# 🔗 Navigation + Footer
st.page_link("pages/1_📊_Dashboard.py", label="📊 Dashboard", icon="📊")
st.page_link("pages/4_Dashboard_Feature_Overview.py", label="📘 Dashboard Feature Overview", icon="📘")
st.page_link("pages/3__Chat_Assistant_Feature_Overview.py", label="📄 Chat Assistant Feature Overview", icon="📄")
st.markdown("---")
st.markdown("Made with ❤️ by Deepti Bahel | Powered by Streamlit + LangChain + Altair")
