
# 📘 Hospital Chat Assistant - v1.2.0 with Filters, KPIs, and Glossary

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
import time

# 🔐 OpenAI Key Setup
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


# 📦 Utility: Export CSV Download Button
def export_csv(dataframe, filename):
    csv = dataframe.to_csv(index=False).encode()
    st.download_button("📩 Download CSV", csv, file_name=f"{filename}.csv", mime="text/csv")


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
    st.marst.markdown("""
    **🧠 Hospital Chat Assistant** helps hospitals explore data interactively.

    - 🤖 Chat with an AI agent
    - 📊 Create charts from prompts or dropdowns
    - 📋 View KPIs and session summary
    - 🔍 Search glossary for help

    Created by Deepti Bahel
    """)

# 📁 Dataset Upload Section
with st.sidebar.expander("📁 Load or Upload Dataset", expanded=True):
    st.marst.markdown("""
    Try with your own CSV or use a sample dataset:
    🔗 [**Download Sample CSV**](https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv)
    """)
    if st.button("Load Sample Hospital Data"):
        sample_url = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"
        df = pd.read_csv(sample_url)
        st.session_state["main_df"] = df
        st.success("✅ Sample dataset loaded.")
        log_event("dataset_loaded", "Sample")

    uploaded_file = st.file_uploader("Upload your CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["main_df"] = df
        st.success("✅ File uploaded successfully.")
        log_event("dataset_loaded", "User CSV")

# 🧾 Exit if no data
if "main_df" not in st.session_state:
    st.warning("⚠️ Please load or upload a dataset.")
    st.stop()

df = st.session_state["main_df"]

# 🧹 Data Cleaning
if "Billing Amount" in df.columns:
    df["Billing Amount"] = pd.to_numeric(df["Billing Amount"].replace('[\$,]', '', regex=True), errors="coerce")
    df["Billing Formatted"] = df["Billing Amount"].apply(lambda x: f"${{x/1000:.1f}}K" if pd.notnull(x) else "N/A")
if "Length of Stay" in df.columns:
    df["Length of Stay"] = pd.to_numeric(df.get("Length of Stay", pd.Series(dtype=float)), errors="coerce")

# 🔍 Glossary (Bonus Feature)
with st.sidebar.expander("🔍 Data Glossary"):
    st.text_input("Search term", key="glossary_search")
    glossary = {
        "Name": "Patient name",
        "Age": "Patient's age",
        "Gender": "Male/Female",
        "Medical Condition": "Primary diagnosis",
        "Hospital": "Hospital name",
        "Billing Amount": "Total charges",
        "Length of Stay": "Days admitted",
        "Date of Admission": "Admission date",
        "Doctor": "Attending doctor"
    }
    for term, desc in glossary.items():
        if st.session_state.glossary_search.lower() in term.lower():
            st.marst.markdown(f"- **{term}**: {desc}")


# 💾 Ensure session keys are initialized
for key in ["chat_history", "query_log", "fallback_log"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "query_log" else {}


# 🧩 Multi-Selection Filters
st.sidebar.marst.markdown("### 🔎 Apply Filters")
hospitals = st.sidebar.multiselect("Filter by Hospital", df["Hospital"].dropna().unique())
conditions = st.sidebar.multiselect("Filter by Condition", df["Medical Condition"].dropna().unique())

filtered_df = df.copy()
if hospitals:
    filtered_df = filtered_df[filtered_df["Hospital"].isin(hospitals)]
if conditions:
    filtered_df = filtered_df[filtered_df["Medical Condition"].isin(conditions)]

log_event("filters_applied", f"Hospitals: {len(hospitals)}, Conditions: {len(conditions)}")

# 📊 KPI Summary Section
st.marst.markdown("## 📈 Summary KPIs")

col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Billing", f"${filtered_df['Billing Amount'].sum():,.2f}")
col2.metric("🛏️ Avg Stay", f"{filtered_df['Length of Stay'].mean():.1f} days")
col3.metric("👥 Total Patients", f"{filtered_df['Name'].nunique()}")

# 📈 Admissions Trend Chart
if "Date of Admission" in filtered_df.columns:
    filtered_df["Date of Admission"] = pd.to_datetime(filtered_df["Date of Admission"])
    trend_data = filtered_df.groupby("Date of Admission")["Billing Amount"].sum().reset_index()
    trend_chart = alt.Chart(trend_data).mark_line().encode(
        x="Date of Admission:T",
        y="Billing Amount:Q"
    ).properties(title="📉 Billing Trend Over Time")
    st.altair_chart(trend_chart, use_container_width=True)



# 💬 Chat Assistant Section
st.marst.markdown("### 💬 Chat with Assistant")
for i, (q, a) in enumerate(st.session_state.get("chat_history", [])):
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
def respond_to_query(query):
    try:
        agent = create_pandas_dataframe_agent(
            llm=ChatOpenAI(temperature=0),
            df=filtered_df,
            verbose=False
        )
        return agent.run(query)
    except Exception as e:
        st.session_state["fallback_log"].append(query)
        return "🤖 I’m currently unable to answer that question. Try rephrasing or ask about another metric!"



for i, s in enumerate(suggestions):
    if cols[i].button(s):
        st.session_state["chat_input"] = s
        st.session_state["query_log"][s] = st.session_state["query_log"].get(s, 0) + 1
        

st.session_state["last_chat_query"] = s
        response = respond_to_query(s)
        st.session_state.chat_history.append((s, response))
        log_event("chat_query", s)


# 📊 Advanced Insights Section
st.marst.markdown("### 📊 Advanced Insights")
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question", placeholder="E.g. Average stay by condition")
    submitted = st.form_submit_button("Send")
    
    if submitted and user_input:
        st.session_state.last_chat_query = user_input
        typing_box = st.empty()
        typing_box.markdown("🤖 Assistant is typing...")
        time.sleep(1.2)
        response = respond_to_query(user_input)
        typing_box.empty()
        st.session_state.chat_history.append((user_input, response))
        log_event("chat_query", user_input)

        # 📋 Copy to clipboard
        with st.expander("📋 Copy Response"):
            st.code(response, language="markdown")
log_event("chat_query", user_input)

st.markdown("### 📊 Advanced Insights")

chart_type = st.selectbox("Choose chart type", ["Bar Chart", "Line Chart", "Pie Chart"])
dimension = st.selectbox("Choose dimension", sorted(["Gender", "Insurance Provider", "Hospital", "Medical Condition", "Date of Admission"]))

log_event("advanced_chart", f"{chart_type} on {dimension}")

if chart_type == "Line Chart" and dimension == "Date of Admission":
    filtered_df["Date of Admission"] = pd.to_datetime(filtered_df["Date of Admission"])
    data = filtered_df.groupby("Date of Admission")["Billing Amount"].mean().reset_index()
    chart = alt.Chart(data).mark_line(point=True).encode(
        x="Date of Admission:T",
        y="Billing Amount:Q"
    ).properties(title="Average Billing Over Time")
    st.info("💡 Tip: You can download the data for this chart as CSV below.")

    # 📥 Export chart data as CSV
    export_csv(data, f"{dimension.lower().replace(' ', '_')}_data")

    
    # 🧠 Optional Chart Summary
    if st.checkbox("🧠 Summarize this chart using GPT", key=f"summary_{chart_type}_{dimension}"):
        try:
            from langchain.prompts import PromptTemplate
            from langchain.llms import OpenAI

            summary_text = data.describe(include='all').to_string()
            prompt = PromptTemplate.from_template(
                "You are a healthcare analyst. Summarize this chart and dataset insightfully:

{summary}"
            )
            llm = OpenAI(temperature=0)
            summary = llm(prompt.format(summary=summary_text))
            st.success("🔍 GPT Insight:")
            st.markdown(summary)
        except Exception as e:
            st.warning(f"GPT Summary unavailable: {e}")


    # 🧠 Optional Chart Summary
    if st.checkbox("🧠 Summarize this chart using GPT", key=f"summary_{chart_type}_{dimension}"):
        try:
            from langchain.prompts import PromptTemplate
            from langchain.llms import OpenAI

            summary_text = data.describe(include='all').to_string()
            prompt = PromptTemplate.from_template(
                "You are a healthcare analyst. Summarize this chart and dataset insightfully:

{summary}"
            )
            llm = OpenAI(temperature=0)
            summary = llm(prompt.format(summary=summary_text))
            st.success("🔍 GPT Insight:")
            st.markdown(summary)
        except Exception as e:
            st.warning(f"GPT Summary unavailable: {e}")


    # 🧠 Optional Chart Summary
    if st.checkbox("🧠 Summarize this chart using GPT", key=f"summary_{chart_type}_{dimension}"):
        try:
            from langchain.prompts import PromptTemplate
            from langchain.llms import OpenAI

            summary_text = data.describe(include='all').to_string()
            prompt = PromptTemplate.from_template(
                "You are a healthcare analyst. Summarize this chart and dataset insightfully:

{summary}"
            )
            llm = OpenAI(temperature=0)
            summary = llm(prompt.format(summary=summary_text))
            st.success("🔍 GPT Insight:")
            st.markdown(summary)
        except Exception as e:
            st.warning(f"GPT Summary unavailable: {e}")

st.altair_chart(chart, use_container_width=True)

elif chart_type == "Bar Chart":
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
    st.info("💡 Tip: You can download the data for this chart as CSV below.")

    # 📥 Export chart data as CSV
    export_csv(data, f"{dimension.lower().replace(' ', '_')}_data")

    st.altair_chart(chart + labels, use_container_width=True)

elif chart_type == "Pie Chart":
    data = filtered_df[dimension].dropna().value_counts().reset_index()
    data.columns = [dimension, "Count"]
    chart = alt.Chart(data).mark_arc(innerRadius=50).encode(
        theta="Count:Q",
        color=alt.Color(f"{dimension}:N"),
        tooltip=[dimension, "Count"]
    ).properties(title=f"{dimension} Distribution")
    st.info("💡 Tip: You can download the data for this chart as CSV below.")

    # 📥 Export chart data as CSV
    export_csv(data, f"{dimension.lower().replace(' ', '_')}_data")

    st.altair_chart(chart, use_container_width=True)

# 📑 Session Summary
st.marst.markdown("### 🧠 Session Summary")
summary_data = {
    "Total Questions Asked": len(st.session_state.get("chat_history", [])),
    "Most Clicked Suggestion": max(st.session_state["query_log"], key=st.session_state["query_log"].get) if st.session_state["query_log"] else "N/A",
    "Active Filters": f"{len(hospitals)} Hospitals, {len(conditions)} Conditions",
    "Rows in View": len(filtered_df)
}
for k, v in summary_data.items():
    st.marst.markdown(f"- **{k}**: {v}")

# 📥 Download Logs
if st.session_state["usage_log"]:
    log_df = pd.DataFrame(st.session_state["usage_log"])
    st.download_button("📥 Download Usage Log", log_df.to_csv(index=False), file_name="usage_log.csv")

# 🔗 Navigation + Footer
st.page_link("pages/1_📊_Dashboard.py", label="📊 Dashboard", icon="📊")
st.page_link("pages/4_Dashboard_Feature_Overview.py", label="📘 Dashboard Feature Overview", icon="📘")
st.page_link("pages/3__Chat_Assistant_Feature_Overview.py", label="📄 Chat Assistant Feature Overview", icon="📄")
st.marst.markdown("---")
st.marst.markdown("Made with ❤️ by Deepti Bahel | Powered by Streamlit + LangChain + Altair")
