
# 📘 Hospital Chat Assistant - v1.3.8 READY FOR DEPLOYMENT

import streamlit as st
import pandas as pd
import os
import numpy as np
import altair as alt
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import openai
import time
from datetime import datetime

# 🧠 OpenAI API Setup
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# 📊 Page Setup
st.set_page_config(page_title="🤖 Hospital Chat Assistant", layout="wide")
st.title("🏥 Hospital Chat Assistant")

# 🐞 Debug Mode
DEBUG_MODE = st.sidebar.checkbox("🐞 Enable Debug Mode")
def debug_log(msg):
    if DEBUG_MODE:
        st.sidebar.markdown(f"🔍 **Debug**: {msg}")

# 📦 CSV Export Utility
def export_csv(dataframe, filename):
    csv = dataframe.to_csv(index=False).encode()
    st.download_button("📩 Download CSV", csv, file_name=f"{filename}.csv", mime="text/csv")

# 📋 Usage Log
if "usage_log" not in st.session_state:
    st.session_state["usage_log"] = []
def log_event(event_type, detail):
    st.session_state["usage_log"].append({
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "detail": detail
    })

# ℹ️ Sidebar - About
with st.sidebar.expander("ℹ️ About This App"):
    st.markdown("""
    **Hospital Chat Assistant** is an AI-powered data exploration tool.
    - 🤖 Chat with an AI agent
    - 📊 Filter data, view insights, and trends
    - 🧠 Summarize data using GPT
    - 🔍 Understand terms with glossary
    """)

# 📁 Sidebar - Upload or Sample Dataset
with st.sidebar.expander("📁 Load or Upload Dataset", expanded=True):
    st.markdown("Drag & drop your CSV or use our sample:")
    if st.button("Load Sample Dataset"):
        df = pd.read_csv("https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv")
        st.session_state["main_df"] = df
        st.success("✅ Sample dataset loaded.")
        log_event("dataset_loaded", "Sample")
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state["main_df"] = df
        st.success("✅ File uploaded.")
        log_event("dataset_loaded", "User CSV")

# 🛑 Stop if no data
if "main_df" not in st.session_state:
    st.warning("🚨 Please load or upload a dataset to proceed.")
    st.stop()

df = st.session_state["main_df"]
df["Billing Amount"] = pd.to_numeric(df["Billing Amount"].replace('[\$,]', '', regex=True), errors="coerce")
df["Length of Stay"] = pd.to_numeric(df.get("Length of Stay", pd.Series(dtype=float)), errors="coerce")
df["Billing Formatted"] = df["Billing Amount"].apply(lambda x: f"${x/1000:.1f}K" if pd.notnull(x) else "N/A")

# 📘 Glossary
with st.sidebar.expander("🔍 Data Glossary"):
    st.text_input("Search glossary", key="glossary_search")
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

# 🔎 Filters
st.sidebar.markdown("### 🔎 Filters")
hospitals = st.sidebar.multiselect("Hospital", df["Hospital"].dropna().unique())
insurance = st.sidebar.multiselect("Insurance Provider", df["Insurance Provider"].dropna().unique())
conditions = st.sidebar.multiselect("Condition", df["Medical Condition"].dropna().unique())

filtered_df = df.copy()
if hospitals: filtered_df = filtered_df[filtered_df["Hospital"].isin(hospitals)]
if insurance: filtered_df = filtered_df[filtered_df["Insurance Provider"].isin(insurance)]
if conditions: filtered_df = filtered_df[filtered_df["Medical Condition"].isin(conditions)]

# 💾 Init State
for k in ["chat_history", "query_log", "fallback_log"]:
    if k not in st.session_state:
        st.session_state[k] = [] if k != "query_log" else {}

# 📊 Summary KPIs
st.subheader("📈 Summary KPIs")
k1, k2, k3 = st.columns(3)
k1.metric("💰 Total Billing", f"${filtered_df['Billing Amount'].sum():,.2f}")
k2.metric("🛏️ Avg Stay", f"{filtered_df['Length of Stay'].mean():.1f} days")
k3.metric("👥 Total Patients", f"{filtered_df['Name'].nunique()}")

# 📉 Trend
if "Date of Admission" in filtered_df.columns:
    filtered_df["Date of Admission"] = pd.to_datetime(filtered_df["Date of Admission"])
    trend = filtered_df.groupby("Date of Admission")["Billing Amount"].sum().reset_index()
    chart = alt.Chart(trend).mark_line(point=True).encode(x="Date of Admission:T", y="Billing Amount:Q").properties(title="📉 Billing Trend")
    st.altair_chart(chart, use_container_width=True)

# 🤖 Chat Assistant
st.subheader("💬 Chat with Assistant")
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

def respond_to_query(query):
    try:
        agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0), df=filtered_df, verbose=False)
        return agent.run(query)
    except Exception:
        st.session_state["fallback_log"].append(query)
        return "🤖 Unable to process this question. Try rephrasing."

scols = st.columns(len(suggestions))
for i, s in enumerate(suggestions):
    if scols[i].button(s):
        response = respond_to_query(s)
        st.session_state.chat_history.append((s, response))
        st.session_state["query_log"][s] = st.session_state["query_log"].get(s, 0) + 1

with st.form("chat_form", clear_on_submit=True):
    q = st.text_input("Ask a question", placeholder="E.g. Average stay by condition")
    submitted = st.form_submit_button("Send")
    if submitted and q:
        response = respond_to_query(q)
        st.session_state.chat_history.append((q, response))
        st.session_state["query_log"][q] = st.session_state["query_log"].get(q, 0) + 1
        with st.expander("📋 Copy Response"):
            st.code(response, language="markdown")

# 📖 Narrative Insights
st.subheader("📖 Narrative Insights")
st.markdown("Click the button to generate an AI-powered summary of the filtered dataset.")
if st.button("Generate Narrative Summary"):
    with st.spinner("Analyzing data..."):
        try:
            from langchain.prompts import PromptTemplate
            from langchain.llms import OpenAI
            prompt = PromptTemplate.from_template( """You are a senior healthcare analyst. Based on the following dataset summary, provide 3 key insights in simple language that a hospital operations team can use to make decisions:{summary}""")
            summary_text = filtered_df.describe(include='all').to_string()
            summary = OpenAI(temperature=0)(prompt.format(summary=summary_text))
            st.success("🔍 Summary:")
            st.markdown(summary)
        except Exception as e:
            st.error(f"Failed to generate summary: {e}")

# 📊 Advanced Insights
st.subheader("📊 Advanced Insights")
st.markdown("""
Use this section to **explore your hospital data visually**!

1. **Choose a chart type** – Pick between bar chart, line chart, or pie chart.
2. **Select what to analyze** – Like Gender, Hospital, Medical Condition, etc.
3. **See the chart appear below!** 📈  
4. **Want to save the data?** Click the **Download CSV** button.

This helps you discover **trends, counts, and patterns** in your data – no coding needed! 💡
""")
chart_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Pie Chart"])
dimension = st.selectbox("Dimension", sorted(["Gender", "Insurance Provider", "Hospital", "Medical Condition", "Date of Admission"]))

if chart_type == "Line Chart" and dimension == "Date of Admission":
    trend = filtered_df.groupby("Date of Admission")["Billing Amount"].mean().reset_index()
    chart = alt.Chart(trend).mark_line(point=True).encode(x="Date of Admission:T", y="Billing Amount:Q").properties(title="Avg Billing Over Time")
    export_csv(trend, "line_chart")
    st.altair_chart(chart, use_container_width=True)

elif chart_type == "Bar Chart":
    bar = filtered_df[dimension].value_counts().reset_index()
    bar.columns = [dimension, "Count"]
    chart = alt.Chart(bar).mark_bar().encode(x=alt.X(f"{dimension}:N", sort="-y"), y="Count:Q", tooltip=[dimension, "Count"])
    labels = alt.Chart(bar).mark_text(align="center", baseline="bottom", dy=-5).encode(x=f"{dimension}:N", y="Count:Q", text="Count:Q")
    export_csv(bar, "bar_chart")
    st.altair_chart(chart + labels, use_container_width=True)

elif chart_type == "Pie Chart":
    pie = filtered_df[dimension].value_counts().reset_index()
    pie.columns = [dimension, "Count"]
    chart = alt.Chart(pie).mark_arc(innerRadius=50).encode(theta="Count:Q", color=alt.Color(f"{dimension}:N"), tooltip=[dimension, "Count"])
    export_csv(pie, "pie_chart")
    st.altair_chart(chart, use_container_width=True)

# 🏆 Leaderboard
st.markdown("### 🏆 Leaderboard")
st.markdown("""
This section shows the **most popular questions** asked so far in this session!  
Each time you click a suggested question or enter your own, it gets tracked here. 📊  

- **Query** = the question asked  
- **Clicks** = how many times it was asked  

Use this leaderboard to see which questions are trending or most useful. 🥇  
You can also **download the log** for future reference!
""")

if st.session_state["query_log"]:
    st.subheader("🏆 Most Clicked Suggestions")
    leaderboard = pd.DataFrame(sorted(st.session_state["query_log"].items(), key=lambda x: x[1], reverse=True), columns=["Query", "Clicks"])
    st.dataframe(leaderboard, use_container_width=True)
    export_csv(leaderboard, "query_leaderboard")

# 📥 Logs
if st.session_state["usage_log"]:
    logs = pd.DataFrame(st.session_state["usage_log"])
    export_csv(logs, "usage_log")
if st.session_state["fallback_log"]:
    errors = pd.DataFrame(st.session_state["fallback_log"], columns=["Unanswered Queries"])
    export_csv(errors, "fallback_log")

# Footer
st.page_link("pages/1_📊_Dashboard.py", label="📊 Dashboard", icon="📊")
st.page_link("pages/4_Dashboard_Feature_Overview.py", label="📘 Dashboard Feature Overview", icon="📘")
st.page_link("pages/3__Chat_Assistant_Feature_Overview.py", label="📄 Chat Assistant Feature Overview", icon="📄")
st.markdown("---")
st.markdown("Made with ❤️ by Deepti Bahel | Powered by Streamlit + LangChain + Altair")
