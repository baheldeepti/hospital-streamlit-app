# 📘 Hospital Chat Assistant - v1.4.2 DEPLOYMENT READY

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

# -------------------------------------
# 🔐 API & Setup
# -------------------------------------
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
st.set_page_config(page_title="🤖 Hospital Chat Assistant", layout="wide")
st.title("🏥 Hospital Chat Assistant")

# -------------------------------------
# 🐞 Debug Tools
# -------------------------------------
DEBUG_MODE = st.sidebar.checkbox("🐞 Enable Debug Mode")
def debug_log(msg):
    if DEBUG_MODE:
        st.sidebar.markdown(f"🔍 **Debug**: {msg}")

# -------------------------------------
# 📦 Utilities
# -------------------------------------
def export_csv(dataframe, filename):
    csv = dataframe.to_csv(index=False).encode()
    st.download_button("📩 Download CSV", csv, file_name=f"{filename}.csv", mime="text/csv")

def log_event(event_type, detail):
    st.session_state["usage_log"].append({
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "detail": detail
    })
    FALLBACK_RESPONSE = """🤖 I’m not able to understand that question right now."""

# -------------------------------------
# 📁 Data Load
# -------------------------------------
def load_dataset():
    st.sidebar.markdown("Upload your own CSV or try the sample dataset.")
    if st.button("Load Sample Dataset"):
        df = pd.read_csv("https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv")
        st.session_state["main_df"] = df
        st.success("✅ Sample dataset loaded.")
        log_event("dataset_loaded", "sample")
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state["main_df"] = df
        st.success("✅ File uploaded successfully.")
        log_event("dataset_loaded", "user_csv")

# -------------------------------------
# 🔍 Filters
# -------------------------------------
def apply_filters(df):
    hospitals = st.sidebar.multiselect("Hospital", df["Hospital"].dropna().unique())
    insurances = st.sidebar.multiselect("Insurance Provider", df["Insurance Provider"].dropna().unique())
    conditions = st.sidebar.multiselect("Medical Condition", df["Medical Condition"].dropna().unique())
    if hospitals: df = df[df["Hospital"].isin(hospitals)]
    if insurances: df = df[df["Insurance Provider"].isin(insurances)]
    if conditions: df = df[df["Medical Condition"].isin(conditions)]
    return df

# -------------------------------------
# 📈 KPIs
# -------------------------------------
def render_kpi_cards(df):
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Total Billing", f"${df['Billing Amount'].sum():,.2f}")
    c2.metric("🛏️ Avg Stay", f"{df['Length of Stay'].mean():.1f} days")
    c3.metric("👥 Total Patients", f"{df['Name'].nunique()}")

# -------------------------------------
# 📉 Trend Chart
# -------------------------------------
def render_trend_chart(df):
    try:
        df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
        trend = df.groupby("Date of Admission")["Billing Amount"].sum().reset_index()
        chart = alt.Chart(trend).mark_line(point=True).encode(
            x="Date of Admission:T", y="Billing Amount:Q"
        ).properties(title="📉 Billing Trend Over Time")
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        debug_log(f"Trend chart error: {e}")

# -------------------------------------
# 🔍 Glossary
# -------------------------------------
def render_glossary():
    glossary = {
        "Name": "Patient’s name", "Age": "Patient age at admission", "Gender": "Male/Female",
        "Medical Condition": "Patient diagnosis", "Date of Admission": "Admission date",
        "Doctor": "Attending physician", "Hospital": "Facility name", "Insurance Provider": "Insurer",
        "Billing Amount": "Total bill", "Discharge Date": "Discharge date", "Room Number": "Patient room",
        "Admission Type": "Emergency/Urgent/Elective", "Blood Type": "Blood group",
        "Medication": "Prescribed drug", "Test Results": "Normal/Abnormal"
    }
    st.text_input("Search glossary", key="glossary_search")
    search = st.session_state.get("glossary_search", "").lower()
    matches = [f"- **{term}**: {desc}" for term, desc in glossary.items() if search in term.lower()]
    st.markdown("
".join(matches) if matches else "🔍 No match found.")

# -------------------------------------
# 📖 Narrative Insights
# -------------------------------------
def render_narrative_insights(df):
    st.subheader("📖 Narrative Insights")
    if st.button("Generate GPT Summary"):
        with st.spinner("Generating insights..."):
            try:
                from langchain.prompts import PromptTemplate
                from langchain.llms import OpenAI
                prompt = PromptTemplate.from_template("""
                You are a senior healthcare analyst. Carefully review this hospital dataset summary and highlight **3 actionable insights** in clear, plain English that even a non-technical operations team can understand.

                Dataset Summary:
                {summary}
                """)
                summary_text = df.describe(include='all').to_string()
                llm = OpenAI(temperature=0)
                summary = llm(prompt.format(summary=summary_text))
                st.success("🔍 Insightful Summary")
                st.markdown(summary)
            except Exception as e:
                st.error(f"Summary failed: {e}")

# -------------------------------------
# 🔗 Navigation & Footer
# -------------------------------------
def render_footer():
    st.page_link("pages/1_📊_Dashboard.py", label="📊 Dashboard", icon="📊")
    st.page_link("pages/4_Dashboard_Feature_Overview.py", label="📘 Dashboard Feature Overview", icon="📘")
    st.page_link("pages/3__Chat_Assistant_Feature_Overview.py", label="📄 Chat Assistant Feature Overview", icon="📄")
    st.markdown("---")
    st.markdown("Made with ❤️ by Deepti Bahel | Powered by Streamlit + LangChain + Altair")

# -------------------------------------
# 📋 Main App Logic
# -------------------------------------
with st.sidebar.expander("📁 Load or Upload Dataset", expanded=True):
    load_dataset()

if "main_df" not in st.session_state:
    st.warning("🚨 Please load or upload a dataset to proceed.")
    st.stop()

# Init logs
for k in ["usage_log", "chat_history", "query_log", "fallback_log"]:
    if k not in st.session_state:
        st.session_state[k] = [] if k != "query_log" else {}

df = st.session_state["main_df"]
df["Billing Amount"] = pd.to_numeric(df["Billing Amount"].replace('[\$,]', '', regex=True), errors="coerce")
df["Length of Stay"] = pd.to_numeric(df.get("Length of Stay", pd.Series(dtype=float)), errors="coerce")

with st.sidebar.expander("🔍 Data Glossary"):
    render_glossary()

st.sidebar.markdown("### 🔎 Filter Data")
filtered_df = apply_filters(df)

render_kpi_cards(filtered_df)
render_trend_chart(filtered_df)
render_narrative_insights(filtered_df)
render_footer()


# 💬 Chat Assistant Section
def render_chat_assistant(filtered_df):
    st.subheader("💬 Chat Assistant")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "query_log" not in st.session_state:
        st.session_state["query_log"] = {}
    if "fallback_log" not in st.session_state:
        st.session_state["fallback_log"] = []

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
            response = respond_to_query(s, filtered_df)
            st.session_state["chat_history"].append((s, response))
            st.session_state["query_log"][s] = st.session_state["query_log"].get(s, 0) + 1
            generate_chart_for_suggestion(s, filtered_df)

    for i, (q, a) in enumerate(st.session_state["chat_history"]):
        message(q, is_user=True, key=f"user_{i}")
        message(a, key=f"bot_{i}")

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question", placeholder="e.g. Average stay by condition")
        submitted = st.form_submit_button("Send")
        if submitted and user_input:
            response = respond_to_query(user_input, filtered_df)
            st.session_state["chat_history"].append((user_input, response))
            st.session_state["query_log"][user_input] = st.session_state["query_log"].get(user_input, 0) + 1
            st.expander("📋 Copy Response").code(response)


# 📊 Advanced Insights Section
def render_advanced_insights(filtered_df):
    st.subheader("📊 Advanced Insights")
    st.markdown("Use this section to visually explore your hospital data.")

    chart_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Pie Chart"])
    dimension = st.selectbox("Column to Analyze", sorted(["Gender", "Hospital", "Medical Condition", "Insurance Provider", "Date of Admission"]))

    if chart_type == "Bar Chart":
        bar = filtered_df[dimension].value_counts().reset_index()
        bar.columns = [dimension, "Count"]
        chart = alt.Chart(bar).mark_bar().encode(
            x=alt.X(f"{dimension}:N", sort="-y"),
            y="Count:Q",
            tooltip=[dimension, "Count"]
        )
        labels = alt.Chart(bar).mark_text(dy=-5).encode(
            x=f"{dimension}:N", y="Count:Q", text="Count:Q"
        )
        st.altair_chart(chart + labels, use_container_width=True)
        export_csv(bar, f"{dimension.lower()}_bar_chart")

    elif chart_type == "Line Chart" and dimension == "Date of Admission":
        line = filtered_df.groupby("Date of Admission")["Billing Amount"].mean().reset_index()
        chart = alt.Chart(line).mark_line(point=True).encode(
            x="Date of Admission:T", y="Billing Amount:Q"
        )
        st.altair_chart(chart, use_container_width=True)
        export_csv(line, "billing_trend")

    elif chart_type == "Pie Chart":
        pie = filtered_df[dimension].value_counts().reset_index()
        pie.columns = [dimension, "Count"]
        chart = alt.Chart(pie).mark_arc(innerRadius=50).encode(
            theta="Count:Q",
            color=alt.Color(f"{dimension}:N"),
            tooltip=[dimension, "Count"]
        )
        st.altair_chart(chart, use_container_width=True)
        export_csv(pie, "pie_chart")


# 🏆 Leaderboard and Logs Section
def render_leaderboard_logs():
    st.subheader("🏆 Leaderboard")
    st.markdown("Most popular questions asked so far.")
    if st.session_state.get("query_log"):
        leaderboard = pd.DataFrame(
            sorted(st.session_state["query_log"].items(), key=lambda x: x[1], reverse=True),
            columns=["Query", "Count"]
        )
        st.dataframe(leaderboard)
        export_csv(leaderboard, "query_leaderboard")

    st.subheader("📥 Logs")
    if st.session_state.get("usage_log"):
        usage = pd.DataFrame(st.session_state["usage_log"])
        export_csv(usage, "usage_log")
    if st.session_state.get("fallback_log"):
        fallback = pd.DataFrame(st.session_state["fallback_log"], columns=["Unanswered Queries"])
        export_csv(fallback, "fallback_log")
