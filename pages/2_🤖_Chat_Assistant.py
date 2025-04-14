# 📘 Hospital Chat Assistant - v1.4.3 STREAMLIT DEPLOY

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import openai
import os
from datetime import datetime
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# 🌐 ENV + CONFIG
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key
st.set_page_config(page_title="🤖 Hospital Chat Assistant", layout="wide")
DEBUG_MODE = st.sidebar.checkbox("🐞 Debug Mode")

# 🧠 Globals
FALLBACK_RESPONSE = """🤖 I’m not able to understand that question right now.

**Try asking something like:**
- *Total billing by hospital*
- *Average stay per condition*
- *Top conditions by test result*
"""

# 🛠️ Caching for sample load
@st.cache_data
def load_sample_data():
    url = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"
    return pd.read_csv(url)

# 📁 Upload/Load
def load_data():
    with st.sidebar.expander("📁 Load or Upload Dataset", expanded=True):
        if st.button("Load Sample Dataset"):
            st.session_state["main_df"] = load_sample_data()
            st.success("✅ Sample loaded")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            st.session_state["main_df"] = pd.read_csv(uploaded)
            st.success("✅ File uploaded")

# 🔍 Filters
def render_filters(df):
    st.sidebar.markdown("### 🔎 Filter Data")
    hospitals = st.sidebar.multiselect("Hospital", df["Hospital"].dropna().unique())
    insurance = st.sidebar.multiselect("Insurance Provider", df["Insurance Provider"].dropna().unique())
    conditions = st.sidebar.multiselect("Medical Condition", df["Medical Condition"].dropna().unique())
    if hospitals: df = df[df["Hospital"].isin(hospitals)]
    if insurance: df = df[df["Insurance Provider"].isin(insurance)]
    if conditions: df = df[df["Medical Condition"].isin(conditions)]
    return df

# 📈 KPIs
def render_kpis(df):
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Total Billing", f"${df['Billing Amount'].sum():,.2f}")
    c2.metric("🛏️ Avg Stay", f"{df['Length of Stay'].mean():.1f} days")
    c3.metric("👥 Total Patients", f"{df['Name'].nunique()}")

# 📉 Trend Chart
def render_trend_chart(df):
    if "Date of Admission" in df.columns:
        df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
        trend = df.groupby("Date of Admission")["Billing Amount"].sum().reset_index()
        chart = alt.Chart(trend).mark_line(point=True).encode(
            x="Date of Admission:T", y="Billing Amount:Q"
        ).properties(title="📉 Billing Trend Over Time")
        st.altair_chart(chart, use_container_width=True)

# 📦 Export
def export_csv(df, name):
    csv = df.to_csv(index=False).encode()
    st.download_button("📩 Download CSV", csv, file_name=f"{name}.csv", mime="text/csv")

# 🧠 GPT Chart Matching
keyword_chart_map = {
    "total billing by hospital": {"chart_type": "bar", "dimension": "Hospital", "metric": "Billing Amount", "aggregation": "sum"},
    "average stay over time": {"chart_type": "line", "dimension": "Date of Admission", "metric": "Length of Stay", "aggregation": "mean"},
    "patient count by gender": {"chart_type": "bar", "dimension": "Gender", "metric": None, "aggregation": "count"},
    "billing trend": {"chart_type": "line", "dimension": "Date of Admission", "metric": "Billing Amount", "aggregation": "sum"},
}

def match_chart_mapping(user_query):
    for key, config in keyword_chart_map.items():
        if key in user_query.lower():
            return config
    return None

def respond_to_query(query, df):
    config = match_chart_mapping(query)
    if config:
        try:
            metric = config["metric"]
            dim = config["dimension"]
            agg = config["aggregation"]
            chart_title = f"{agg.capitalize()} {metric or 'count'} by {dim}"
            if config["chart_type"] == "bar":
                if agg == "count":
                    data = df[dim].value_counts().reset_index()
                    data.columns = [dim, "Count"]
                else:
                    data = df.groupby(dim)[metric].agg(agg).reset_index()
                chart = alt.Chart(data).mark_bar().encode(x=f"{dim}:N", y=data.columns[1], tooltip=[dim, data.columns[1]])
                st.altair_chart(chart, use_container_width=True)
                export_csv(data, f"chart_{dim.lower()}")
            elif config["chart_type"] == "line":
                data = df.groupby(dim)[metric].agg(agg).reset_index()
                chart = alt.Chart(data).mark_line(point=True).encode(
                    x=f"{dim}:T" if 'Date' in dim else f"{dim}:N",
                    y=f"{metric}:Q",
                    tooltip=[dim, metric]
                ).properties(title=chart_title)
                st.altair_chart(chart, use_container_width=True)
                export_csv(data, f"chart_{dim.lower()}")
            return f"📊 Chart: {chart_title}"
        except Exception as e:
            return f"⚠️ Error generating chart: {e}"

    try:
        agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0), df=df, verbose=False)
        return agent.run(query)
    except Exception:
        st.session_state["fallback_log"].append(query)
        return FALLBACK_RESPONSE

# 💬 Chat UI
def chat_ui(df):
    st.subheader("💬 Chat Assistant")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
        st.session_state["query_log"] = {}
        st.session_state["fallback_log"] = []
    suggestions = [
        "Total billing by hospital", "Patient count by gender",
        "Average stay over time", "Billing trend"
    ]
    cols = st.columns(len(suggestions))
    for i, s in enumerate(suggestions):
        if cols[i].button(s):
            response = respond_to_query(s, df)
            st.session_state["chat_history"].append((s, response))
            st.session_state["query_log"][s] = st.session_state["query_log"].get(s, 0) + 1

    for i, (q, a) in enumerate(st.session_state["chat_history"]):
        message(q, is_user=True, key=f"user_{i}")
        message(a, key=f"bot_{i}")

    with st.form("chat_input", clear_on_submit=True):
        user_input = st.text_input("Ask a question", placeholder="e.g. Average stay by condition")
        if st.form_submit_button("Send") and user_input:
            response = respond_to_query(user_input, df)
            st.session_state["chat_history"].append((user_input, response))
            st.session_state["query_log"][user_input] = st.session_state["query_log"].get(user_input, 0) + 1
            st.expander("📋 Copy Response").code(response)

# 📖 Narrative Insights
def render_narrative(df):
    st.subheader("📖 Narrative Insights")
    if st.button("Generate Summary"):
        try:
            prompt = PromptTemplate.from_template("""
            You are a senior healthcare analyst. Based on this dataset summary, share 3 insights anyone can understand:
            {summary}
            """)
            summary_text = df.describe(include='all').to_string()
            llm = OpenAI(temperature=0)
            summary = llm(prompt.format(summary=summary_text))
            st.success("🔍 Summary:")
            st.markdown(summary)
        except Exception as e:
            st.error(f"Error generating summary: {e}")

# 📊 Advanced Insights
def render_advanced_insights(df):
    st.subheader("📊 Advanced Insights")
    chart_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Pie Chart"])
    dim = st.selectbox("Dimension", sorted(["Gender", "Hospital", "Medical Condition", "Insurance Provider", "Date of Admission"]))
    if chart_type == "Bar Chart":
        data = df[dim].value_counts().reset_index()
        data.columns = [dim, "Count"]
        chart = alt.Chart(data).mark_bar().encode(x=dim, y="Count", tooltip=[dim, "Count"])
        st.altair_chart(chart, use_container_width=True)
        export_csv(data, "bar_chart")
    elif chart_type == "Line Chart":
        if dim in df.columns:
            df[dim] = pd.to_datetime(df[dim], errors="coerce")
        data = df.groupby(dim)["Billing Amount"].mean().reset_index()
        chart = alt.Chart(data).mark_line(point=True).encode(x=dim, y="Billing Amount", tooltip=[dim, "Billing Amount"])
        st.altair_chart(chart, use_container_width=True)
        export_csv(data, "line_chart")
    elif chart_type == "Pie Chart":
        data = df[dim].value_counts().reset_index()
        data.columns = [dim, "Count"]
        chart = alt.Chart(data).mark_arc(innerRadius=50).encode(theta="Count", color=dim, tooltip=[dim, "Count"])
        st.altair_chart(chart, use_container_width=True)
        export_csv(data, "pie_chart")

# 📘 Glossary
def render_glossary():
    glossary = {
        "Name": "Patient’s name",
        "Age": "Patient age",
        "Gender": "Male/Female",
        "Medical Condition": "Diagnosis",
        "Date of Admission": "Admission date",
        "Hospital": "Facility name",
        "Insurance Provider": "Insurer",
        "Billing Amount": "Total billed",
        "Discharge Date": "Discharge date",
        "Room Number": "Room assigned",
        "Admission Type": "Emergency/Urgent/Elective",
        "Blood Type": "Blood group",
        "Medication": "Prescribed medication",
        "Test Results": "Results of lab tests"
    }
    with st.sidebar.expander("📘 Glossary"):
        search = st.text_input("Search glossary", key="glossary_search").lower()
        matches = [f"- **{k}**: {v}" for k, v in glossary.items() if search in k.lower()]
        st.markdown("\n".join(matches) if matches else "No matches found.")

# 📥 Logs
def render_logs():
    st.subheader("📥 Logs & Leaderboard")
    if st.session_state.get("query_log"):
        leaderboard = pd.DataFrame(sorted(st.session_state["query_log"].items(), key=lambda x: x[1], reverse=True), columns=["Query", "Count"])
        st.dataframe(leaderboard)
        export_csv(leaderboard, "query_leaderboard")
    if st.session_state.get("fallback_log"):
        errors = pd.DataFrame(st.session_state["fallback_log"], columns=["Unanswered"])
        st.dataframe(errors)
        export_csv(errors, "fallback_log")

# 🚀 Main App
def main():
    st.markdown("# 🏥 Hospital Chat Assistant")
    load_data()
    if "main_df" not in st.session_state:
        st.stop()
    df = st.session_state["main_df"]
    df["Billing Amount"] = pd.to_numeric(df["Billing Amount"].replace('[\$,]', '', regex=True), errors="coerce")
    df["Length of Stay"] = pd.to_numeric(df.get("Length of Stay", pd.Series(dtype=float)), errors="coerce")
    df = render_filters(df)
    render_kpis(df)
    render_trend_chart(df)
    chat_ui(df)
    render_narrative(df)
    render_advanced_insights(df)
    render_logs()
    render_glossary()
    st.markdown("---\nMade with ❤️ by Deepti Bahel | Powered by Streamlit + LangChain + Altair")

if __name__ == "__main__":
    main()
