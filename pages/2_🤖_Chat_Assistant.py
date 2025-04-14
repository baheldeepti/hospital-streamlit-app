# 📘 Hospital Chat Assistant - v1.4.3 DEPLOYMENT READY (MODULAR + CHAT FIXED)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import openai
from datetime import datetime
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 🔐 OPENAI API
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# 🔧 CONFIG
st.set_page_config(page_title="🤖 Hospital Chat Assistant", layout="wide")

# 📁 Load Data Section
st.markdown("## 📂 Load Your Hospital Data")
st.info("""
Welcome to the **Hospital Chat Assistant**!

To get started, please **upload your hospital CSV file** or click the **Load Sample Dataset** button below to try it out with example data.

- The data should include common hospital columns like: `Hospital`, `Patient Name`, `Billing Amount`, `Length of Stay`, `Medical Condition`, etc.
- Once loaded, you can explore filters, ask questions, view KPIs, and generate insights.
""")

col1, col2 = st.columns([1, 2])

with col1:
    load_sample = st.button("📥 Load Sample Dataset")

with col2:
    uploaded_file = st.file_uploader("Or upload your own CSV file:", type=["csv"])

# 🧠 Handle Data Load
if load_sample:
    sample_url = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"
    df = pd.read_csv(sample_url)
    st.session_state["main_df"] = df
    st.success("✅ Sample dataset loaded successfully!")

elif uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state["main_df"] = df
        st.success("✅ Your dataset was uploaded and loaded!")
    except Exception as e:
        st.error(f"🚨 Failed to load file: {e}")

# 🚫 Stop if no data is loaded
if "main_df" not in st.session_state:
    st.warning("⚠️ Please load sample data or upload your own CSV file to begin.")
    st.stop()

st.title("🏥 Hospital Chat Assistant")

# 📊 DEBUG MODE
DEBUG_MODE = st.sidebar.checkbox("🐞 Enable Debug Mode")
def debug_log(msg):
    if DEBUG_MODE:
        st.sidebar.markdown(f"🔍 **Debug**: {msg}")

# 📁 FILE UPLOAD & SAMPLE DATA
@st.cache_data
def load_sample_data():
    url = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"
    return pd.read_csv(url)

def load_data():
    with st.sidebar.expander("📁 Load or Upload Dataset", expanded=True):
        if st.button("Load Sample Dataset"):
            df = load_sample_data()
            st.session_state["main_df"] = df
            st.success("✅ Sample dataset loaded.")
        uploaded = st.file_uploader("Upload your CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state["main_df"] = df
            st.success("✅ File uploaded successfully.")

# 🧮 FILTERS
def render_filters(df):
    st.sidebar.markdown("### 🔎 Filter Data")
    hospitals = st.sidebar.multiselect("Hospital", df["Hospital"].dropna().unique())
    insurances = st.sidebar.multiselect("Insurance Provider", df["Insurance Provider"].dropna().unique())
    conditions = st.sidebar.multiselect("Medical Condition", df["Medical Condition"].dropna().unique())
    if hospitals:
        df = df[df["Hospital"].isin(hospitals)]
    if insurances:
        df = df[df["Insurance Provider"].isin(insurances)]
    if conditions:
        df = df[df["Medical Condition"].isin(conditions)]
    return df

# 📈 KPIs
def render_kpis(df):
    st.subheader("📈 Summary KPIs")
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Total Billing", f"${df['Billing Amount'].sum():,.2f}")
    c2.metric("🛏️ Avg Stay", f"{df['Length of Stay'].mean():.1f} days")
    c3.metric("👥 Total Patients", f"{df['Name'].nunique()}")

# 📉 TREND CHART
def render_trend_chart(df):
    st.subheader("📉 Billing Trend Over Time")
    try:
        df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
        trend = df.groupby("Date of Admission")["Billing Amount"].sum().reset_index()
        chart = alt.Chart(trend).mark_line(point=True).encode(
            x="Date of Admission:T", y="Billing Amount:Q"
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        debug_log(f"Trend chart error: {e}")

# 💬 CHAT ASSISTANT
FALLBACK_RESPONSE = """🤖 I’m not able to understand that question right now.

**Try asking something like:**
- *Total billing by hospital*
- *Average stay over time*
- *Top conditions by test result*
"""

keyword_chart_map = {
    "total billing by hospital": {"chart_type": "bar", "dimension": "Hospital", "metric": "Billing Amount", "aggregation": "sum"},
    "average stay over time": {"chart_type": "line", "dimension": "Date of Admission", "metric": "Length of Stay", "aggregation": "mean"},
    "patient count by gender": {"chart_type": "bar", "dimension": "Gender", "aggregation": "count"},
    "billing trend": {"chart_type": "line", "dimension": "Date of Admission", "metric": "Billing Amount", "aggregation": "sum"}
}

def match_chart_mapping(query):
    query = query.lower()
    for key in keyword_chart_map:
        if key in query:
            return keyword_chart_map[key]
    return None

def export_csv(df, filename):
    csv = df.to_csv(index=False).encode()
    st.download_button("📩 Download CSV", csv, file_name=f"{filename}.csv", mime="text/csv")

def respond_to_query(query, df):
    config = match_chart_mapping(query)
    if config:
        try:
            chart_type = config["chart_type"]
            dimension = config["dimension"]
            metric = config.get("metric")
            aggregation = config.get("aggregation", "count")
            chart_title = f"{aggregation.capitalize()} {metric or 'records'} by {dimension}"

            if chart_type == "bar":
                data = df[dimension].value_counts().reset_index() if aggregation == "count" else df.groupby(dimension)[metric].agg(aggregation).reset_index()
                col_name = data.columns[1]
                chart = alt.Chart(data).mark_bar().encode(
                    x=alt.X(f"{dimension}:N", sort="-y"),
                    y=f"{col_name}:Q",
                    tooltip=[dimension, col_name]
                ).properties(title=chart_title)
                st.altair_chart(chart, use_container_width=True)
                export_csv(data, f"{dimension.lower()}_bar")
                return f"📊 Chart: {chart_title}"

            elif chart_type == "line":
                data = df.groupby(dimension)[metric].agg(aggregation).reset_index()
                chart = alt.Chart(data).mark_line(point=True).encode(
                    x=alt.X(f"{dimension}:T" if "Date" in dimension else f"{dimension}:N"),
                    y=f"{metric}:Q",
                    tooltip=[dimension, metric]
                ).properties(title=chart_title)
                st.altair_chart(chart, use_container_width=True)
                export_csv(data, f"{dimension.lower()}_line")
                return f"📊 Chart: {chart_title}"
        except Exception as e:
            debug_log(f"Chart render error: {e}")
            return "⚠️ Chart generation failed."

    # fallback to GPT
    try:
        agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0), df=df, verbose=False)
        return agent.run(query)
    except Exception as e:
        debug_log(f"LangChain agent error: {e}")
        st.session_state["fallback_log"].append(query)
        return FALLBACK_RESPONSE

def chat_ui(df):
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
        "Total billing by insurance provider",
        "Average stay over time"
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

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question", placeholder="e.g. Total billing by hospital")
        submitted = st.form_submit_button("Send")
        if submitted and user_input:
            response = respond_to_query(user_input, df)
            st.session_state["chat_history"].append((user_input, response))
            st.session_state["query_log"][user_input] = st.session_state["query_log"].get(user_input, 0) + 1
            st.expander("📋 Copy Response").code(response)

# 🧠 NARRATIVE INSIGHTS
def render_narrative(df):
    st.subheader("📖 Narrative Insights")
    if st.button("🧠 Generate Summary"):
        with st.spinner("Generating insights..."):
            try:
                from langchain.prompts import PromptTemplate
                from langchain.llms import OpenAI
                summary_text = df.describe(include='all').to_string()
                prompt = PromptTemplate.from_template("""
                You are a healthcare data expert. Based on the following summary, give 3 key insights in simple terms.

                Dataset Summary:
                {summary}
                """)
                llm = OpenAI(temperature=0)
                summary = llm(prompt.format(summary=summary_text))
                st.markdown(summary)
            except Exception as e:
                st.error(f"❌ Summary failed: {e}")

# 📊 ADVANCED CHARTING
def render_advanced_insights(df):
    st.subheader("📊 Advanced Insights")
    chart_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Pie Chart"])
    dimension = st.selectbox("Dimension", ["Gender", "Hospital", "Medical Condition", "Insurance Provider", "Date of Admission"])
    if chart_type == "Bar Chart":
        data = df[dimension].value_counts().reset_index()
        data.columns = [dimension, "Count"]
        chart = alt.Chart(data).mark_bar().encode(x=f"{dimension}:N", y="Count:Q", tooltip=[dimension, "Count"])
        st.altair_chart(chart, use_container_width=True)
        export_csv(data, f"{dimension.lower()}_bar_chart")
    elif chart_type == "Line Chart" and dimension == "Date of Admission":
        df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
        line = df.groupby("Date of Admission")["Billing Amount"].mean().reset_index()
        chart = alt.Chart(line).mark_line(point=True).encode(x="Date of Admission:T", y="Billing Amount:Q")
        st.altair_chart(chart, use_container_width=True)
        export_csv(line, "billing_trend")
    elif chart_type == "Pie Chart":
        pie = df[dimension].value_counts().reset_index()
        pie.columns = [dimension, "Count"]
        chart = alt.Chart(pie).mark_arc(innerRadius=50).encode(theta="Count:Q", color=f"{dimension}:N", tooltip=[dimension, "Count"])
        st.altair_chart(chart, use_container_width=True)
        export_csv(pie, "pie_chart")

# 📋 GLOSSARY
def render_glossary():
    with st.sidebar.expander("🔍 Data Glossary"):
        glossary = {
            "Name": "Patient name", "Age": "Age at admission", "Gender": "Male/Female",
            "Medical Condition": "Diagnosis", "Date of Admission": "When patient admitted",
            "Hospital": "Facility name", "Insurance Provider": "Insurer",
            "Billing Amount": "Total charges", "Discharge Date": "When discharged"
        }
        search = st.text_input("Search glossary", key="glossary_search").lower()
        matches = [f"- **{k}**: {v}" for k, v in glossary.items() if search in k.lower()]
        st.markdown("\n".join(matches) if matches else "🔍 No match found.")

# 🏁 MAIN RUN LOGIC
def main():
    #load_data()
    if "main_df" not in st.session_state:
        st.warning("🚨 Please load or upload a dataset to proceed.")
        return
    df = st.session_state["main_df"]
    df["Billing Amount"] = pd.to_numeric(df["Billing Amount"].replace('[\$,]', '', regex=True), errors="coerce")
    df["Length of Stay"] = pd.to_numeric(df.get("Length of Stay", pd.Series(dtype=float)), errors="coerce")
    filtered_df = render_filters(df)

    render_kpis(filtered_df)
    render_trend_chart(filtered_df)
    chat_ui(filtered_df)
    render_narrative(filtered_df)
    render_advanced_insights(filtered_df)
    render_glossary()

    st.divider()
    st.markdown("Made with ❤️ by Deepti Bahel | Powered by Streamlit + LangChain + Altair")

if __name__ == "__main__":
    main()
