# 📘 Hospital Chat Assistant - PR-Ready Version with Logging and Enhancements

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

# 🔐 OpenAI Key Setup
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

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
    st.markdown("""
    **🧠 Hospital Chat Assistant** is a smart dashboard and AI assistant built to help hospitals explore their data effortlessly.

    #### 🛠️ Powered By:
    - **Streamlit** for UI
    - **LangChain + OpenAI** for conversational logic
    - **Altair** for interactive visualizations

    👩‍⚕️ Created for healthcare analysts, data teams, and curious users to make insights more accessible.
    """)

# 📁 Sidebar: Dataset Loader + Sample Link
with st.sidebar.expander("📁 Load or Upload Dataset", expanded=True):
    st.markdown("""
    If you don't have your own data yet, you can use our sample hospital dataset to try out the dashboard.  
    🔗 [**Download Sample CSV**](https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv)
    """)
    if st.button("Load Sample Hospital Data"):
        try:
            sample_url = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"
            df = pd.read_csv(sample_url)
            st.session_state["main_df"] = df
            st.success("✅ Sample dataset loaded.")
            log_event("dataset_loaded", "Sample dataset")
        except Exception as e:
            st.error(f"❌ Could not load sample dataset: {e}")
    uploaded_file = st.file_uploader("Upload your CSV", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["main_df"] = df
            st.success("✅ File uploaded successfully.")
            log_event("dataset_loaded", "User uploaded dataset")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# 🧾 Dataset Required
if "main_df" not in st.session_state:
    st.warning("⚠️ Please upload or load a dataset to begin.")
    st.stop()

df = st.session_state["main_df"]

# 🔧 Data Cleanup
if "Billing Amount" in df.columns:
    df["Billing Amount"] = pd.to_numeric(df["Billing Amount"].replace('[\$,]', '', regex=True), errors="coerce")
if "Length of Stay" in df.columns:
    df["Length of Stay"] = pd.to_numeric(df.get("Length of Stay", pd.Series(dtype=float)), errors="coerce")

# 📚 Data Glossary
with st.sidebar.expander("📚 Data Glossary", expanded=False):
    st.markdown("""
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

# 💬 Chat with Assistant
st.markdown("### 💬 Chat with Assistant")
for i, (q, a) in enumerate(st.session_state.chat_history):
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
        st.session_state["query_log"][s] = st.session_state["query_log"].get(s, 0) + 1

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question", key="chat_input", placeholder="E.g. Average stay by condition")
    submitted = st.form_submit_button("Send")

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
        return "⚠️ Code execution not supported in this environment."

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

if submitted and user_input:
    with st.spinner("🤖 Assistant is thinking..."):
        response = respond_to_query(user_input)
        response = add_tooltip(response, tooltips)
        st.session_state.chat_history.append((user_input, response))
        log_event('chat_query', user_input)

# 📊 Auto Chart Preview
st.info("💡 Once a chart is generated, you can download the underlying data as a CSV using the link below the chart.")
st.markdown("### 📊 Auto Chart Preview")

def export_chart_data(dataframe, filename):
    csv_data = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="{filename}.csv">📩 Download Data (CSV)</a>'
    st.markdown(href, unsafe_allow_html=True)

if "chat_input" in st.session_state and submitted:
    query = st.session_state.chat_input.lower()
    log_event('chart_generated', query)

    if "billing" in query and "hospital" in query:
        chart_data = df.groupby("Hospital")["Billing Amount"].sum().reset_index()
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("Hospital:N", sort="-y"),
            y="Billing Amount:Q"
        )
        labels = alt.Chart(chart_data).mark_text(
            align="center", baseline="bottom", dy=-2, fontSize=12
        ).encode(
            x="Hospital:N",
            y="Billing Amount:Q",
            text=alt.Text("Billing Amount:Q", format=".2f")
        )
        st.altair_chart(chart + labels, use_container_width=True)
        export_chart_data(chart_data, "billing_by_hospital")

    elif "gender" in query:
        chart_data = df["Gender"].value_counts().reset_index()
        chart_data.columns = ["Gender", "Count"]
        chart = alt.Chart(chart_data).mark_bar().encode(
            x="Gender:N",
            y="Count:Q"
        )
        labels = alt.Chart(chart_data).mark_text(
            align="center", baseline="bottom", dy=-2, fontSize=12
        ).encode(
            x="Gender:N",
            y="Count:Q",
            text=alt.Text("Count:Q")
        )
        st.altair_chart(chart + labels, use_container_width=True)
        export_chart_data(chart_data, "patient_count_by_gender")

# 📊 Advanced Insights
st.markdown("### 📊 Advanced Insights")
st.markdown("Use the dropdowns below to generate custom visualizations.")

chart_type = st.selectbox("Choose chart type", ["Bar Chart", "Line Chart", "Pie Chart"])
dimension = st.selectbox("Choose dimension to analyze", sorted(["Gender", "Insurance Provider", "Hospital", "Medical Condition", "Date of Admission"]))

log_event('advanced_chart', f'{chart_type} on {dimension}')

if chart_type == "Line Chart" and dimension == "Date of Admission":
    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
    data = df.groupby("Date of Admission")["Billing Amount"].mean().reset_index()
    chart = alt.Chart(data).mark_line(point=True).encode(
        x="Date of Admission:T",
        y="Billing Amount:Q",
        tooltip=["Date of Admission:T", "Billing Amount:Q"]
    ).properties(title="Average Billing Over Time")
    st.altair_chart(chart, use_container_width=True)
    export_chart_data(data, "line_chart_billing_trend")

elif chart_type == "Bar Chart":
    data = df[dimension].dropna().value_counts().reset_index()
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
    export_chart_data(data, f"bar_chart_{dimension.lower().replace(' ', '_')}")

elif chart_type == "Pie Chart":
    data = df[dimension].dropna().value_counts().reset_index()
    data.columns = [dimension, "Count"]
    pie = alt.Chart(data).mark_arc(innerRadius=50).encode(
        theta="Count:Q",
        color=alt.Color(f"{dimension}:N"),
        tooltip=[dimension, "Count"]
    ).properties(title=f"{dimension} Distribution")
    st.altair_chart(pie, use_container_width=True)
    export_chart_data(data, f"pie_chart_{dimension.lower().replace(' ', '_')}")

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

# 📥 Download Usage Log
if st.session_state["usage_log"]:
    log_df = pd.DataFrame(st.session_state["usage_log"])
    st.download_button("📥 Download Usage Log", log_df.to_csv(index=False), file_name="usage_log.csv")

# 🔗 Page Navigation
st.page_link("pages/1_📊_Dashboard.py", label="📊 Dashboard", icon="📊")
st.page_link("pages/4_Dashboard_Feature_Overview.py", label="📘 Dashboard Feature Overview", icon="📘")
st.page_link("pages/3__Chat_Assistant_Feature_Overview.py", label="📄 Chat Assistant Feature Overview", icon="📄")

# 👣 Footer Branding
st.markdown("---")
st.markdown("Made with ❤️ by Deepti Bahel | Powered by Streamlit + LangChain + Altair")
