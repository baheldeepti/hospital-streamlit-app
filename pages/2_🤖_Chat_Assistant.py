# 📘 Hospital Chat Assistant - v1.4.1 (Free-form Query Enhanced)

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

# 🔐 OpenAI Key Setup
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# 📊 Page Setup
st.set_page_config(page_title="\U0001F916 Hospital Chat Assistant", layout="wide")
st.title("🏥 Hospital Chat Assistant")

# 🐞 Debug Mode
DEBUG_MODE = st.sidebar.checkbox("🐞 Enable Debug Mode")
def debug_log(message_text):
    if DEBUG_MODE:
        st.sidebar.markdown(f"🔍 **Debug**: {message_text}")

# 📦 CSV Export Helper
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

# 🧩 Helper: Chart Generator from Query Keywords
def generate_chart_for_query(query):
    q = query.lower()
    if "billing" in q and "hospital" in q:
        data = filtered_df.groupby("Hospital")["Billing Amount"].sum().reset_index()
        chart = alt.Chart(data).mark_bar().encode(
            x="Hospital:N", y="Billing Amount:Q", tooltip=["Hospital", "Billing Amount"]
        ).properties(title="💰 Total Billing by Hospital")
        st.altair_chart(chart, use_container_width=True)
        export_csv(data, "billing_by_hospital")
        return True
    elif "average stay" in q or "length of stay" in q:
        if "Date of Admission" in filtered_df.columns:
            data = filtered_df.groupby("Date of Admission")["Length of Stay"].mean().reset_index()
            chart = alt.Chart(data).mark_line(point=True).encode(
                x="Date of Admission:T", y="Length of Stay:Q"
            ).properties(title="📈 Avg Length of Stay Over Time")
            st.altair_chart(chart, use_container_width=True)
            export_csv(data, "avg_stay_trend")
            return True
    elif "gender" in q and "patients" in q:
        data = filtered_df["Gender"].value_counts().reset_index()
        data.columns = ["Gender", "Count"]
        chart = alt.Chart(data).mark_bar().encode(
            x="Gender:N", y="Count:Q", tooltip=["Gender", "Count"]
        ).properties(title="👥 Patient Count by Gender")
        st.altair_chart(chart, use_container_width=True)
        export_csv(data, "gender_distribution")
        return True
    return False  # fallback to GPT

# 🛑 Data Check
if "main_df" not in st.session_state:
    st.warning("🚨 Please load or upload a dataset to proceed.")
    st.stop()

df = st.session_state["main_df"]
df["Billing Amount"] = pd.to_numeric(df["Billing Amount"].replace('[\$,]', '', regex=True), errors="coerce")
df["Length of Stay"] = pd.to_numeric(df.get("Length of Stay", pd.Series(dtype=float)), errors="coerce")
filtered_df = df.copy()

# 💬 Chat Assistant Logic with Chart Fallback
st.subheader("💬 Chat Assistant")
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def respond_to_query(query):
    if generate_chart_for_query(query):
        return "📊 Chart generated based on your question!"
    try:
        agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0), df=filtered_df, verbose=False)
        return agent.run(query)
    except Exception as e:
        st.session_state["fallback_log"] = st.session_state.get("fallback_log", [])
        st.session_state["fallback_log"].append(query)
        return "\U0001F916 I couldn’t understand that. Try a question like: 'Total billing by hospital' or 'Average stay over time'."

# UI for chat history and input
for i, (q, a) in enumerate(st.session_state["chat_history"]):
    message(q, is_user=True, key=f"user_{i}")
    message(a, key=f"bot_{i}")

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question", placeholder="e.g. Total billing by hospital")
    submitted = st.form_submit_button("Send")
    if submitted and user_input:
        response = respond_to_query(user_input)
        st.session_state["chat_history"].append((user_input, response))
        st.expander("📋 Copy Response").code(response)

# 📖 Narrative Insights
st.subheader("📖 Narrative Insights")
st.markdown("Click the button to generate an AI-powered summary of the filtered dataset.")
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
            summary_text = filtered_df.describe(include='all').to_string()
            llm = OpenAI(temperature=0)
            summary = llm(prompt.format(summary=summary_text))
            st.success("🔍 Insightful Summary")
            st.markdown(summary)
        except Exception as e:
            st.error(f"Summary failed: {e}")

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

st.markdown("Pick a chart type and data column to visualize.")

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
    st.info("💡 Tip: CSV download available below the chart.")
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

# 🏆 Leaderboard
st.subheader("🏆 Leaderboard")
st.markdown("""
This section shows the **most popular questions** asked so far in this session!  
Each time you click a suggested question or enter your own, it gets tracked here. 📊  

- **Query** = the question asked  
- **Clicks** = how many times it was asked  

Use this leaderboard to see which questions are trending or most useful. 🥇  
You can also **download the log** for future reference!
""")
st.markdown("### Query Log")
if st.session_state["query_log"]:
    leaderboard = pd.DataFrame(sorted(st.session_state["query_log"].items(), key=lambda x: x[1], reverse=True), columns=["Query", "Count"])
    st.dataframe(leaderboard)
    export_csv(leaderboard, "query_leaderboard")

# 📥 Logs
st.markdown("### Usage Log")
if st.session_state["usage_log"]:
    log_df = pd.DataFrame(st.session_state["usage_log"])
    export_csv(log_df, "usage_log")
st.markdown("### Fallback Log")
if st.session_state["fallback_log"]:
    fallback_df = pd.DataFrame(st.session_state["fallback_log"], columns=["Unanswered Queries"])
    export_csv(fallback_df, "fallback_log")

# 📘 Glossary
with st.sidebar.expander("🔍 Data Glossary"):
    st.text_input("Search glossary", key="glossary_search")
    glossary = {
        "Name": "Patient’s name",
        "Age": "Patient age at admission",
        "Gender": "Male/Female",
        "Medical Condition": "Patient diagnosis",
        "Date of Admission": "Admission date",
        "Doctor": "Attending physician",
        "Hospital": "Facility name",
        "Insurance Provider": "Insurer",
        "Billing Amount": "Total bill",
        "Discharge Date": "Discharge date",
        "Room Number": "Patient room",
        "Admission Type": "Emergency/Urgent/Elective",
        "Blood Type": "Blood group",
        "Medication": "Prescribed drug",
        "Test Results": "Normal/Abnormal"
    }
    search = st.session_state.get("glossary_search", "").lower()
    for term, desc in glossary.items():
        if search in term.lower():
            st.markdown(f"- **{term}**: {desc}")

# 🐞 Debug Snapshot
if DEBUG_MODE:
    st.markdown("### 🧪 Debug Snapshot")
    st.write("Chat History", st.session_state.get("chat_history"))
    st.write("Query Log", st.session_state.get("query_log"))
    st.write("Fallback Log", st.session_state.get("fallback_log"))

# 🔗 Navigation + Footer
st.page_link("pages/1_📊_Dashboard.py", label="📊 Dashboard", icon="📊")
st.page_link("pages/4_Dashboard_Feature_Overview.py", label="📘 Dashboard Feature Overview", icon="📘")
st.page_link("pages/3__Chat_Assistant_Feature_Overview.py", label="📄 Chat Assistant Feature Overview", icon="📄")
st.markdown("---")
st.markdown("Made with ❤️ by Deepti Bahel | Powered by Streamlit + LangChain + Altair")
