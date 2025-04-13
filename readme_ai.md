# 🏥 Hospital BI Agent

An end-to-end AI-powered analytics assistant that integrates:
- 📊 Forecasting with Prophet via FastAPI
- 📈 Interactive Power BI Dashboard guidance
- 🤖 Natural Language Q&A using LangChain + CrewAI
- 🌐 Streamlit-based user interface with file upload support

---

## 🚀 Features

- 📅 **Billing Forecasting** using historical data with Prophet
- 📋 **Upload hospital data (CSV)** to run custom analysis
- 📊 **Power BI dashboard integration** for visual KPIs
- 💬 **Conversational AI agent** that remembers context
- 🧠 Built with **LangChain**, **FastAPI**, **Streamlit**, and **CrewAI**

---

## 🧱 Project Structure

hospital_ai_agent/ ├── api/ # FastAPI service (forecasting + agent) ├── agents/ # Optional crew agent setup ├── langchain/ # LangChain tools and agent runner ├── streamlit_ui/ # Streamlit frontend app ├── models/ # Model loading utils (optional) ├── data/ # Sample and uploaded datasets ├── requirements.txt # All dependencies └── README.md

yaml
Copy
Edit

---

## ⚙️ Setup Instructions

### 1. Clone the Repo & Install Dependencies

```bash
git clone https://github.com/yourusername/hospital_ai_agent.git
cd hospital_ai_agent
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
2. Run the Backend (FastAPI)
bash
Copy
Edit
uvicorn api.main:app --reload
3. Run the Streamlit App
bash
Copy
Edit
streamlit run streamlit_ui/app.py
💬 Chat Interface
In the Chat with BI Agent tab, you can ask questions like:

“How did the billing change over time?”

“What’s the average LOS for diabetic patients?”

“Run anomaly detection on the recent uploads.”

Chat memory keeps prior queries in session for coherent dialogue.

🧪 Sample Forecast Input
You can test forecasting with:

json
Copy
Edit
{
  "ds": ["2024-01-01", "2024-02-01", "2024-03-01"],
  "y": [45000, 47000, 48000]
}
☁️ Deployment
FastAPI (Backend)
Deploy to Render or Railway

Use start command:

bash
Copy
Edit
uvicorn api.main:app --host 0.0.0.0 --port 8000
Streamlit (Frontend)
Push to GitHub

Go to Streamlit Cloud

Set app entry path as: streamlit_ui/app.py

🙌 Acknowledgments
This project uses:

LangChain

Prophet Forecasting

Streamlit

FastAPI

CrewAI

📬 Questions?
For feedback, issues, or feature requests, feel free to open an issue or reach out!
