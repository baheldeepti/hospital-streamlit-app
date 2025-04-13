import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
from prophet import Prophet
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans

st.set_page_config(page_title="Hospital Dashboard", layout="wide")
st.title("🏥 Easy-to-Understand Hospital Data Dashboard")

st.markdown("""
Welcome to the Hospital Executive Insights Dashboard — designed to provide a clear, high-level view of operational performance and key healthcare metrics.  
Each visualization is paired with concise commentary to support strategic decision-making, resource planning, and performance optimization.
""")

# Power BI
st.header("📊 Embedded Power BI Dashboard")
powerbi_url = "https://app.powerbi.com/reportEmbed?reportId=adba6afc-7417-4ef1-a069-a4c9b7287a7b&autoAuth=true&ctid=81a7563b-844e-45bb-a4f0-b2f1ed65acb7"
components.iframe(powerbi_url, width=1000, height=600, scrolling=True)

# Sample CSV
st.subheader("📥 Sample Data to Explore")
st.markdown("""
If you don't have your own data yet, you can use our **sample hospital dataset** to try out the dashboard.  
🔗 [Download Sample CSV](https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv)
""")

file = st.file_uploader("📁 Upload your hospital dataset (.csv)", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    st.info("✅ Custom dataset uploaded successfully!")
else:
    sample_url = "https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv"
    df = pd.read_csv(sample_url)
    st.warning("⚠️ No file uploaded. Using the **default sample dataset** for demo purposes.")

df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
st.session_state.main_df = df

# Billing Forecast
st.header("📈 Monthly Billing Forecast")
billing_df = df[['Date of Admission', 'Billing Amount']].copy().rename(columns={'Date of Admission': 'ds', 'Billing Amount': 'y'})
billing_df = billing_df.groupby(pd.Grouper(key='ds', freq='M')).sum().reset_index()
model = Prophet()
model.fit(billing_df)
forecast = model.predict(model.make_future_dataframe(periods=6, freq='M'))
st.pyplot(model.plot(forecast))

# LOS Forecast
st.header("📉 Average Length of Stay Forecast")
los_df = df[['Date of Admission', 'Length of Stay']].copy().rename(columns={'Date of Admission': 'ds', 'Length of Stay': 'y'})
los_df = los_df.groupby(pd.Grouper(key='ds', freq='M')).mean().reset_index()
los_model = Prophet()
los_model.fit(los_df)
los_forecast = los_model.predict(los_model.make_future_dataframe(periods=6, freq='M'))
st.pyplot(los_model.plot(los_forecast))

# Anomaly Detection
st.header("🚨 Billing Anomaly Detection")
iso = IsolationForest(contamination=0.02, random_state=42)
df['anomaly'] = iso.fit_predict(df[['Billing Amount']])
fig3, ax3 = plt.subplots()
sns.boxplot(x='anomaly', y='Billing Amount', data=df, ax=ax3)
st.pyplot(fig3)

# 🔍 Abnormal Test Results — Model Comparison
st.header("🧠 Predicting Abnormal Test Results (Model Comparison)")
st.markdown("Compare different ML models to predict abnormal test results.")

df['Test Results Encoded'] = df['Test Results'].map({'Normal': 0, 'Abnormal': 1, 'Inconclusive': 2})
X = df[['Age', 'Billing Amount', 'Length of Stay']].copy()
X['Condition'] = LabelEncoder().fit_transform(df['Medical Condition'])
X['Medication'] = LabelEncoder().fit_transform(df['Medication'])
y = (df['Test Results Encoded'] == 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Support Vector Machine": SVC(kernel='linear', probability=True)
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
st.dataframe(results_df.style.format({"Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}"}))
best_model_name = results_df.iloc[0]["Model"]
st.success(f"🎯 Best Model Based on F1 Score: {best_model_name}")

# Clustering
st.header("🧪 Patient Clusters")
cluster_data = df[['Age', 'Billing Amount', 'Length of Stay']]
cluster_data['Condition'] = LabelEncoder().fit_transform(df['Medical Condition'])
scaled = StandardScaler().fit_transform(cluster_data)
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled)
fig4, ax4 = plt.subplots()
sns.scatterplot(x='Billing Amount', y='Length of Stay', hue='Cluster', data=df, palette='Set2', ax=ax4)
st.pyplot(fig4)

st.success("🎉 You're done exploring the hospital data like a pro!")

# Summary
st.markdown("""
---

## 📌 Summary View of the Analysis

- **Billing Forecast**: Predicts steady revenue using historical trends.
- **Length of Stay Forecast**: Shows patient stay duration trends.
- **Anomaly Detection**: Flags unusual billing amounts.
- **Model Comparison**: Evaluates models for predicting abnormal test results.
- **Patient Clusters**: Segments patients by behavior for better insights.
""")
