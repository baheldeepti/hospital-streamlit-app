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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans

st.set_page_config(page_title="Hospital Dashboard", layout="wide")
st.title("\U0001F3E5 Easy-to-Understand Hospital Data Dashboard")

st.markdown("""
Welcome to the Hospital Executive Insights Dashboard — designed to provide a clear, high-level view of operational performance and key healthcare metrics.  
Each visualization is paired with concise commentary to support strategic decision-making, resource planning, and performance optimization.
""")

st.header("\U0001F4CA Embedded Power BI Dashboard")
powerbi_url = "https://app.powerbi.com/reportEmbed?reportId=adba6afc-7417-4ef1-a069-a4c9b7287a7b&autoAuth=true&ctid=81a7563b-844e-45bb-a4f0-b2f1ed65acb7"
components.iframe(powerbi_url, width=1000, height=600, scrolling=True)

st.subheader("\U0001F4C5 Sample Data to Explore")
st.markdown("""
If you don't have your own data yet, you can use our **sample hospital dataset** to try out the dashboard.  
\U0001F517 [Download Sample CSV](https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv)
""")

file = st.file_uploader("\U0001F4C1 Upload your hospital dataset (.csv)", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    st.info("\u2705 Custom dataset uploaded successfully!")
else:
    sample_url = "https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv"
    df = pd.read_csv(sample_url)
    st.warning("\u26A0\uFE0F No file uploaded. Using the **default sample dataset** for demo purposes.")

df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
st.session_state.main_df = df

# Billing Forecast
st.header("\U0001F4C8 Monthly Billing Forecast")
billing_df = df[['Date of Admission', 'Billing Amount']].copy().rename(columns={'Date of Admission': 'ds', 'Billing Amount': 'y'})
billing_df = billing_df.groupby(pd.Grouper(key='ds', freq='M')).sum().reset_index()
model = Prophet()
model.fit(billing_df)
forecast = model.predict(model.make_future_dataframe(periods=6, freq='M'))
st.pyplot(model.plot(forecast))

# LOS Forecast
st.header("\U0001F4C9 Average Length of Stay Forecast")
los_df = df[['Date of Admission', 'Length of Stay']].copy().rename(columns={'Date of Admission': 'ds', 'Length of Stay': 'y'})
los_df = los_df.groupby(pd.Grouper(key='ds', freq='M')).mean().reset_index()
los_model = Prophet()
los_model.fit(los_df)
los_forecast = los_model.predict(los_model.make_future_dataframe(periods=6, freq='M'))
st.pyplot(los_model.plot(los_forecast))

# Anomaly Detection
st.header("\U0001F6A8 Billing Anomaly Detection")
iso = IsolationForest(contamination=0.02, random_state=42)
df['anomaly'] = iso.fit_predict(df[['Billing Amount']])
fig3, ax3 = plt.subplots()
sns.boxplot(x='anomaly', y='Billing Amount', data=df, ax=ax3)
st.pyplot(fig3)

# ML Model Comparison
st.header("\U0001F9E0 Predicting Abnormal Test Results (Model Comparison)")
st.markdown("Compare different ML models and visualize key metrics. Upload your test data below to try predictions.")

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
st.dataframe(results_df.style.format("{:.2%}"))

# Metric Visualizations
for metric in ["F1 Score", "Accuracy", "Precision", "Recall"]:
    st.subheader(f"\U0001F4CA {metric} by Model")
    st.bar_chart(results_df.set_index("Model")[metric])

# Optional: Upload test data to predict with best model
st.subheader("\U0001F4DD Predict on New Test Data")
st.markdown("""
**📋 Test File Format Instructions**

Before uploading, ensure your file includes the following columns:

- `Age` (numeric)
- `Billing Amount` (numeric)
- `Length of Stay` (numeric)
- `Medical Condition` (categorical text)
- `Medication` (categorical text)

📝 Sample row:
```
Age,Billing Amount,Length of Stay,Medical Condition,Medication
45,12000,3,Hypertension,Drug A
```

Missing columns or mismatched formats may lead to prediction errors.
""")

test_file = st.file_uploader("Upload test dataset for prediction (.csv)", type=["csv"], key="test")
model_choice = st.selectbox("Select model to use for prediction:", results_df["Model"])

if test_file:
    test_data = pd.read_csv(test_file)
    test_data['Condition'] = LabelEncoder().fit_transform(test_data['Medical Condition'])
    test_data['Medication'] = LabelEncoder().fit_transform(test_data['Medication'])
    X_new = test_data[['Age', 'Billing Amount', 'Length of Stay', 'Condition', 'Medication']]
    X_new_scaled = scaler.transform(X_new)
    selected_model = models[model_choice]
    y_pred_new = selected_model.predict(X_new_scaled)
    test_data['Predicted Abnormal'] = y_pred_new
    st.write("Predictions on uploaded data:")
    st.dataframe(test_data)

# Clustering
st.header("\U0001F9EA Patient Clusters")
cluster_data = df[['Age', 'Billing Amount', 'Length of Stay']]
cluster_data['Condition'] = LabelEncoder().fit_transform(df['Medical Condition'])
scaled = StandardScaler().fit_transform(cluster_data)
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled)
fig4, ax4 = plt.subplots()
sns.scatterplot(x='Billing Amount', y='Length of Stay', hue='Cluster', data=df, palette='Set2', ax=ax4)
st.pyplot(fig4)

st.success("🎉 You're done exploring the hospital data like a pro!")

# Cluster-wise Model Comparison
st.header("📊 Cluster-wise Model Evaluation vs Overall")
st.markdown("Let's check if applying models separately on clusters improves performance compared to using the full dataset.")

cluster_metrics = []
for cluster_id in sorted(df['Cluster'].unique()):
    cluster_subset = df[df['Cluster'] == cluster_id].copy()
    Xc = cluster_subset[['Age', 'Billing Amount', 'Length of Stay']].copy()
    Xc['Condition'] = LabelEncoder().fit_transform(cluster_subset['Medical Condition'])
    Xc['Medication'] = LabelEncoder().fit_transform(cluster_subset['Medication'])
    yc = (cluster_subset['Test Results Encoded'] == 1).astype(int)

    if yc.nunique() > 1:
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(Xc, yc, stratify=yc, test_size=0.2, random_state=42)
        X_train_c_scaled = scaler.fit_transform(X_train_c)
        X_test_c_scaled = scaler.transform(X_test_c)

        for name, model in models.items():
            model.fit(X_train_c_scaled, y_train_c)
            y_pred_c = model.predict(X_test_c_scaled)
            cluster_metrics.append({
                "Cluster": f"Cluster {cluster_id}",
                "Model": name,
                "Accuracy": accuracy_score(y_test_c, y_pred_c),
                "Precision": precision_score(y_test_c, y_pred_c),
                "Recall": recall_score(y_test_c, y_pred_c),
                "F1 Score": f1_score(y_test_c, y_pred_c)
            })

cluster_df = pd.DataFrame(cluster_metrics)
st.dataframe(cluster_df.style.format("{:.2%}"))

# Compare Clustered vs Full
st.markdown("### 📈 Comparison: Overall vs Clustered Performance (Averaged)")

# Statistical Significance Testing
st.subheader("🧪 T-Test Between Overall and Clustered F1 Scores")
from scipy.stats import ttest_ind

comparison_results = []
for model in results_df['Model']:
    f1_overall = results_df[results_df['Model'] == model]['F1 Score'].values
    f1_cluster = cluster_df[cluster_df['Model'] == model]['F1 Score'].values
    if len(f1_cluster) > 1:
        t_stat, p_val = ttest_ind(f1_cluster, f1_overall, equal_var=False)
        comparison_results.append({"Model": model, "T-Statistic": t_stat, "P-Value": p_val})

if comparison_results:
    ttest_df = pd.DataFrame(comparison_results)
    st.dataframe(ttest_df.style.format({"T-Statistic": "{:.3f}", "P-Value": "{:.4f}"}))
    st.markdown("Models with **P < 0.05** indicate statistically significant differences between cluster-wise and overall F1 scores.")

# Confusion Matrices
st.subheader("📊 Per-Cluster Confusion Matrices as Heatmaps")
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
import io
from matplotlib.backends.backend_pdf import PdfPages

pdf_buffer = io.BytesIO()
pdf_pages = PdfPages(pdf_buffer)

for cluster_id in sorted(df['Cluster'].unique()):
    cluster_subset = df[df['Cluster'] == cluster_id].copy()
    Xc = cluster_subset[['Age', 'Billing Amount', 'Length of Stay']].copy()
    Xc['Condition'] = LabelEncoder().fit_transform(cluster_subset['Medical Condition'])
    Xc['Medication'] = LabelEncoder().fit_transform(cluster_subset['Medication'])
    yc = (cluster_subset['Test Results Encoded'] == 1).astype(int)

    if yc.nunique() > 1:
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(Xc, yc, stratify=yc, test_size=0.2, random_state=42)
        X_train_c_scaled = scaler.fit_transform(X_train_c)
        X_test_c_scaled = scaler.transform(X_test_c)

        # User model selection for tuning
        model_option = st.selectbox(
            f"Select model to tune for Cluster {cluster_id}",
            ("Logistic Regression", "Random Forest", "XGBoost", "LightGBM"),
            key=f"model_cluster_{cluster_id}"
        )

        if model_option == "Logistic Regression":
            param_grid = {"C": [0.01, 0.1, 1, 10]}
            grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=3)
        elif model_option == "Random Forest":
            param_grid = {"n_estimators": [50, 100], "max_depth": [None, 5, 10]}
            grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
        elif model_option == "XGBoost":
            from xgboost import XGBClassifier
            param_grid = {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
            grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid, cv=3)
        elif model_option == "LightGBM":
            from lightgbm import LGBMClassifier
            param_grid = {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
            grid = GridSearchCV(LGBMClassifier(), param_grid, cv=3)

        grid.fit(X_train_c_scaled, y_train_c)
        model = grid.best_estimator_

        y_pred_c = model.predict(X_test_c_scaled)
        y_prob_c = model.predict_proba(X_test_c_scaled)[:, 1]

        cm = confusion_matrix(y_test_c, y_pred_c)
        st.markdown(f"#### Cluster {cluster_id} - Logistic Regression (Tuned)")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - Cluster {cluster_id}")
        st.pyplot(fig)
        pdf_pages.savefig(fig)

        # ROC Curve with CI
        from sklearn.utils import resample
        bootstrapped_scores = []
        for i in range(100):
            indices = resample(range(len(y_test_c)), replace=True)
            if len(set(y_test_c.iloc[indices])) < 2:
                continue
            score = roc_auc_score(y_test_c.iloc[indices], y_prob_c[indices])
            bootstrapped_scores.append(score)

        ci_lower = max(0.0, np.percentile(bootstrapped_scores, 2.5))
        ci_upper = min(1.0, np.percentile(bootstrapped_scores, 97.5))

        fpr, tpr, _ = roc_curve(y_test_c, y_prob_c)
        auc = roc_auc_score(y_test_c, y_prob_c)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.2f} (CI: {ci_lower:.2f}–{ci_upper:.2f})")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"ROC Curve - Cluster {cluster_id}")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
        pdf_pages.savefig(fig_roc)

pdf_pages.close()
pdf_buffer.seek(0)

# Export Results
st.subheader("📤 Downloadable Reports")

# Performance Summary
st.subheader("📘 Performance Summary Across All Clusters")
summary_df = combined_df.groupby("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]].mean().reset_index()
st.dataframe(summary_df.style.format("{:.2%}"))

# PDF Export
st.download_button(
    label="Download Confusion Matrices & ROC Curves as PDF",
    data=pdf_buffer,
    file_name="cluster_model_evaluation.pdf",
    mime="application/pdf"
)

# CSV Export
import io
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
    combined_df.to_excel(writer, index=False, sheet_name='Combined Results')
    summary_df.to_excel(writer, index=False, sheet_name='Summary')

    # Per-cluster sheets
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_data = cluster_df[cluster_df['Cluster'] == f"Cluster {cluster_id}"].copy()
        cluster_data.to_excel(writer, index=False, sheet_name=f'Cluster {cluster_id}')

    # Conditional formatting
    workbook  = writer.book
    for sheet_name in ['Summary', 'Combined Results']:
        worksheet = writer.sheets[sheet_name]
        for col_idx, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1 Score'], start=1):
            worksheet.conditional_format(1, col_idx, 100, col_idx, {
                'type': '3_color_scale',
                'min_color': "#F8696B",
                'mid_color': "#FFEB84",
                'max_color': "#63BE7B"
            })

    writer.save()
    excel_buffer.seek(0)

if st.download_button("Download Report as Excel", data=excel_buffer, file_name="model_performance_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"):
    st.success("Excel file ready for download.")

if st.download_button("Download Combined Results as CSV", combined_df.to_csv(index=False).encode('utf-8'), file_name="model_cluster_comparison.csv"):
    st.success("CSV file ready for download.")

overall_avg = results_df.copy()
overall_avg['Cluster'] = 'Overall'
combined_df = pd.concat([cluster_df, overall_avg], ignore_index=True)

for metric in ["F1 Score", "Accuracy", "Precision", "Recall"]:
    st.subheader(f"📉 {metric}: Cluster Avg vs Overall")
    chart_data = combined_df.groupby(['Model', 'Cluster'])[metric].mean().unstack().fillna(0)
    st.bar_chart(chart_data)

# Summary
st.markdown("""
---

## \U0001F4CC Summary View of the Analysis

- **Billing Forecast**: Predicts steady revenue using historical trends.
- **Length of Stay Forecast**: Shows patient stay duration trends.
- **Anomaly Detection**: Flags unusual billing amounts.
- **Model Comparison**: Evaluates models for predicting abnormal test results.
- **Patient Clusters**: Segments patients by behavior for better insights.
""")
