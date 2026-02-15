import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Assignment 2", layout="centered")
st.title("ML Assignment 2 - Classification App")

FEATURE_COLUMNS = [
    'age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
    'cholestoral', 'fasting_blood_sugar', 'rest_ecg', 'Max_heart_rate',
    'exercise_induced_angina', 'oldpeak', 'slope',
    'vessels_colored_by_flourosopy', 'thalassemia', 'target'
]

template_df = pd.DataFrame(columns=FEATURE_COLUMNS)

st.download_button(
    label="⬇️ Download Test CSV Template",
    data=template_df.to_csv(index=False).encode("utf-8"),
    file_name="test_data_template.csv",
    mime="text/csv",
)

models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl"),
}

uploaded_file = st.file_uploader("Upload CSV file (test data only)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", df.head())

    target_col = st.selectbox("Select target column", df.columns)

    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        model_name = st.selectbox("Choose Model", list(models.keys()))
        model = models[model_name]

        y_pred = model.predict(X)

        st.subheader("Evaluation Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "MCC"],
            "Value": [
                accuracy_score(y, y_pred),
                precision_score(y, y_pred, average="weighted"),
                recall_score(y, y_pred, average="weighted"),
                f1_score(y, y_pred, average="weighted"),
                matthews_corrcoef(y, y_pred),
            ]
        })
        st.table(metrics_df)

        try:
            y_proba = model.predict_proba(X)[:, 1]
            auc_val = roc_auc_score(y, y_proba)
            st.write("AUC:", auc_val)
        except Exception:
            st.write("AUC: Not available for this model")

        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
else:
    st.info("Please upload a test CSV to see predictions and metrics.")
