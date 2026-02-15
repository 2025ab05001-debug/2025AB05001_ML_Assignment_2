import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("ML Assignment 2 - Classification App")

# Load models
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
    X = df.drop(columns=[target_col])
    y = df[target_col]

    model_name = st.selectbox("Choose Model", list(models.keys()))
    model = models[model_name]

    y_pred = model.predict(X)

    st.subheader("Evaluation Metrics")
    st.write("Accuracy:", accuracy_score(y, y_pred))
    st.write("Precision:", precision_score(y, y_pred, average="weighted"))
    st.write("Recall:", recall_score(y, y_pred, average="weighted"))
    st.write("F1 Score:", f1_score(y, y_pred, average="weighted"))
    st.write("MCC:", matthews_corrcoef(y, y_pred))

    try:
        y_proba = model.predict_proba(X)[:, 1]
        st.write("AUC:", roc_auc_score(y, y_proba))
    except:
        st.write("AUC: Not available for this model")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
