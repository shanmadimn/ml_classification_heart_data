import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease ML Models", layout="wide")

st.title("Heart Disease Classification — ML Models App")

# -------------------------
# Load Models and Scaler
# -------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic.pkl"),
        "Decision Tree": joblib.load("model/dt.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/nb.pkl"),
        "Random Forest": joblib.load("model/rf.pkl"),
        "XGBoost": joblib.load("model/xgb.pkl")
    }
    scaler = joblib.load("model/scaler.pkl")
    return models, scaler

models, scaler = load_models()

# -------------------------
# Upload CSV
# -------------------------
uploaded_file = st.file_uploader(
    "Upload TEST dataset CSV (last column must be target)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------
    # Split Features / Target
    # -------------------------
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # numeric safety cleaning
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())

    # scaled version
    X_scaled = scaler.transform(X)

    # -------------------------
    # Model Selection
    # -------------------------
    model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[model_name]

    if model_name in ["Logistic Regression", "KNN", "XGBoost"]:
        X_use = X_scaled
    else:
        X_use = X.values

    # -------------------------
    # Predictions
    # -------------------------
    y_pred = model.predict(X_use)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_use)[:, 1]
    else:
        y_prob = y_pred

    # -------------------------
    # Metrics
    # -------------------------
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    try:
        auc = roc_auc_score(y, y_prob)
    except:
        auc = None

    st.subheader("Evaluation Metrics")

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", f"{acc:.3f}")
    c1.metric("Precision", f"{prec:.3f}")

    c2.metric("Recall", f"{rec:.3f}")
    c2.metric("F1 Score", f"{f1:.3f}")

    if auc is not None:
        c3.metric("AUC", f"{auc:.3f}")
    else:
        c3.metric("AUC", "NA")

    c3.metric("MCC", f"{mcc:.3f}")

    # -------------------------
    # Confusion Matrix
    # -------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # -------------------------
    # Classification Report
    # -------------------------
    st.subheader("Classification Report")

    report = classification_report(y, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # -------------------------
    # Show Predictions
    # -------------------------
    st.subheader("Predictions Preview")

    out = df.copy()
    out["Prediction"] = y_pred
    st.dataframe(out.head())

else:
    st.info("Upload a CSV file to start.")
