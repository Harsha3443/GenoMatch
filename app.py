import pandas as pd
import numpy as np
import streamlit as st
import joblib
from tensorflow.keras.models import load_model

def apply_styles():
    st.markdown("""
        <style>
        /* MAIN PAGE BACKGROUND */
        [data-testid="stAppViewContainer"] {
            background-color: #002147;
        }

        /* TOP HEADER (MAKE TRANSPARENT) */
        [data-testid="stHeader"] {
            background: transparent;
        }

        /* CENTER CONTENT + PADDING */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        /* TITLES */
        h1, h2, h3, h4 {
            color: #2a7bd6 !important;
            font-family: "Segoe UI", system-ui, sans-serif;
        }

        h1 {
            text-align: center !important;
            font-weight: 800 !important;
        }

        /* FILE UPLOADER CARD */
        [data-testid="stFileUploader-dropzone"] {
            background-color: #ffffff;
            border: 2px dashed #2a7bd6;
            border-radius: 12px;
        }

        /* BUTTONS */
        .stButton > button {
            background-color: #2a7bd6 !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            padding: 8px 18px !important;
            border: none !important;
            font-weight: 600 !important;
        }

        .stButton > button:hover {
            background-color: #1e5ba1 !important;
            color: #ffffff !important;
        }

        /* DATAFRAME WRAPPER */
        .stDataFrame {
            border-radius: 10px !important;
            overflow: hidden !important;
            background-color: #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = load_model("disease_risk_model.keras")
    artifacts = joblib.load("artifacts.pkl")
    return model, artifacts


model, artifacts = load_artifacts()
scaler = artifacts["scaler"]
feature_names = artifacts["feature_names"]
label_encoder = artifacts["label_encoder"]
target_col = artifacts["target_col"]
num_classes = artifacts["num_classes"]

st.set_page_config(page_title="GenoMatch – Disease Risk Predictor", layout="centered")
apply_styles()

st.title("🧬 GenoMatch – Disease Risk Predictor","align:center")
st.write("Upload a CSV file with the same structure as the training dataset to predict disease inheritance risk.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df_input.head())

    if target_col in df_input.columns:
        df_features = df_input.drop(columns=[target_col])
    else:
        df_features = df_input.copy()

    X_enc = pd.get_dummies(df_features, drop_first=True)

    for col in feature_names:
        if col not in X_enc.columns:
            X_enc[col] = 0

    X_enc = X_enc[feature_names]

    X_scaled = scaler.transform(X_enc)

    preds = model.predict(X_scaled)

    st.subheader("Prediction Results")

    if num_classes <= 2:
        probs = preds.flatten()
        percent = probs * 100
        yes_no = np.where(percent >= 50, "YES", "NO")

        result_df = df_input.copy()
        result_df["inherit_risk_percent"] = np.round(percent, 2)
        result_df["inherit_prediction"] = yes_no

        st.dataframe(result_df)

        avg_risk = float(np.mean(percent))
        st.markdown("### Summary")
        if avg_risk >= 50:
            st.success(f"On average, there is a HIGH chance to inherit disease risk ({avg_risk:.2f}%).")
        else:
            st.info(f"On average, there is a LOW chance to inherit disease risk ({avg_risk:.2f}%).")

    else:
        pred_classes = np.argmax(preds, axis=1)
        max_probs = np.max(preds, axis=1) * 100

        if label_encoder is not None:
            labels = label_encoder.inverse_transform(pred_classes)
        else:
            labels = pred_classes

        result_df = df_input.copy()
        result_df["predicted_class"] = labels
        result_df["class_confidence_percent"] = np.round(max_probs, 2)

        st.dataframe(result_df)
        st.warning("Detected multi-class target. YES/NO inheritance prediction is only applied for binary problems.")
else:
    st.info("Please upload a CSV file to see predictions.")
