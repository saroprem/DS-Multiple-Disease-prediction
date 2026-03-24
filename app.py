import streamlit as st
import pickle
import numpy as np

# Helper to load pickle files
def load_pickle(name):
    return pickle.load(open(name, "rb"))

# Load all models & scalers & feature lists
kidney_model = load_pickle("kidney_model.pkl")
kidney_scaler = load_pickle("kidney_scaler.pkl")
kidney_features = load_pickle("kidney_features.pkl")

liver_model = load_pickle("liver_model.pkl")
liver_scaler = load_pickle("liver_scaler.pkl")
liver_features = load_pickle("liver_features.pkl")

parkinson_model = load_pickle("parkinson_model.pkl")
parkinson_scaler = load_pickle("parkinson_scaler.pkl")
parkinson_features = load_pickle("parkinson_features.pkl")

# Page settings
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide")

st.title("🩺 Multiple Disease Prediction System")

# Tabs
tab1, tab2, tab3 = st.tabs(["Kidney Disease", "Liver Disease", "Parkinsons"])

# ------------------ Kidney Disease ------------------
with tab1:
    st.header("Kidney Disease Prediction")
    inputs = {}

    for feat in kidney_features:
        if feat == "classification":
            continue
        inputs[feat] = st.text_input(f"{feat}")

    if st.button("Predict Kidney Disease"):
        try:
            row = []
            for feat in kidney_features:
                if feat == "classification":
                    continue
                val = inputs[feat]
                if val == "":
                    val = 0
                row.append(float(val))

            arr = np.array(row).reshape(1, -1)
            arr_scaled = kidney_scaler.transform(arr)

            pred = kidney_model.predict(arr_scaled)[0]
            proba = kidney_model.predict_proba(arr_scaled)[0].max()

            if pred == 1:
                st.error(f"Result: Chronic Kidney Disease Detected. Confidence: {proba:.2f}")
            else:
                st.success(f"Result: No Kidney Disease. Confidence: {proba:.2f}")

        except Exception as e:
            st.error(f"Error: {e}")

# ------------------ Liver Disease ------------------
with tab2:
    st.header("Liver Disease Prediction")
    inputs = {}

    for feat in liver_features:
        if feat == "Target":
            continue
        inputs[feat] = st.text_input(f"{feat}")

    if st.button("Predict Liver Disease"):
        try:
            row = []
            for feat in liver_features:
                if feat == "Target":
                    continue
                val = inputs[feat]
                if val == "":
                    val = 0
                row.append(float(val))

            arr = np.array(row).reshape(1, -1)
            arr_scaled = liver_scaler.transform(arr)

            pred = liver_model.predict(arr_scaled)[0]
            proba = liver_model.predict_proba(arr_scaled)[0].max()

            if pred == 1:
                st.error(f"Result: Liver Disease Detected. Confidence: {proba:.2f}")
            else:
                st.success(f"Result: No Liver Disease. Confidence: {proba:.2f}")

        except Exception as e:
            st.error(f"Error: {e}")

# ------------------ Parkinson's ------------------
with tab3:
    st.header("Parkinson's Disease Prediction")
    inputs = {}

    for feat in parkinson_features:
        if feat == "status":
            continue
        inputs[feat] = st.text_input(f"{feat}")

    if st.button("Predict Parkinsons"):
        try:
            row = []
            for feat in parkinson_features:
                if feat == "status":
                    continue
                val = inputs[feat]
                if val == "":
                    val = 0
                row.append(float(val))

            arr = np.array(row).reshape(1, -1)
            arr_scaled = parkinson_scaler.transform(arr)

            pred = parkinson_model.predict(arr_scaled)[0]
            proba = parkinson_model.predict_proba(arr_scaled)[0].max()

            if pred == 1:
                st.error(f"Result: Parkinson's Detected ⚠️ | Confidence: {proba:.2f}")
            else:
                st.success(f"Result: No Parkinson's 🙂 | Confidence: {proba:.2f}")

        except Exception as e:
            st.error(f"Error: {e}")
