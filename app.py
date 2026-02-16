import streamlit as st
import numpy as np
import pickle

# ------------------------------------------------
# Page Config
# ------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Detection",
    page_icon="❤️",
    layout="wide"
)

# ------------------------------------------------
# Load Model & Scaler (IMPORTANT)
# ------------------------------------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ------------------------------------------------
# Title
# ------------------------------------------------
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>❤️ Heart Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("---")

st.write("### Enter Patient Details")

# ------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------
st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", 1, 120, 45)

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])

trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)

chol = st.sidebar.number_input("Cholesterol", 100, 600, 200)

fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

restecg = st.sidebar.selectbox("Resting ECG (0-2)", [0, 1, 2])

thalach = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)

exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])

oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 10.0, 1.0)

slope = st.sidebar.selectbox("Slope (0-2)", [0, 1, 2])

ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])

thal = st.sidebar.selectbox("Thal (0-3)", [0, 1, 2, 3])

# ------------------------------------------------
# Prediction
# ------------------------------------------------
if st.sidebar.button("Predict"):

    # Feature order must EXACTLY match training
    features = np.array([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak,
                          slope, ca, thal]])

    # Only transform (DO NOT fit)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ Heart Disease Not Detected")
