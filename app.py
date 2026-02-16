import streamlit as st
import numpy as np
import pickle

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Heart Health Intelligence",
    page_icon="🫀",
    layout="wide"
)

# ------------------------------------------------------------
# CUSTOM PREMIUM CSS
# ------------------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

.header {
    text-align: center;
    padding: 20px 0;
}

.title {
    font-size: 42px;
    font-weight: 700;
    color: white;
}

.subtitle {
    font-size: 18px;
    color: #94a3b8;
}

.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 35px rgba(0,0,0,0.4);
}

.predict-btn button {
    width: 100%;
    height: 55px;
    font-size: 20px;
    font-weight: bold;
    border-radius: 14px;
    background: linear-gradient(90deg, #ef4444, #dc2626);
    color: white;
    border: none;
}

.result-card {
    margin-top: 35px;
    padding: 35px;
    border-radius: 18px;
    text-align: center;
    font-size: 26px;
    font-weight: 600;
}

.footer {
    text-align: center;
    margin-top: 60px;
    font-size: 14px;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# LOAD MODEL & SCALER
# ------------------------------------------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("""
<div class="header">
    <div class="title">🫀 Heart Health Intelligence</div>
    <div class="subtitle">KNN Regression Model | Heart Severity Prediction</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------
# INPUT SECTION
# ------------------------------------------------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120, 45)

        sex_label = st.selectbox("Sex", ["Male", "Female"])
        sex = 1 if sex_label == "Male" else 0

        cp_label = st.selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina",
             "Non-Anginal Pain", "Asymptomatic"]
        )
        cp_map = {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-Anginal Pain": 2,
            "Asymptomatic": 3
        }
        cp = cp_map[cp_label]

        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)

    with col2:
        chol = st.number_input("Cholesterol", 100, 600, 200)

        fbs_label = st.selectbox(
            "Fasting Blood Sugar",
            ["Normal (≤120 mg/dl)", "High (>120 mg/dl)"]
        )
        fbs = 1 if "High" in fbs_label else 0

        restecg_label = st.selectbox(
            "Resting ECG Result",
            ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"]
        )
        restecg_map = {
            "Normal": 0,
            "ST-T Abnormality": 1,
            "Left Ventricular Hypertrophy": 2
        }
        restecg = restecg_map[restecg_label]

        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)

    with col3:
        exang_label = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        exang = 1 if exang_label == "Yes" else 0

        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)

        slope_label = st.selectbox(
            "Slope of ST Segment",
            ["Upsloping", "Flat", "Downsloping"]
        )
        slope_map = {
            "Upsloping": 0,
            "Flat": 1,
            "Downsloping": 2
        }
        slope = slope_map[slope_label]

        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])

        thal_label = st.selectbox(
            "Thalassemia Type",
            ["Normal", "Fixed Defect", "Reversible Defect"]
        )
        thal_map = {
            "Normal": 1,
            "Fixed Defect": 2,
            "Reversible Defect": 3
        }
        thal = thal_map[thal_label]

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# PREPARE FEATURES
# ------------------------------------------------------------
features = np.array([[age, sex, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak,
                      slope, ca, thal]])

# ------------------------------------------------------------
# PREDICT BUTTON
# ------------------------------------------------------------
st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
predict = st.button("🔍 Run AI Prediction")
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# PREDICTION OUTPUT
# ------------------------------------------------------------
# ------------------------------------------------------------
# PREDICTION OUTPUT
# ------------------------------------------------------------
if predict:
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    severity_score = round(prediction[0], 2)

    # Convert score into user-friendly risk level
    if severity_score < 0.75:
        color = "#065f46"
        title = "🟢 Low Risk"
        description = "No significant signs of heart disease detected."
    elif severity_score < 1.75:
        color = "#92400e"
        title = "🟡 Moderate Risk"
        description = "Some risk indicators present. Medical consultation recommended."
    else:
        color = "#7f1d1d"
        title = "🔴 High Risk"
        description = "High probability of heart disease. Immediate medical attention advised."

    st.markdown(f"""
    <div class="result-card" style="background:{color};">
        <div style="font-size:32px;">{title}</div>
        <br>
        <div style="font-size:18px; font-weight:400;">
            {description}
        </div>
        <br><br>
        <div style="font-size:14px; opacity:0.7;">
            Model Score: {severity_score}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("""
<div class="footer">
Built with ❤️ by Akshit Gajera | Machine Learning Portfolio
</div>
""", unsafe_allow_html=True)
