# ============================================================
# 🫀 Heart Health Intelligence Platform (Enterprise Edition)
# Developed by Akshit Gajera
# ============================================================

import streamlit as st
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Heart Health Intelligence",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# LOAD MODEL & SCALER
# ------------------------------------------------------------
@st.cache_resource
def load_objects():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_objects()

# ------------------------------------------------------------
# PREMIUM DARK MEDICAL UI
# ------------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f172a,#1e293b);
    font-family: 'Segoe UI', sans-serif;
}
.main-header {
    font-size: 2.6rem;
    font-weight: 800;
    text-align: center;
}
.sub-header {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 2rem;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    margin-bottom: 25px;
}
.kpi {
    font-size: 60px;
    font-weight: bold;
    text-align: center;
}
.stButton>button {
    width: 100%;
    height: 55px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    background: linear-gradient(90deg,#ef4444,#dc2626);
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🫀 Project Overview")
    st.info("""
    Machine Learning-Based Heart Severity Prediction  
    
    Algorithm: KNN Regression  
    Features: 13  
    Scaling: StandardScaler  
    """)

    st.markdown("---")
    st.metric("Model Type", "KNN")
    st.metric("Prediction", "Severity Score")
    st.metric("Deployment", "Streamlit")

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown('<div class="main-header">🫀 Heart Health Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Cardiovascular Risk Assessment</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Prediction",
    "Analytics",
    "Model Insights",
    "Health Report"
])

# ============================================================
# TAB 1 – PREDICTION
# ============================================================
with tab1:

    st.markdown("### 🧬 Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120, 45)
        sex = 1 if st.selectbox("Sex", ["Male", "Female"]) == "Male" else 0
        cp = st.selectbox("Chest Pain Type", [0,1,2,3])
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)

    with col2:
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar >120", [0,1])
        restecg = st.selectbox("Resting ECG", [0,1,2])
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)

    with col3:
        exang = st.selectbox("Exercise Induced Angina", [0,1])
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
        slope = st.selectbox("Slope", [0,1,2])
        ca = st.selectbox("Major Vessels", [0,1,2,3])
        thal = st.selectbox("Thalassemia", [1,2,3])

    st.markdown("---")

    predict = st.button("🔍 Run AI Prediction")

    if predict:

        features = np.array([[age, sex, cp, trestbps, chol, fbs,
                              restecg, thalach, exang, oldpeak,
                              slope, ca, thal]])

        scaled = scaler.transform(features)
        prediction = model.predict(scaled)
        severity_score = round(prediction[0], 2)

        st.session_state["severity"] = severity_score

        # Risk categorization
        if severity_score < 0.75:
            risk = "Low Risk"
            color = "green"
        elif severity_score < 1.75:
            risk = "Moderate Risk"
            color = "orange"
        else:
            risk = "High Risk"
            color = "red"

        st.markdown(
            f"""
            <div class="card">
                <div class="kpi" style="color:{color};">{risk}</div>
                <div style="text-align:center;">Severity Score: {severity_score}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# TAB 2 – ANALYTICS
# ============================================================
with tab2:

    if "severity" in st.session_state:

        severity_score = st.session_state["severity"]

        st.markdown("### 📊 Risk Gauge")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=severity_score,
            title={'text': "Heart Severity Score"},
            gauge={
                'axis': {'range': [0, 3]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 1], 'color': "#16a34a"},
                    {'range': [1, 2], 'color': "#facc15"},
                    {'range': [2, 3], 'color': "#dc2626"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🧬 Feature Profile Radar")

        radar_features = {
            "Age": age/100,
            "Cholesterol": chol/600,
            "Blood Pressure": trestbps/200,
            "Heart Rate": thalach/220,
            "Oldpeak": oldpeak/10,
            "Vessels": ca/3
        }

        radar_df = pd.DataFrame(dict(
            r=list(radar_features.values()),
            theta=list(radar_features.keys())
        ))

        fig_radar = px.line_polar(radar_df, r='r', theta='theta', line_close=True)
        fig_radar.update_traces(fill='toself')
        fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0,1])))

        st.plotly_chart(fig_radar, use_container_width=True)

    else:
        st.info("Run prediction first.")

# ============================================================
# TAB 3 – MODEL INSIGHTS
# ============================================================
with tab3:

    st.success("""
    ✔ KNN handles non-linear medical patterns  
    ✔ Scaled features improve distance-based learning  
    ✔ Cholesterol & Blood Pressure strong indicators  
    ✔ Exercise-induced angina important factor  
    """)

# ============================================================
# TAB 4 – HEALTH REPORT
# ============================================================
with tab4:

    if "severity" in st.session_state:

        severity_score = st.session_state["severity"]

        st.markdown("### 📝 Personalized Health Summary")

        if severity_score < 1:
            st.success("Your heart indicators appear stable. Maintain healthy lifestyle.")
        elif severity_score < 2:
            st.warning("Moderate risk detected. Consider regular health checkups.")
        else:
            st.error("High risk detected. Immediate medical consultation advised.")

    else:
        st.info("Run prediction to generate report.")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94a3b8;'>© 2026 Akshit Gajera | Heart Health Intelligence Platform</div>",
    unsafe_allow_html=True
)
