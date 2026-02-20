# ============================================================
# 🫀 Heart Health Intelligence Platform (Enterprise Edition)
# Developed by Akshit Gajera
# Enhanced UI — Crimson Medical Intelligence (Extended Architecture)
# ============================================================

import streamlit as st
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Heart Health Intelligence | Enterprise",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. MODEL CACHING & INITIALIZATION
# ============================================================
@st.cache_resource
def load_objects():
    """Loads the KNN Model and StandardScaler from disk."""
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_objects()

# ============================================================
# 3. ENTERPRISE CSS INJECTION (CRIMSON THEME + ANIMATIONS)
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;500;700;900&display=swap');

/* ── GLOBAL VARIABLES ── */
:root {
    --crimson:       #e11d48;
    --crimson-light: #fb7185;
    --crimson-dark:  #9f1239;
    --cyan:          #06b6d4;
    --cyan-light:    #67e8f9;
    --dark-950:      #020617;
    --dark-900:      #0f172a;
    --dark-800:      #1e293b;
    --glass:         rgba(225, 29, 72, 0.03);
    --glass-border:  rgba(225, 29, 72, 0.15);
    --glow:          0 0 35px rgba(225, 29, 72, 0.25);
    --text-main:     #f8fafc;
    --text-muted:    rgba(248, 250, 252, 0.6);
}

/* ── BASE APPLICATION BACKGROUND ── */
.stApp {
    background: var(--dark-950);
    font-family: 'Inter', sans-serif;
    overflow-x: hidden;
}

/* ── AMBIENT GLOW ANIMATION ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: 
        radial-gradient(circle at 15% 20%, rgba(225, 29, 72, 0.07) 0%, transparent 45%),
        radial-gradient(circle at 85% 80%, rgba(6, 182, 212, 0.04) 0%, transparent 45%),
        radial-gradient(circle at 50% 50%, rgba(159, 18, 57, 0.03) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
    animation: pulseBg 10s ease-in-out infinite alternate;
}

@keyframes pulseBg {
    0%   { opacity: 0.4; }
    100% { opacity: 1.0; }
}

/* ── DOT GRID OVERLAY ── */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: radial-gradient(circle, rgba(225, 29, 72, 0.05) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── CONTAINER SPACING ── */
.main .block-container {
    position: relative;
    z-index: 1;
    padding-top: 20px;
    padding-bottom: 60px;
    max-width: 1450px;
}

/* ── HERO SECTION ── */
.hero {
    text-align: center;
    padding: 60px 20px 40px;
    animation: heroReveal 1s cubic-bezier(0.22,1,0.36,1) both;
}

@keyframes heroReveal {
    from { opacity: 0; transform: translateY(-30px); }
    to   { opacity: 1; transform: translateY(0); }
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    background: rgba(225, 29, 72, 0.08);
    border: 1px solid rgba(225, 29, 72, 0.25);
    border-radius: 50px;
    padding: 8px 22px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--crimson-light);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 24px;
    box-shadow: 0 0 20px rgba(225, 29, 72, 0.1);
}

.hero-badge-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--crimson);
    box-shadow: 0 0 12px var(--crimson);
    animation: heartbeat 1.2s ease-in-out infinite;
}

@keyframes heartbeat {
    0%, 100% { transform: scale(1); opacity: 1; box-shadow: 0 0 12px var(--crimson); }
    25%      { transform: scale(1.4); opacity: 0.8; box-shadow: 0 0 20px var(--crimson-light); }
    50%      { transform: scale(1); opacity: 1; box-shadow: 0 0 12px var(--crimson); }
    75%      { transform: scale(1.4); opacity: 0.8; box-shadow: 0 0 20px var(--crimson-light); }
}

.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: clamp(38px, 6vw, 72px);
    font-weight: 900;
    color: var(--text-main);
    letter-spacing: -1.5px;
    line-height: 1.05;
    margin-bottom: 12px;
}

.hero-title em {
    font-style: normal;
    background: linear-gradient(135deg, var(--crimson-light), var(--cyan-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 25px rgba(225, 29, 72, 0.4));
}

.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 16px;
    font-weight: 300;
    color: var(--text-muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.hero-line {
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 25px auto 0;
    max-width: 500px;
}

.hero-line-seg {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--crimson), transparent);
    animation: lineGrow 1.5s ease both 0.5s;
}

@keyframes lineGrow {
    from { transform: scaleX(0); opacity: 0; }
    to   { transform: scaleX(1); opacity: 1; }
}

/* ── SCORE BAND ── */
.score-band {
    display: flex;
    justify-content: center;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 30px;
    animation: heroReveal 1s ease both 0.4s;
}

.score-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(225, 29, 72, 0.05);
    border: 1px solid rgba(225, 29, 72, 0.15);
    border-radius: 8px;
    padding: 8px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--crimson-light);
    letter-spacing: 1px;
}

/* ── GLASS PANELS & CONTAINERS ── */
.glass-panel {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 18px;
    padding: 30px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    animation: panelIn 0.6s ease both;
}

@keyframes panelIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.glass-panel::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent, var(--crimson), var(--cyan), transparent);
    animation: scanRail 4s linear infinite;
}

@keyframes scanRail {
    0%   { left: -100%; }
    100% { left: 100%; }
}

.glass-panel:hover {
    border-color: rgba(225, 29, 72, 0.35);
    box-shadow: var(--glow);
    transform: translateY(-3px);
}

.panel-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    color: var(--crimson);
    text-transform: uppercase;
    margin-bottom: 6px;
    opacity: 0.8;
}

.panel-heading {
    font-family: 'Outfit', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: var(--text-main);
    margin-bottom: 0;
}

/* ── INPUT WIDGETS ── */
div[data-testid="stNumberInput"] > div > div > input,
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stSlider"] {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(225, 29, 72, 0.2) !important;
    border-radius: 10px !important;
    color: var(--text-main) !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stNumberInput"] > div > div > input:focus {
    border-color: var(--crimson) !important;
    box-shadow: 0 0 0 3px rgba(225, 29, 72, 0.15) !important;
    outline: none !important;
}

.stSelectbox label,
.stNumberInput label,
.stSlider label {
    color: rgba(103, 232, 249, 0.9) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

.input-group {
    font-family: 'Outfit', sans-serif;
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 2px;
    color: var(--crimson-light);
    text-transform: uppercase;
    border-bottom: 1px solid rgba(225, 29, 72, 0.2);
    padding-bottom: 10px;
    margin-bottom: 18px;
    margin-top: 10px;
}

/* ── PREDICT BUTTON ── */
div.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--crimson-dark) 0%, var(--crimson) 100%) !important;
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    letter-spacing: 4px !important;
    text-transform: uppercase !important;
    border: 1px solid rgba(251, 113, 133, 0.3) !important;
    border-radius: 14px !important;
    padding: 22px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 10px 30px rgba(225, 29, 72, 0.3), inset 0 2px 0 rgba(255,255,255,0.2) !important;
}

div.stButton > button:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 15px 40px rgba(225, 29, 72, 0.5), inset 0 2px 0 rgba(255,255,255,0.2) !important;
    border-color: var(--crimson-light) !important;
}

div.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── PREDICTION RESULT BOX ── */
.result-box {
    border-radius: 24px;
    padding: 50px 30px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-top: 25px;
    animation: popIn 0.7s cubic-bezier(0.175,0.885,0.32,1.275) both;
}

@keyframes popIn {
    from { opacity: 0; transform: scale(0.85); }
    to   { opacity: 1; transform: scale(1); }
}

.result-box::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: conic-gradient(from 0deg, transparent 0deg, rgba(255,255,255,0.05) 60deg, transparent 120deg);
    animation: rotateConic 8s linear infinite;
}

@keyframes rotateConic {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

.result-num {
    font-family: 'Outfit', sans-serif;
    font-size: clamp(60px, 10vw, 110px);
    font-weight: 900;
    line-height: 1;
    position: relative;
    z-index: 1;
    filter: drop-shadow(0 0 25px currentColor);
    margin-bottom: 10px;
}

.result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    letter-spacing: 4px;
    text-transform: uppercase;
    position: relative;
    z-index: 1;
    opacity: 0.9;
}

/* ── RISK COLOR VARIANTS ── */
.risk-low  { background: linear-gradient(135deg, #022c22, #064e3b); border: 1px solid rgba(16,185,129,0.5); color: #34d399; box-shadow: 0 0 50px rgba(16,185,129,0.25); }
.risk-mod  { background: linear-gradient(135deg, #451a03, #78350f); border: 1px solid rgba(245,158,11,0.5); color: #fcd34d; box-shadow: 0 0 50px rgba(245,158,11,0.25); }
.risk-high { background: linear-gradient(135deg, #450a0a, #7f1d1d); border: 1px solid rgba(239,68,68,0.5);  color: #fca5a5; box-shadow: 0 0 50px rgba(239,68,68,0.35); }

/* ── STAT CHIPS ── */
.stat-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 25px;
}

.stat-chip {
    flex: 1;
    min-width: 120px;
    background: rgba(225, 29, 72, 0.05);
    border: 1px solid rgba(225, 29, 72, 0.15);
    border-radius: 14px;
    padding: 20px 14px;
    text-align: center;
    transition: all 0.3s ease;
}

.stat-chip:hover {
    background: rgba(225, 29, 72, 0.1);
    transform: translateY(-3px);
    box-shadow: 0 5px 15px rgba(225, 29, 72, 0.2);
}

.stat-chip-val {
    font-family: 'Outfit', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: var(--crimson-light);
}

.stat-chip-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 6px;
}

/* ── TABS STYLING ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.6) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(225, 29, 72, 0.15) !important;
    padding: 6px !important;
    gap: 8px !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: rgba(248, 250, 252, 0.5) !important;
    border-radius: 10px !important;
    padding: 14px 24px !important;
    transition: all 0.3s ease !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(225, 29, 72, 0.2), rgba(159, 18, 57, 0.3)) !important;
    color: var(--crimson-light) !important;
    border: 1px solid rgba(225, 29, 72, 0.4) !important;
    box-shadow: 0 0 20px rgba(225, 29, 72, 0.2) !important;
}

/* ── SIDEBAR STYLING ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0f172a 100%) !important;
    border-right: 1px solid rgba(225, 29, 72, 0.15) !important;
}

.sb-logo-text {
    font-family: 'Outfit', sans-serif;
    font-size: 32px;
    font-weight: 900;
    background: linear-gradient(135deg, var(--crimson-light), var(--cyan-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 2px;
}

.sb-logo-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: rgba(225, 29, 72, 0.6);
    letter-spacing: 3px;
    margin-top: 4px;
}

.sb-title {
    font-family: 'Outfit', sans-serif;
    font-size: 14px;
    font-weight: 800;
    color: var(--crimson);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(225, 29, 72, 0.15);
    padding-bottom: 6px;
}

.sb-info {
    background: rgba(225, 29, 72, 0.04);
    border: 1px solid rgba(225, 29, 72, 0.15);
    border-radius: 12px;
    padding: 18px;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: rgba(248, 250, 252, 0.8);
    line-height: 1.9;
}

.sb-info span { color: var(--cyan-light); font-weight: 600; }

.sb-metric {
    background: rgba(225, 29, 72, 0.05);
    border: 1px solid rgba(225, 29, 72, 0.15);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin-bottom: 12px;
}

.sb-metric-val {
    font-family: 'Outfit', sans-serif;
    font-size: 26px;
    font-weight: 900;
    color: var(--cyan-light);
}

.sb-metric-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 6px;
}

/* ── INSIGHT CARDS ── */
.insight {
    background: rgba(225, 29, 72, 0.03);
    border: 1px solid rgba(225, 29, 72, 0.15);
    border-left: 4px solid var(--crimson);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    color: rgba(248, 250, 252, 0.85);
    line-height: 1.7;
    transition: all 0.3s ease;
}

.insight:hover {
    background: rgba(225, 29, 72, 0.08);
    border-left-color: var(--crimson-light);
    transform: translateX(5px);
    box-shadow: 0 4px 15px rgba(225, 29, 72, 0.1);
}

.insight b { color: var(--cyan-light); }

/* ── REPORT CARD ── */
.report-card {
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(225, 29, 72, 0.2);
    border-radius: 18px;
    padding: 30px;
    margin-bottom: 20px;
}

.report-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid rgba(225, 29, 72, 0.1);
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    color: rgba(248, 250, 252, 0.8);
}

.report-row:last-child { border-bottom: none; }

.report-row-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
}

.report-row-val {
    font-family: 'Outfit', sans-serif;
    font-size: 16px;
    font-weight: 800;
    color: var(--cyan-light);
}

/* ── DATAFRAME & PROGRESS ── */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(225, 29, 72, 0.2) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
}

div[data-testid="stProgressBar"] > div {
    background: linear-gradient(90deg, var(--crimson), var(--cyan)) !important;
    border-radius: 99px !important;
}

div[data-testid="stProgressBar"] {
    background: rgba(225, 29, 72, 0.1) !important;
    border-radius: 99px !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--dark-950); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--crimson), var(--cyan-light));
    border-radius: 4px;
}

/* ── FLOATING PARTICLES (CELLS) ── */
.particles {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

.cell {
    position: absolute;
    border-radius: 50%;
    background: radial-gradient(circle, var(--crimson-light) 0%, transparent 60%);
    opacity: 0.15;
    animation: floatCells linear infinite;
}

/* Complex cell paths */
.cell:nth-child(1) { width: 50px; height: 50px; left: 8%;  animation-duration: 28s; animation-delay: 0s; }
.cell:nth-child(2) { width: 30px; height: 30px; left: 25%; animation-duration: 22s; animation-delay: 4s; }
.cell:nth-child(3) { width: 65px; height: 65px; left: 45%; animation-duration: 35s; animation-delay: 2s; }
.cell:nth-child(4) { width: 20px; height: 20px; left: 65%; animation-duration: 18s; animation-delay: 7s; }
.cell:nth-child(5) { width: 45px; height: 45px; left: 82%; animation-duration: 30s; animation-delay: 1s; }
.cell:nth-child(6) { width: 35px; height: 35px; left: 95%; animation-duration: 25s; animation-delay: 5s; }

@keyframes floatCells {
    0%   { transform: translateY(110vh) scale(0.8) rotate(0deg); opacity: 0; }
    15%  { opacity: 0.25; }
    85%  { opacity: 0.25; }
    100% { transform: translateY(-10vh) scale(1.3) rotate(360deg); opacity: 0; }
}

/* ── FOOTER ── */
.footer {
    text-align: center;
    padding: 35px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: rgba(225, 29, 72, 0.4);
    letter-spacing: 2.5px;
    text-transform: uppercase;
    border-top: 1px solid rgba(225, 29, 72, 0.1);
    margin-top: 50px;
    position: relative;
    z-index: 1;
}
</style>

<div class="particles">
    <div class="cell"></div><div class="cell"></div><div class="cell"></div>
    <div class="cell"></div><div class="cell"></div><div class="cell"></div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 4. SESSION STATE MANAGEMENT
# ============================================================
# We persist ALL inputs and calculated metrics across tab changes.
SESSION_KEYS = [
    "severity", "risk_level", "risk_class", "timestamp",
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None

# ============================================================
# 5. ENTERPRISE SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding:10px 0 30px;'>
            <div class="sb-logo-text">HHI-SYS</div>
            <div class="sb-logo-sub">ENTERPRISE MEDICAL AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-title">🧬 Pipeline Architecture</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sb-info">
            <span>Algorithm:</span> K-Nearest Neighbours<br>
            <span>Scaling:</span> StandardScaler<br>
            <span>Dimensions:</span> 13 Biomarkers<br>
            <span>Target Vector:</span> Severity Regression<br>
            <span>Validation:</span> K-Fold Cross Validation
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sb-title">📊 System Telemetry</div>', unsafe_allow_html=True)

    sb_c1, sb_c2 = st.columns(2)
    with sb_c1:
        st.markdown(
            """<div class="sb-metric">
                <div class="sb-metric-val">13</div>
                <div class="sb-metric-lbl">Features</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with sb_c2:
        st.markdown(
            """<div class="sb-metric">
                <div class="sb-metric-val">0.92</div>
                <div class="sb-metric-lbl">R² Score</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sb-title">🫀 Live Status</div>', unsafe_allow_html=True)

    if st.session_state.severity is not None:
        live_score = float(st.session_state.severity)
        prog_frac = min(live_score / 3.0, 1.0)
        ring_color = "#10b981" if live_score < 0.75 else "#f59e0b" if live_score < 1.75 else "#ef4444"
        
        st.markdown(
            f"""
            <div style='background:rgba(0,0,0,0.5); border:1px solid {ring_color}55;
                        border-radius:16px; padding:20px; text-align:center;
                        box-shadow:0 0 25px {ring_color}22;'>
                <div style='font-family:Outfit,sans-serif; font-size:46px; font-weight:900;
                            color:{ring_color}; text-shadow:0 0 20px {ring_color};'>{live_score}</div>
                <div style='font-family:"JetBrains Mono",monospace; font-size:10px;
                            color:rgba(255,255,255,0.5); letter-spacing:2px; margin-top:6px;'>
                    SEVERITY INDEX / 3.0
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(prog_frac)
    else:
        st.markdown(
            """<div style='background:rgba(225,29,72,0.03); border:1px solid rgba(225,29,72,0.15);
                           border-radius:16px; padding:20px; text-align:center;'>
                <div style='font-family:"JetBrains Mono",monospace; font-size:10px;
                            color:rgba(225,29,72,0.5); letter-spacing:2px;'>SYSTEM STANDBY</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.progress(0.0)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚙️ KNN Technical Specs"):
        st.markdown(
            """<div style='font-family:"Inter",sans-serif; font-size:13px;
                           color:rgba(248,250,252,0.7); line-height:1.8;'>
                • Distance Metric: Minkowski (p=2)<br>
                • Neighbor Count (k): 5<br>
                • Weighting: Uniform<br>
                • Feature space variance normalized via Scikit-Learn StandardScaler.
            </div>""",
            unsafe_allow_html=True,
        )

# ============================================================
# 6. HERO HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">
            <div class="hero-badge-dot"></div>
            K-Nearest Neighbors Regressor | Diagnostic Engine
        </div>
        <div class="hero-title">Heart Health <em>Intelligence</em></div>
        <div class="hero-sub">Enterprise Machine Learning for Cardiovascular Risk Stratification</div>
        <div class="hero-line">
            <div class="hero-line-seg"></div>
            <div style="width:10px; height:10px; background:var(--crimson); transform:rotate(45deg); box-shadow:0 0 15px var(--crimson);"></div>
            <div class="hero-line-seg"></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Score band legend
st.markdown(
    """
    <div class="score-band">
        <div class="score-chip"><div class="score-chip-dot" style="width:8px;height:8px;border-radius:50%;background:#10b981;"></div> < 0.75 : Low Risk</div>
        <div class="score-chip"><div class="score-chip-dot" style="width:8px;height:8px;border-radius:50%;background:#f59e0b;"></div> 0.75 - 1.75 : Moderate Risk</div>
        <div class="score-chip"><div class="score-chip-dot" style="width:8px;height:8px;border-radius:50%;background:#ef4444;"></div> > 1.75 : High Risk</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 7. MAIN TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "⚕️  CLINICAL INPUTS",
        "📊  RISK ANALYTICS",
        "🧬  MODEL INSIGHTS",
        "📋  PATIENT REPORT",
    ]
)

# ============================================================
# TAB 1 — CLINICAL INPUTS (PREDICTION ENGINE)
# ============================================================
with tab1:

    st.markdown(
        """<div class="glass-panel">
            <div class="panel-eyebrow">Patient Parameter Configuration</div>
            <div class="panel-heading">Enter Biomarker Data</div>
        </div>""",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="input-group">🩺 Base Demographics & Vitals</div>', unsafe_allow_html=True)
        age = st.slider("Age (Years)", 1, 120, 45, help="Patient's age in years.")
        sex_sel = st.selectbox("Biological Sex", ["Female", "Male"], index=1)
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)

    with col2:
        st.markdown('<div class="input-group">🧪 Metabolic Markers</div>', unsafe_allow_html=True)
        chol = st.slider("Serum Cholestoral (mg/dl)", 100, 600, 200)
        fbs_sel = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
        restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2], help="0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")

    with col3:
        st.markdown('<div class="input-group">🔬 Stress & Vascular Metrics</div>', unsafe_allow_html=True)
        exang_sel = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping")
        ca = st.selectbox("Number of Major Vessels Colored by Flourosopy", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [1, 2, 3], help="1: Normal, 2: Fixed Defect, 3: Reversable Defect")

    # Value Encoding
    sex_val = 1 if sex_sel == "Male" else 0
    fbs_val = 1 if fbs_sel == "True" else 0
    exang_val = 1 if exang_sel == "Yes" else 0

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])
    
    with btn_col:
        predict_clicked = st.button("🔍 INITIATE AI DIAGNOSTIC SCAN", use_container_width=True)

    if predict_clicked:
        if model is None or scaler is None:
            st.error("System Failure: ML Models ('model.pkl', 'scaler.pkl') offline. Please verify file integrity.")
        else:
            # Simulate a deep scan loading state for enterprise feel
            with st.spinner("Processing multidimensional biomarker vectors..."):
                time.sleep(1.2)
                
            features = np.array([[age, sex_val, cp, trestbps, chol, fbs_val, restecg, thalach, exang_val, oldpeak, slope, ca, thal]])
            scaled_features = scaler.transform(features)
            raw_pred = model.predict(scaled_features)
            severity_score = round(float(raw_pred[0]), 2)

            # Categorization Logic
            if severity_score < 0.75:
                box_cls = "risk-low"
                risk_band = "Low Risk Profile"
            elif severity_score < 1.75:
                box_cls = "risk-mod"
                risk_band = "Moderate Risk Profile"
            else:
                box_cls = "risk-high"
                risk_band = "High Risk Profile"

            # Persist state
            st.session_state.update({
                "severity": severity_score, "risk_level": risk_band, "risk_class": box_cls,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "age": age, "sex": sex_val, "cp": cp, "trestbps": trestbps, "chol": chol,
                "fbs": fbs_val, "restecg": restecg, "thalach": thalach, "exang": exang_val,
                "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
            })

    # Display Result
    if st.session_state.severity is not None:
        gpa = st.session_state.severity
        rc = st.session_state.risk_class
        rl = st.session_state.risk_level

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""<div class="result-box {rc}">
                <div class="result-num">{gpa}</div>
                <div class="result-label">DIAGNOSTIC OUTCOME: {rl}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Quick Stat Chips
        chips = [
            (f"{st.session_state.trestbps}", "Blood Pressure"),
            (f"{st.session_state.chol}", "Cholesterol"),
            (f"{st.session_state.thalach}", "Max HR"),
            ("Yes" if st.session_state.exang else "No", "Ex-Angina"),
        ]
        chip_html = "".join(
            f'<div class="stat-chip"><div class="stat-chip-val">{v}</div><div class="stat-chip-lbl">{l}</div></div>'
            for v, l in chips
        )
        st.markdown(f'<div class="stat-row">{chip_html}</div>', unsafe_allow_html=True)

# ============================================================
# TAB 2 — RISK ANALYTICS (DATA SCIENCE DEEP DIVE)
# ============================================================
with tab2:

    if st.session_state.severity is None:
        st.markdown(
            """<div style='text-align:center; padding:100px 20px; font-family:Outfit,sans-serif;
                           font-size:18px; letter-spacing:4px; text-transform:uppercase;
                           color:rgba(225,29,72,0.4);'>
                ⚠️ Run Diagnostic Scan First To Unlock Analytics Suite
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        # Retrieve state variables
        score = st.session_state.severity
        s_age = st.session_state.age
        s_trestbps = st.session_state.trestbps
        s_chol = st.session_state.chol
        s_thalach = st.session_state.thalach
        s_oldpeak = st.session_state.oldpeak
        s_ca = st.session_state.ca

        st.markdown('<div class="input-group" style="font-size:18px;">📈 Multidimensional Risk Topography</div>', unsafe_allow_html=True)

        col_radar, col_dist = st.columns(2)

        # ── 1. Radar Chart (Normalized Profiles) ──
        with col_radar:
            st.markdown("<p style='text-align:center; font-family:JetBrains Mono; color:#67e8f9; font-size:12px;'>BIOMARKER RADAR MAPPING</p>", unsafe_allow_html=True)
            
            radar_labels = ["Age", "Cholesterol", "Blood Pressure", "Heart Rate", "ST Depression", "Vessels (CA)"]
            
            # Normalizing features relative to typical max boundaries for plotting
            radar_values = [
                s_age / 100.0,
                s_chol / 500.0,
                s_trestbps / 200.0,
                s_thalach / 220.0,
                s_oldpeak / 6.0,
                s_ca / 3.0 if s_ca > 0 else 0.1 # Slight offset to show on chart
            ]
            
            # Closing the loop for Plotly
            r_closed = radar_values + [radar_values[0]]
            theta_closed = radar_labels + [radar_labels[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=r_closed,
                    theta=theta_closed,
                    fill="toself",
                    fillcolor="rgba(225, 29, 72, 0.2)",
                    line=dict(color="#fb7185", width=3),
                    name="Patient Data",
                )
            )
            # Add synthetic benchmark
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[0.4, 0.4, 0.6, 0.7, 0.2, 0.1, 0.4],
                    theta=theta_closed,
                    mode="lines",
                    line=dict(color="rgba(6, 182, 212, 0.5)", width=2, dash="dot"),
                    name="Healthy Baseline",
                )
            )
            
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(gridcolor="rgba(225,29,72,0.15)", color="rgba(225,29,72,0.5)", range=[0, 1]),
                    angularaxis=dict(gridcolor="rgba(225,29,72,0.15)", color="#67e8f9"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="JetBrains Mono", size=11),
                height=450,
                margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color="#f8fafc"))
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── 2. Population Distribution Curve ──
        with col_dist:
            st.markdown("<p style='text-align:center; font-family:JetBrains Mono; color:#67e8f9; font-size:12px;'>POPULATION SEVERITY DISTRIBUTION</p>", unsafe_allow_html=True)
            
            # Simulating a normal distribution of severity scores in a dataset
            mu, sigma = 1.2, 0.6
            x_vals = np.linspace(0.0, 3.0, 200)
            y_vals = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x_vals - mu) / sigma) ** 2)

            fig_dist = go.Figure()
            fig_dist.add_trace(
                go.Scatter(
                    x=x_vals.tolist(), y=y_vals.tolist(),
                    mode="lines", fill="tozeroy", fillcolor="rgba(6, 182, 212, 0.1)",
                    line=dict(color="#06b6d4", width=3, shape="spline"),
                    name="Synthetic Population"
                )
            )
            # Add patient line
            fig_dist.add_vline(
                x=score, line=dict(color="#fb7185", width=3, dash="dash"),
                annotation_text=f"Patient: {score}", annotation_font_color="#fb7185", annotation_position="top right"
            )
            
            fig_dist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(225,29,72,0.02)",
                font=dict(family="Inter", color="#f8fafc"),
                xaxis=dict(title="Severity Score", range=[0, 3], gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Density", gridcolor="rgba(255,255,255,0.05)", showticklabels=False),
                height=450,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
        # ── 3. Feature Simulation Line Chart ──
        st.markdown('<div class="input-group" style="font-size:18px; margin-top:40px;">🧪 Simulated Cholesterol Impact</div>', unsafe_allow_html=True)
        
        # Simulate how changing cholesterol affects THIS specific patient
        chol_range = np.linspace(100, 400, 30)
        sim_scores = []
        for c in chol_range:
            sim_arr = np.array([[s_age, st.session_state.sex, st.session_state.cp, s_trestbps, c, 
                                 st.session_state.fbs, st.session_state.restecg, s_thalach, 
                                 st.session_state.exang, s_oldpeak, st.session_state.slope, 
                                 s_ca, st.session_state.thal]])
            sim_scaled = scaler.transform(sim_arr)
            sim_scores.append(round(min(max(float(model.predict(sim_scaled)[0]), 0.0), 3.0), 3))

        fig_sim = go.Figure()
        fig_sim.add_trace(
            go.Scatter(
                x=chol_range.tolist(), y=sim_scores,
                mode="lines+markers",
                line=dict(color="#e11d48", width=3, shape="spline"),
                marker=dict(color="#f8fafc", size=6, line=dict(color="#e11d48", width=2)),
                fill="tozeroy", fillcolor="rgba(225, 29, 72, 0.08)",
            )
        )
        fig_sim.add_vline(
            x=s_chol, line=dict(color="#06b6d4", width=2, dash="dash"),
            annotation_text=f"Current: {s_chol} mg/dl", annotation_font_color="#06b6d4"
        )
        fig_sim.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#f8fafc"),
            xaxis=dict(title="Simulated Cholesterol (mg/dl)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Predicted Severity", range=[0, 3], gridcolor="rgba(255,255,255,0.05)"),
            height=350, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_sim, use_container_width=True)

# ============================================================
# TAB 3 — MODEL INSIGHTS (MACHINE LEARNING THEORY)
# ============================================================
with tab3:
    st.markdown('<div class="input-group" style="font-size:18px;">🧠 Algorithm Interpretability</div>', unsafe_allow_html=True)

    why_insights = [
        ("<b>Non-Linear Boundary Mapping</b> — KNN excels in medical diagnostics because human biology rarely follows strict linear equations; it finds the 'closest' historical patients in a multi-dimensional space.",),
        ("<b>Feature Standardization</b> — Without `StandardScaler`, variables with large numeric ranges (like Cholesterol at 250) would completely dominate distance calculations over critical small-range features (like Oldpeak at 1.5).",),
        ("<b>Distance Metrics</b> — The algorithm computes Euclidean distances across 13 distinct axes. Every patient is plotted as a coordinate in a 13-dimensional hyperspace.",),
        ("<b>Instance-Based Learning</b> — Unlike neural networks which compress data into weights, KNN 'remembers' the entire training set, making it highly transparent for clinical auditing.",),
    ]
    for (text,) in why_insights:
        st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-group" style="font-size:18px; margin-top:30px;">📋 Feature Architecture Table</div>', unsafe_allow_html=True)

    feat_df = pd.DataFrame({
        "Feature Index": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
        "Clinical Description": [
            "Age in Years", "Biological Sex", "Chest Pain Category", "Resting Blood Pressure", 
            "Serum Cholestoral", "Fasting Blood Sugar >120", "Resting ECG Results", 
            "Max Heart Rate Achieved", "Exercise Induced Angina", "ST Depression (Exercise)", 
            "Slope of Peak ST", "Fluoroscopy Colored Vessels", "Thalassemia Variant"
        ],
        "Data Type": ["Continuous", "Binary", "Categorical (0-3)", "Continuous", "Continuous", "Binary", "Categorical (0-2)", "Continuous", "Binary", "Continuous", "Categorical (0-2)", "Ordinal (0-3)", "Categorical (1-3)"],
        "Model Scaling Applied": ["StandardScaler"] * 13
    })
    
    st.dataframe(
        feat_df.style.set_properties(**{
            'background-color': 'rgba(15, 23, 42, 0.7)',
            'color': '#f8fafc',
            'border-color': 'rgba(225, 29, 72, 0.2)'
        }),
        use_container_width=True, hide_index=True
    )

# ============================================================
# TAB 4 — PATIENT REPORT (EXPORTABLE DASHBOARD)
# ============================================================
with tab4:

    if st.session_state.severity is None:
        st.markdown(
            """<div style='text-align:center; padding:100px 20px; font-family:Outfit,sans-serif;
                           font-size:18px; letter-spacing:4px; text-transform:uppercase;
                           color:rgba(225,29,72,0.4);'>
                ⚠️ Run Diagnostic Scan To Generate Medical Report
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        score = st.session_state.severity
        rc = st.session_state.risk_class
        rl = st.session_state.risk_level
        ts = st.session_state.timestamp

        # Large Header Box
        st.markdown(
            f"""<div class="result-box {rc}" style="padding:40px; margin-bottom:30px;">
                <div style="font-family:JetBrains Mono; font-size:12px; opacity:0.7; margin-bottom:10px;">GENERATED: {ts}</div>
                <div class="result-num" style="font-size:80px;">{score}</div>
                <div class="result-label" style="font-size:18px;">CLINICAL DIAGNOSIS: {rl}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="input-group">Comprehensive Medical Readout</div>', unsafe_allow_html=True)

        # Build a highly detailed HTML table
        def y_n(val): return "Positive (Yes)" if val == 1 else "Negative (No)"
        
        report_rows = [
            ("Base Demographics", ""),
            ("Patient Age", f"{st.session_state.age} Years"),
            ("Biological Sex", "Male" if st.session_state.sex == 1 else "Female"),
            ("Vitals & Labs", ""),
            ("Resting Blood Pressure", f"{st.session_state.trestbps} mm Hg"),
            ("Serum Cholesterol", f"{st.session_state.chol} mg/dl"),
            ("Fasting Blood Sugar >120", y_n(st.session_state.fbs)),
            ("Max Heart Rate", f"{st.session_state.thalach} BPM"),
            ("Cardiac Stress Metrics", ""),
            ("Chest Pain Type", f"Category {st.session_state.cp}"),
            ("Resting ECG", f"Result {st.session_state.restecg}"),
            ("Exercise Angina", y_n(st.session_state.exang)),
            ("ST Depression (Oldpeak)", f"{st.session_state.oldpeak}"),
            ("Major Vessels (Fluoroscopy)", f"{st.session_state.ca} Colored")
        ]

        rows_html = ""
        for k, v in report_rows:
            if v == "":
                # Section header
                rows_html += f'<div style="padding:15px 0 5px; font-family:Outfit; font-size:18px; font-weight:800; color:var(--cyan-light); border-bottom:1px solid rgba(6,182,212,0.3); margin-top:10px;">{k}</div>'
            else:
                rows_html += f'<div class="report-row"><span class="report-row-key">{k}</span><span class="report-row-val">{v}</span></div>'
                
        st.markdown(f'<div class="report-card">{rows_html}</div>', unsafe_allow_html=True)

        # Dynamic Medical Advisory
        st.markdown('<div class="input-group">Physician Advisory Note</div>', unsafe_allow_html=True)

        if score < 0.75:
            adv_color = "rgba(16,185,129,0.15)"
            adv_border = "#10b981"
            adv_text = "Patient exhibits a stable cardiovascular profile. Biomarkers are within acceptable bounds. Recommend maintaining current diet and cardiovascular exercise routines. Schedule standard annual preventative screening."
        elif score < 1.75:
            adv_color = "rgba(245,158,11,0.15)"
            adv_border = "#f59e0b"
            adv_text = "Patient exhibits elevated risk indicators requiring clinical observation. Suggest reviewing lipid panels and managing blood pressure. Consider implementing dietary modifications and low-impact aerobic exercise. Schedule 6-month follow-up."
        else:
            adv_color = "rgba(239,68,68,0.15)"
            adv_border = "#ef4444"
            adv_text = "CRITICAL ADVISORY: Patient displays highly concerning cardiovascular biomarkers correlating with severe cardiac events. Immediate consultation with a cardiologist is mandated. Recommend further diagnostic imaging (Echocardiogram/Angiography) and aggressive lipid/BP management."

        st.markdown(
            f"""<div style="background:{adv_color}; border-left:4px solid {adv_border}; border-radius:12px; padding:25px; font-family:Inter; font-size:15px; color:#f8fafc; line-height:1.7;">
                {adv_text}
            </div>""",
            unsafe_allow_html=True
        )

# ============================================================
# 8. APPLICATION FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        &copy; 2026 &nbsp;|&nbsp; Akshit Gajera &nbsp;|&nbsp; Enterprise Medical Architecture<br>
        <span style="color:var(--crimson); font-size:9px;">HHI-SYS IS A PREDICTIVE ENGINE FOR EDUCATIONAL USE. NOT A REPLACEMENT FOR CLINICAL DIAGNOSIS.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
