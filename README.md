# Heart-Disease-Detection

Deployment URL :- https://heart-disease-predictor-model.streamlit.app/

# 🫀 Heart Disease Detection using K-Nearest Neighbors (KNN)
A Machine Learning web application that predicts whether a patient has heart disease based on medical attributes.
Built using **K-Nearest Neighbors (KNN)** algorithm and deployed with **Streamlit**.

📁 Dataset Overview

This project uses a Heart Disease Dataset that contains medical and physiological information of patients, aimed at predicting the presence of heart disease. The dataset includes various clinical parameters such as age, blood pressure, cholesterol levels, and heart-related test results.

It is widely used for building machine learning classification models to assist in early detection and risk assessment of cardiovascular diseases.

📊 Dataset Summary
| Property        | Value                     |
| --------------- | ------------------------- |
| Dataset Type    | Healthcare / Medical Data |
| Data Type       | Structured (Tabular)      |
| Feature Types   | Numerical + Categorical   |
| Target Variable | Heart Disease Presence    |
| Task Type       | Binary Classification     |


🎯 Target Variable
target

This column indicates whether a patient has heart disease:

| Value | Meaning                   |
| ----- | ------------------------- |
| 0     | No Heart Disease          |
| 1     | Presence of Heart Disease |


🔑 Key Features

The dataset includes important medical attributes such as:

Age – Age of the patient
Sex – Gender of the patient
Chest Pain Type (cp)
Resting Blood Pressure (trestbps)
Cholesterol Level (chol)
Fasting Blood Sugar (fbs)
Resting ECG Results (restecg)
Maximum Heart Rate Achieved (thalach)
Exercise-Induced Angina (exang)
ST Depression (oldpeak)
Slope of Peak Exercise ST Segment (slope)
Number of Major Vessels (ca)
Thalassemia (thal)

These features are critical indicators used in diagnosing heart disease.

🎯 Objective of the Dataset

The main objective of this dataset is to:

Predict whether a patient is likely to have heart disease
Analyze the relationship between medical attributes and heart conditions
Assist in early diagnosis using machine learning models

🧠 Analysis Use Cases

This dataset can be used for:

Heart disease prediction models
Medical risk analysis
Feature importance analysis
Healthcare analytics
Classification model benchmarking

## 🚀 Live Features
    - Interactive user input interface
    - Real-time prediction
    - Feature scaling using StandardScaler
    - Clean and modern UI
    - 90% Model Accuracy

## 📊 Dataset Features
The model is trained on medical attributes:

    - age
    - sex
    - cp (Chest Pain Type)
    - trestbps (Resting Blood Pressure)
    - chol (Cholesterol)
    - fbs (Fasting Blood Sugar)
    - restecg (Resting ECG)
    - thalach (Maximum Heart Rate Achieved)
    - exang (Exercise Induced Angina)
    - oldpeak
    - slope
    - ca (Number of Major Vessels)
    - thal
    - target

## 🧠 Model Details

    - Algorithm: K-Nearest Neighbors (KNN)
    - Accuracy: ~90%
    - Feature Scaling: StandardScaler
    - Model Serialization: Pickle
    - Deployment: Streamlit

## 📂 Project Structure

    Heart-Disease-Detection-KNN/
    ├── app.py
    ├── model.pkl
    ├── scaler.pkl
    ├── requirements.txt
    ├── README.md
    └── heart.csv  


---

## ▶️ Run Locally
1. Clone repository :-
   
         https://github.com/akshitgajera1013/Heart-Disease-Detection.git

2. Install dependencies :-
   
        pip install -r requirements.txt

3. Run the app :-

        streamlit run app.py

## 🎯 Output
        - ✅ Heart Disease Not Detected
        - ⚠️ Heart Disease Detected

## 💡 Future Improvements
    - Add probability score
    - Add model comparison
    - Deploy on Streamlit Cloud
    - Add visualization dashboard

## 👨‍💻 Author
Akshit Gajera  
Aspiring Data Scientist | Machine Learning Enthusiast  


