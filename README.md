# Heart-Disease-Detection

Deployment URL :- https://heart-disease-predictor-model.streamlit.app/

# 🫀 Heart Disease Detection using K-Nearest Neighbors (KNN)
A Machine Learning web application that predicts whether a patient has heart disease based on medical attributes.
Built using **K-Nearest Neighbors (KNN)** algorithm and deployed with **Streamlit**.

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


