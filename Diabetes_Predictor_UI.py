import streamlit as st
import requests

API_URL = "http://127.0.0.1:5000/predict"

st.title("🩺 Diabetes Prediction App")

Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
Glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
Age = st.number_input("Age", min_value=0, max_value=120, value=30)

if st.button("Predict Diabetes"):
    input_data = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age
    }

    response = requests.post(API_URL, json=input_data)

    if response.status_code == 200:
        result = response.json()["prediction"]
        if result == 1:
            st.error("⚠️ Patient is likely to have Diabetes")
        else:
            st.success("✅ Patient is unlikely to have Diabetes")
    else:
        st.warning("Something went wrong. Please check backend API.")
