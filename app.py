import streamlit as st
import requests

# Title of the app
st.title("KrishiTech")

# Header and subheader
st.header("Welcome Farmer")
st.subheader("Enter necessary details of soil and region")

# Input fields
n = st.number_input("Enter Nitrogen content:")
p = st.number_input("Enter Phosphorous content:")
k = st.number_input("Enter Potassium content:")
temp = st.number_input("Enter Temperature:")
humidity = st.number_input("Enter Humidity :")
ph = st.number_input("Enter pH level:")
rainfall = st.number_input("Enter rainfall:")

# Button for soil prediction
if st.button("Submit"):
    # Prepare data payload
    data = {
        "nitrogen": n,
        "phosphorous": p,
        "potassium": k,
        "temperature": temp,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }

    # Send request to FastAPI endpoint
    response = requests.post("http://127.0.0.1:8000/predictfarm", json=data)

    # Display prediction result
    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Value: {result['prediction']}")
    else:
        st.error("Failed to get prediction. Please try again.")

# Dropdown for production prediction
options1 = ["Wheat"]
choice1 = st.selectbox("Select the crop:", options1)

options2 = [2024, 2025, 2026] 
choice2 = st.selectbox("Select year of production", options2)

# Button for production prediction
if st.button("Predict Production"):
    data = {"year": choice2}
    response = requests.post("http://127.0.0.1:8000/predictproduction", json=data)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Year: {choice2}, Predicted Production: {result['predicted_production']:.2f} tonnes")
    else:
        st.error("Failed to get production prediction. Please try again.")
