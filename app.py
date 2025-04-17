import streamlit as st
import requests
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import os

# Title of the app
st.title("KrishiTech")

# Header and subheader
st.header("Welcome Farmer")
st.subheader("Enter necessary details of soil and region")

# Input fields
n = st.number_input("Enter Nitrogen content:", min_value=0.0, format="%f")
p = st.number_input("Enter Phosphorous content:", min_value=0.0, format="%f")
k = st.number_input("Enter Potassium content:", min_value=0.0, format="%f")
temp = st.number_input("Enter Temperature:", min_value=0.0, format="%f")
humidity = st.number_input("Enter Humidity :", min_value=0.0, format="%f")
ph = st.number_input("Enter pH level:", min_value=0.0, format="%f")
rainfall = st.number_input("Enter rainfall:", min_value=0.0, format="%f")

# Button for soil prediction
if st.button("Submit"):
    # Validate all fields are non-zero
    if n == 0.0 or p == 0.0 or k == 0.0 or temp == 0.0 or humidity == 0.0 or ph == 0.0 or rainfall == 0.0:
        st.error("All soil fields are required and must be greater than zero.")
    else:
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
            st.success(f"Recommended Crop: {result['recommended_crop']}")
        else:
            st.error("Failed to get prediction. Please try again.")

# State selection dropdown
state_options = ["Uttar Pradesh"]
selected_state = st.selectbox("Select your region", state_options)

# ==============================
# Step 1: Load and Clean Data
# ==============================
def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    data.iloc[:, 0] = "Uttar Pradesh"  # Set state to "Uttar Pradesh"
    # Fill district names
    current_city = None
    for index, row in data.iterrows():
        if pd.notna(row[1]):
            current_city = row[1]
        else:
            data.at[index, data.columns[1]] = current_city
    data.to_csv('cleaned_data.csv', index=False)  # Save cleaned data
    return data

# ==============================
# Step 2: Preprocess and Aggregate Data
# ==============================
def preprocess_data(data):
    data[['State', 'District']] = data[['State', 'District']].ffill()
    data['Year'] = data['Year'].str.split(' - ').str[0].astype(int)
    # Clean numeric columns
    numeric_columns = ['Area (Hectare)', 'Yield (Tonne/Hectare)', 'Production (Tonnes)']
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column].astype(str).str.replace(',', '', regex=True), errors='coerce')
    # Aggregate by year
    aggregated_data = data.groupby('Year').agg(
        Total_Area=('Area (Hectare)', 'sum'),
        Average_Yield=('Yield (Tonne/Hectare)', 'mean'),
        Total_Production=('Production (Tonnes)', 'sum')
    ).reset_index()
    return aggregated_data

# ==============================
# Step 3: Train the Model
# ==============================
def train_and_save_model(X, y, filename='wheat_model.pkl'):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=1.0, random_state=42)
    model.fit(X_scaled, y)
    with open(filename, 'wb') as f:
        pickle.dump((model, scaler), f)
    return model, scaler

# ==============================
# Step 4: Predict Future Production
# ==============================
def predict_future(year, aggregated_data, model, scaler):
    avg_area = aggregated_data['Total_Area'].mean() * (1 + 0.018 * (year - 2022))
    avg_yield = aggregated_data['Average_Yield'].mean() * (1 + 0.018 * (year - 2022))
    # Debug: Show calculation details in Streamlit
    future_data = pd.DataFrame({'Year': [year], 'Total_Area': [avg_area], 'Average_Yield': [avg_yield]})
    future_data_scaled = scaler.transform(future_data)
    future_production = model.predict(future_data_scaled)[0]
    return future_production

# ==============================
# Model Loading/Training Logic
# ==============================
MODEL_PATH = 'wheat_model.pkl'
CSV_PATH = 'wheat_excel.xls.csv'

if not os.path.exists(MODEL_PATH):
    raw_data = load_and_clean_data(CSV_PATH)
    agg_data = preprocess_data(raw_data)
    X = agg_data[['Year', 'Total_Area', 'Average_Yield']]
    y = agg_data['Total_Production']
    model, scaler = train_and_save_model(X, y, MODEL_PATH)
else:
    with open(MODEL_PATH, 'rb') as f:
        model, scaler = pickle.load(f)
    raw_data = load_and_clean_data(CSV_PATH)
    agg_data = preprocess_data(raw_data)

# Dropdown for production prediction
options1 = ["Wheat"]
choice1 = st.selectbox("Select the crop:", options1)

options2 = [2024, 2025, 2026]
choice2 = st.selectbox("Select year of production", options2)

# Button for production prediction
if st.button("Predict Production"):
    # Use the integrated model for prediction
    try:
        pred = predict_future(choice2, agg_data, model, scaler)
        st.success(f"Year: {choice2}, Predicted Wheat Production: {pred:.2f} tonnes")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
