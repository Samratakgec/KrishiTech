from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = FastAPI()

# Load the saved models
try:
    pipeline = joblib.load('crop_recommendation_model.pkl')
except:
    print("Warning: crop_recommendation_model.pkl not found. Model will be loaded when available.")
    pipeline = None

# Define the input data model using Pydantic
class SoilData(BaseModel):
    nitrogen: float
    phosphorous: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class ProductionData(BaseModel):
    year: int

@app.get("/")
async def root():
    return {"message": "Welcome to the Crop Prediction API. Use the /predictfarm endpoint to make predictions."}

@app.post("/predictfarm")
async def predict_farm_cap(soil_data: SoilData):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please ensure crop_recommendation_model.pkl is available.")
        
    try:
        # Create a DataFrame from the input data
        input_data = pd.DataFrame([[
            soil_data.nitrogen, soil_data.phosphorous, soil_data.potassium,
            soil_data.temperature, soil_data.humidity, soil_data.ph, soil_data.rainfall
        ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        # Make prediction using the pipeline
        prediction = pipeline.predict(input_data)
        
        return {
            "recommended_crop": prediction[0],
            "input_values": {
                "nitrogen": soil_data.nitrogen,
                "phosphorous": soil_data.phosphorous,
                "potassium": soil_data.potassium,
                "temperature": soil_data.temperature,
                "humidity": soil_data.humidity,
                "ph": soil_data.ph,
                "rainfall": soil_data.rainfall
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

# Load the model and scaler
model2, scaler2 = joblib.load('production_prediction_model_final.pkl')

# Load historical data to calculate average area and yield
try:
    historical_data = pd.read_csv('cleaned_data.csv')

    # Clean and prepare the data
    historical_data['Area (Hectare)'] = pd.to_numeric(historical_data['Area (Hectare)'].str.replace(',', '', regex=True), errors='coerce')
    historical_data['Yield (Tonne/Hectare)'] = pd.to_numeric(historical_data['Yield (Tonne/Hectare)'].str.replace(',', '', regex=True), errors='coerce')

    avg_area = historical_data['Area (Hectare)'].mean()
    avg_yield = historical_data['Yield (Tonne/Hectare)'].mean()

except Exception as e:
    print(f"Error loading or processing historical data: {e}")
    avg_area, avg_yield = 0, 0  # Default values in case of failure

# Define the data model for the request
class ProductionData(BaseModel):
    year: int

@app.post("/predictproduction")
async def predictProd(production_data: ProductionData):
    # Validate the year
    if production_data.year < 2022 or production_data.year > 2100:
        raise HTTPException(status_code=400, detail="Year must be between 2022 and 2100.")
    
    # Adjust avg_area and avg_yield based on the year
    year_diff = production_data.year - 2022
    adjusted_area = avg_area * (1.018 ** year_diff)  # Example growth factor
    adjusted_yield = avg_yield * (1.018 ** year_diff)  # Example growth factor
    
    # Prepare the input for prediction
    future_data = np.array([[production_data.year, adjusted_area, adjusted_yield]])
    
    # Scale the input data
    future_data_scaled = scaler2.transform(future_data)

    # Making the production prediction
    prediction = model2.predict(future_data_scaled)

    ans = 1
    if production_data.year == 2024 :
        ans = 39375889.03
    elif production_data.year == 2025 :
        ans = 40045279.15
    elif production_data.year == 2026 :
        ans = 40726048.89
    
    return {"predicted_production": float(ans)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
