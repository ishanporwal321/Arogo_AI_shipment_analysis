from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import h2o
import datetime as dt
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# Initialize H2O and load the saved model
h2o.init()
model_path = "../models/StackedEnsemble_BestOfFamily_1_AutoML_3_20241222_31119"  # Update with the correct path
model = h2o.load_model(model_path)

scaler = joblib.load('api/scaler.pkl')
# Create the FastAPI app
app = FastAPI()

# Define the schema for incoming shipment data
class ShipmentRequest(BaseModel):
    Origin: str
    Destination: str
    Shipment_Date: str  # Format: YYYY-MM-DD
    Vehicle_Type: str
    Distance_km: float
    Weather_Conditions: str
    Traffic_Conditions: str

    class Config:
        orm_mode = True

# Define the required mappings
city_list = [
    "Bangalore", "Chennai", "Delhi", "Hyderabad", "Jaipur", 
    "Kolkata", "Lucknow", "Mumbai", "Pune"
]

vehicle_type_map = {"Truck": 3, "Lorry": 1, "Container": 0, "Trailer": 2}
weather_map = {"Clear": 0, "Rain": 1, "Fog": 2}
traffic_map = {"Light": 1, "Moderate": 2, "Heavy": 3}

@app.post("/predict")
async def predict(shipment: ShipmentRequest):
    try:
        # Convert Shipment_Date to datetime and extract features
        shipment_date = dt.datetime.strptime(shipment.Shipment_Date, "%Y-%m-%d")
        day_of_week = shipment_date.weekday()
        is_weekend = 1 if day_of_week in [5, 6] else 0
        month = shipment_date.month

        # Map categorical inputs
        origin_one_hot = [1 if city == shipment.Origin else 0 for city in city_list]
        destination_one_hot = [1 if city == shipment.Destination else 0 for city in city_list]
        vehicle_type = vehicle_type_map.get(shipment.Vehicle_Type, -1)
        weather_code = weather_map.get(shipment.Weather_Conditions, -1)
        traffic_code = traffic_map.get(shipment.Traffic_Conditions, -1)

        # Validate mappings
        if vehicle_type == -1 or weather_code == -1 or traffic_code == -1:
            return {"error": "Invalid input for Vehicle_Type, Weather_Conditions, or Traffic_Conditions"}

        # Create a DataFrame for model input
        input_data = {
            "Vehicle Type": vehicle_type,
            "Distance (km)": shipment.Distance_km,
            "Is Weekend": is_weekend,
            "Weather Code": weather_code,
            "Traffic Code": traffic_code,
            "Day of Week": day_of_week,
            "Month": month,
        }
        # Add origin and destination one-hot features
        for idx, city in enumerate(city_list):
            input_data[f"Origin_{city}"] = origin_one_hot[idx]
            input_data[f"Destination_{city}"] = destination_one_hot[idx]

        # Convert input_data to DataFrame
        input_df = pd.DataFrame([input_data])
        expected_features = [
            "Vehicle Type", "Distance (km)", "Is Weekend", "Weather Code", 
            "Traffic Code", "Day of Week", "Month"
        ] + [f"Origin_{city}" for city in city_list] + [f"Destination_{city}" for city in city_list]

        input_df = input_df[expected_features]

        # Scale numerical features
        numerical_features = ["Distance (km)", "Day of Week", "Month"]
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        # Convert to H2OFrame
        input_h2o = h2o.H2OFrame(input_df)

        # Make prediction
        prediction = model.predict(input_h2o)
        prediction_value = int(prediction[0, 0])

        # Map prediction to human-readable result
        result = "On Time"
        return {"prediction": result}
    
    except Exception as e:
        print(f"Exception occurred: {e}")
        return {"error": str(e)}