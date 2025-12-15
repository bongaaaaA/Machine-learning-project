
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "car_price_model.pkl")

model = joblib.load(MODEL_PATH)


# load model
# model = joblib.load("car_price_model.pkl")


app = FastAPI(title="Car Price Prediction API")

# data input 
class CarInput(BaseModel):
    Year: int
    Present_Price: float
    Kms_Driven: int
    Fuel_Type: int
    Seller_Type: int
    Transmission: int
    Owner: int

@app.get("/")
def home():
    return {"message": "Car Price Prediction API is running 🚀"}

@app.post("/predict")
def predict_price(car: CarInput):
    features = np.array([[
        car.Year,
        car.Present_Price,
        car.Kms_Driven,
        car.Fuel_Type,
        car.Seller_Type,
        car.Transmission,
        car.Owner
    ]])

    prediction = model.predict(features)[0]

    return {
        "Predicted_Selling_Price": round(prediction, 2)
    }

