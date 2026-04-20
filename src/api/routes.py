from fastapi import APIRouter, HTTPException
from src.api.schemas import PassengerData, PredictionResponse
import pandas as pd
import joblib
import os

router = APIRouter()

# En iyi modeli yükleyelim (Dinamik yol veya config'den alınabilir)
MODEL_PATH = "models/titanic_survival_logistic_regression_20260420.pkl"

@router.post("/predict", response_model=PredictionResponse)
def predict(data: PassengerData):
    try:
        # 1. Modeli yükle
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=500, detail="Model file not found!")
        
        model = joblib.load(MODEL_PATH)
        
        # 2. Gelen veriyi DataFrame'e çevir
        input_df = pd.DataFrame([data.dict()])
        
        # 3. Tahmin yap
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        return {
            "passenger_id": 999, # Örnek ID
            "survived": int(prediction),
            "probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))