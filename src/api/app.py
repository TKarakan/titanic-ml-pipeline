from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import os
import pandas as pd
from src.utils.logger import get_logger
from src.pipeline.train_pipeline import build_pipeline
from src.utils.io_helper import load_model

logger = get_logger(__name__)

app_state = {"model": None, "pipeline": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # En yeni modeli bul
        model_dir = Path("/app/models")
        model_files = sorted(model_dir.glob("*.pkl"), reverse=True)

        if not model_files:
            raise FileNotFoundError("Model bulunamadı — önce train_pipeline.py çalıştır.")

        model_path = model_files[0]
        app_state["model"]    = load_model(model_path)
        app_state["pipeline"] = build_pipeline()

        # Pipeline'ı fit et — train verisiyle
        # Model zaten eğitildi, pipeline'ın transformer'larını da fit etmemiz lazım
        from src.utils.io_helper import load_csv
        df = load_csv("/app/data/raw/train.csv")
        X  = df.drop(columns=["Survived"])
        app_state["pipeline"].fit(X)

        logger.info(f"Model ve pipeline yüklendi: {model_path.name}")

    except Exception as e:
        logger.error(f"Başlatma hatası: {e}")
        raise

    yield
    app_state["model"]    = None
    app_state["pipeline"] = None


app = FastAPI(title="Titanic Prediction API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok" if app_state["model"] is not None else "waiting",
        "model_loaded": app_state["model"] is not None
    }


@app.get("/")
def root():
    return {"message": "Titanic API is online"}


@app.post("/predict")
def predict(payload: dict):
    if app_state["model"] is None or app_state["pipeline"] is None:
        return {"error": "Model henüz hazır değil."}

    try:
        # Ham veriyi pipeline'dan geçir
        df              = pd.DataFrame([payload])
        df_transformed  = app_state["pipeline"].transform(df)

        prediction  = app_state["model"].predict(df_transformed)[0]
        probability = app_state["model"].predict_proba(df_transformed)[0][1]

        return {
            "survived":            int(prediction),
            "survival_probability": round(float(probability), 4),
            "verdict":             "Hayatta" if prediction == 1 else "Hayatta kalamadı"
        }

    except Exception as e:
        logger.error(f"Tahmin hatası: {e}")
        return {"error": str(e)}