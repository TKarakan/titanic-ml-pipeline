import pandas as pd
import joblib
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_csv(path):
    try:
        df = pd.read_csv(path)
        logger.info(f"Csv başarıyla yüklendi:{path} | Satır sayısı: {len(df)} ")
        return df
    except Exception as e:
        logger.error(f"CSV yüklenirken hata oluştu! Yol: {path} | Hata: {e}")
        raise 
    
def save_data(df, path, format="csv"):
    try:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            df.to_csv(save_path, index=False)
        elif format == "json":
            df.to_json(save_path, orient ="records")
        elif format == "excel":
            df.to_excel(save_path, index=False)
        else:
            raise ValueError(f"Desteklenmeyen format: {format}")
        
        logger.info(f"Veri başarıyla kaydedildi: {save_path} | Format: {format}")

    except Exception as e:
        logger.error(f"Veri kaydedilirken hata oluştu! Yol: {path} | Hata: {e}")
        raise 
    
    
def load_model(path):
    try:
        model = joblib.load(path)
        logger.info(f"Model yüklendi: {path}")
        
        return model
    
    except Exception as e:
        logger.error(f"Model yüklenirken hata oluştu: Yol:{path} | Hata : {e}")
        raise
    
def save_model(model, path):
    try:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model,path)
        logger.info(f"Model başarıyla kaydedildi: {path}")
        
    except Exception as e:
        logger.error(f"Model kaydedilirken hata oluştu! Yol: {path} | Hata : {e}")
        raise

def load_submission(path):
    try:
        df = pd.read_csv(path)
        logger.info(f"Submission yüklendi: {path} | Satir: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Submission yüklenemedi! Yol: {path} | Hata: {e}")
        raise
    
def save_submission(df, path):
    try:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Submission kaydedildi: {save_path}")
    except Exception as e:
        logger.error(f"Submission kaydedilemedi! Yol: {path} | Hata: {e}")
        raise
    



