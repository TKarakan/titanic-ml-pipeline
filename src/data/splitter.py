import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config_parser import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_config = load_config("config/config.yaml")


def split(df, target_col="Survived"):
    """
    DataFrame'i train ve validation olarak böler.
    Stratified split uygular — hedef değişkenin dağılımını korur.

    Döndürür: X_train, X_val, y_train, y_val
    """
    try:
        logger.info("Veri bölme işlemi başlıyor...")

        # Survived'ı hemen ayır
        # X sadece feature'lar — pipeline bunu dönüştürecek
        # y hedef — pipeline buna dokunmayacak

        if target_col not in df.columns:
            raise KeyError(target_col)

       
        # X → özellikler (feature'lar)
        # y → tahmin edeceğimiz şey (Survived)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Config'den parametreleri al
        test_size = _config["model_params"]["test_size"]       # 0.2
        random_state = _config["model_params"]["random_state"] # 42

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y              

        # stratify=y — Survived dağılımını train ve val'de eşit tutar
        # %38 hayatta kalan varsa, her iki sette de %38 olur
        )

        logger.info(
            f"Bölme tamamlandı | "
            f"Train: {len(X_train)} satır | "
            f"Validation: {len(X_val)} satır | "
            f"Test size: {test_size}"
        )

        # Oranları kontrol et — stratify çalıştı mı?
        train_survival_rate = y_train.mean().round(3)
        val_survival_rate = y_val.mean().round(3)

        logger.info(
            f"Hayatta kalma oranı | "
            f"Train: {train_survival_rate} | "
            f"Validation: {val_survival_rate}"
        )

        return X_train, X_val, y_train, y_val

    except KeyError as e:
        # target_col DataFrame'de yoksa spesifik hata ver
        # "Survived" mi yazdın, "survived" mi — küçük harf farkı bile KeyError
        logger.error(f"Sütun bulunamadı: {e} — DataFrame'de '{target_col}' var mı?")
        raise

    except Exception as e:
        logger.error(f"Veri bölme hatası: {e}")
        raise