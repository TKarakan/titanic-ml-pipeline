from sklearn.pipeline import Pipeline
from src.utils.config_parser import load_config
from src.utils.io_helper import load_csv
from src.utils.logger import get_logger
from src.data.cleaner import Cleaner
from src.data.feature_eng import FeatureEngineer
from src.data.encoder import Encoder
from src.data.splitter import split
from src.models.trainer import train_all
from src.models.predictor import predict_batch

logger = get_logger(__name__)
_config = load_config("config/config.yaml")


def build_pipeline():
    """
    sklearn Pipeline'ı kurar ve döndürür.

    Neden ayrı fonksiyon?
    train_pipeline ve predict ayrı yerlerde
    aynı pipeline'a ihtiyaç duyar.
    Tekrar yazmamak için buradan çağırırlar.

    Sıra önemli:
    1. Cleaner      → önce pisliği at, null doldur
    2. FeatureEng   → temiz veriden yeni sütun türet
    3. Encoder      → kategorikleri sayıya çevir
    """
    return Pipeline([
        ("cleaner",     Cleaner()),
        ("feature_eng", FeatureEngineer()),
        ("encoder",     Encoder())
    ])


def run():
    """
    Tüm eğitim sürecini çalıştırır.
    """
    try:
        logger.info("=" * 60)
        logger.info("EĞİTİM BAŞLIYOR")
        logger.info("=" * 60)

        # 1. Ham veriyi yükle
        raw_path = _config["paths"]["raw_data"]
        df = load_csv(raw_path)
        logger.info(f"Ham veri yüklendi: {df.shape}")

        # 2. Split et — pipeline'dan ÖNCE
        # Neden önce? Data leakage önlemek için.
        # FeatureEngineer.fit() title_means'i sadece
        # train verisinden öğrenmeli.
        X_train, X_val, y_train, y_val = split(df)

        # 3. Pipeline'ı kur
        pipeline = build_pipeline()

        # 4. Pipeline'ı sadece train'e fit et
        # fit_transform → fit + transform birlikte
        # val'e sadece transform — fit değil
        logger.info("Pipeline fit ediliyor...")
        pipeline.fit(X_train)
        X_train_enc = pipeline.transform(X_train)
        X_val_enc   = pipeline.transform(X_val)

        logger.info(
            f"Pipeline tamamlandı | "
            f"Train shape: {X_train_enc.shape} | "
            f"Val shape: {X_val_enc.shape}"
        )

        # 5. Tüm modelleri eğit ve değerlendir
        # train_all → MODELS listesindeki her modeli döner
        # evaluator her model için MLflow'a görsel loglar
        results, _ = train_all(
            X_train_enc, X_val_enc,
            y_train, y_val
        )

        # 6. En iyi modeli bul
        # F1'e göre sıralıyoruz — dengesiz veri için doğru metrik
        best_name = max(results, key=lambda name: results[name]["f1"])
        logger.info(f"En iyi model: {best_name} | F1: {results[best_name]['f1']}")

        # 7. En iyi modelle Kaggle submission oluştur
        # test.csv ham veri — pipeline tekrar transform edecek
        best_model = _get_best_model(results)
        predict_batch(
            pipeline=pipeline,
            model=best_model,
            test_path="data/raw/test.csv",
            experiment_name=best_name
        )

        logger.info("=" * 60)
        logger.info("EĞİTİM TAMAMLANDI")
        logger.info("=" * 60)

        return results

    except Exception as e:
        logger.error(f"Pipeline hatası: {e}")
        raise


def _get_best_model(results):
    """
    En iyi F1 skoruna sahip modeli döndürür.

    trainer.py modelleri diske kaydetti — exporter ile.
    Şimdi en iyisini diskten yükleyip döndürüyoruz.

    Neden diskten yüklüyoruz?
    train_all sadece metrics döndürüyor, model objesini değil.
    Model diske kaydedildi, oradan alıyoruz.
    """
    from src.utils.io_helper import load_model
    from src.utils.exporter import _today
    from pathlib import Path

    best_name = max(results, key=lambda name: results[name]["f1"])
    project   = _config["project_name"]["name"]
    model_dir = Path(_config["paths"]["model_export_dir"])

    # exporter'ın isimlendirme convention'ına uygun
    model_path = model_dir / f"{project}_{best_name}_{_today()}.pkl"

    logger.info(f"En iyi model yükleniyor: {model_path}")
    return load_model(model_path)


if __name__ == "__main__":
    run()