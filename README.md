# 🚢 Titanic Survival Predictor

> 15 Nisan 1912 — O gece gemide olsaydınız, hayatta kalır mıydınız?

Makine öğrenmesi ile Titanik yolcularının hayatta kalma olasılığını tahmin eden, production-ready bir ML sistemi.

---

## 🎯 Proje Hakkında

Bu proje, veri bilimi ve ML mühendisliğinin bir arada uygulandığı end-to-end bir pipeline içerir. Ham veri işlemeden model eğitimine, REST API'den Docker deployment'a kadar tüm süreç üretim kalitesinde yazılmıştır.

---

## 🏗️ Mimari

```
titanic/
├── config/
│   └── config.yaml              # Merkezi konfigürasyon
├── data/
│   ├── raw/                     # Ham veri (train.csv, test.csv)
│   └── processed/               # İşlenmiş veri
├── models/                      # Eğitilmiş modeller (.pkl)
├── outputs/
│   └── submissions/             # Kaggle submission dosyaları
├── logs/                        # Uygulama logları
├── notebooks/
│   └── eda.ipynb                # Keşifsel veri analizi
├── src/
│   ├── utils/                   # Altyapı katmanı
│   │   ├── config_parser.py     # YAML okuyucu
│   │   ├── logger.py            # Merkezi logger
│   │   ├── io_helper.py         # Dosya okuma/yazma
│   │   └── exporter.py          # Artifact isimlendirme ve kaydetme
│   ├── data/                    # Veri işleme katmanı
│   │   ├── cleaner.py           # Null doldurma, gereksiz sütun silme
│   │   ├── feature_eng.py       # Title, FamilySize, IsAlone türetme
│   │   ├── encoder.py           # One-hot ve binary encoding
│   │   └── splitter.py          # Train/validation split
│   ├── models/                  # Model katmanı
│   │   ├── trainer.py           # 4 model eğitimi + MLflow loglama
│   │   ├── evaluator.py         # Confusion matrix, ROC, feature importance
│   │   └── predictor.py         # Tek ve toplu tahmin
│   ├── pipeline/
│   │   └── train_pipeline.py    # End-to-end eğitim orkestrasyonu
│   ├── api/
│   │   └── app.py               # FastAPI REST API
│   └── web/
│       └── ui.py                # Streamlit arayüzü
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## ⚙️ Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| Veri işleme | pandas, scikit-learn Pipeline |
| Modeller | Logistic Regression, Random Forest, XGBoost, SVM |
| Experiment tracking | MLflow |
| API | FastAPI + Uvicorn |
| Arayüz | Streamlit |
| Containerization | Docker + Docker Compose |
| Loglama | Python logging |

---

## 🚀 Kurulum

### 1. Veri setini indir

```bash
pip install kaggle
kaggle competitions download -c titanic
unzip titanic.zip -d data/raw/
```

### 2. Modeli eğit

```bash
pip install -r requirements.txt
python -m src.pipeline.train_pipeline
```

### 3. Docker ile çalıştır

```bash
docker-compose up --build
```

---

## 🌐 Servisler

| Servis | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI Swagger | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

---

## 🔬 ML Pipeline

```
Ham Veri
    ↓
Cleaner          → Gereksiz sütunlar (PassengerId, Ticket, Cabin)
                   Embarked null → 'C' (Pclass+Fare analizine dayalı)
    ↓
FeatureEngineer  → Title (Name'den regex ile)
                   Age null → Title bazlı ortalama
                   FamilySize = SibSp + Parch + 1
                   IsAlone = FamilySize == 1
    ↓
Encoder          → Sex → binary (0/1)
                   Embarked → one-hot (3 sütun)
                   Title → one-hot (5 sütun)
    ↓
Split            → %80 train / %20 validation (stratified)
    ↓
Trainer          → 4 model paralel eğitim
                   MLflow ile metrik ve görsel loglama
    ↓
Evaluator        → Confusion Matrix, ROC Curve, Feature Importance
    ↓
Predictor        → Kaggle submission CSV
```

---

## 📊 Model Sonuçları

MLflow UI üzerinden tüm deneyleri karşılaştırabilirsiniz:

```bash
http://localhost:5000
```

Karşılaştırılan metrikler: Accuracy, Precision, Recall, F1, ROC-AUC

---

## 🐳 Docker Servisleri

```yaml
api       → FastAPI  → port 8000
ui        → Streamlit → port 8501
mlflow    → MLflow UI → port 5000
```

`api` servisi healthcheck ile izlenir. `ui` servisi API tamamen hazır olmadan başlamaz.

---

## 📡 API Kullanımı

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 1,
    "Name": "Smith, Mrs. Jane",
    "Sex": "female",
    "Age": 29.0,
    "SibSp": 0,
    "Parch": 0,
    "Ticket": "UNKNOWN",
    "Fare": 78.0,
    "Cabin": null,
    "Embarked": "C"
  }'
```

```json
{
  "survived": 1,
  "survival_probability": 0.9123,
  "verdict": "Hayatta"
}
```

---

## 🗂️ Konfigürasyon

Tüm ayarlar `config/config.yaml` üzerinden yönetilir:

```yaml
paths:
  raw_data: "data/raw/train.csv"
  processed_data_dir: "data/processed"
  model_export_dir: "models/"
  submission_dir: "outputs/submissions"

model_params:
  test_size: 0.2
  random_state: 42

logging:
  log_file_path: "logs/titanic_project.log"
```