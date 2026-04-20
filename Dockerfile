# Base image — Python 3.11 slim
# slim → gereksiz sistem araçları yok, image daha küçük
# 3.11 → kararlı, production için iyi
FROM python:3.11-slim

# curl healthcheck için gerekli
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Çalışma dizini — container içinde her şey buraya gidecek
WORKDIR /app

# Önce sadece requirements.txt kopyala
# Neden önce? Docker layer cache — requirements değişmemişse
# pip install tekrar çalışmaz, hızlanır
COPY requirements.txt .

# Kütüphaneleri kur
RUN pip install --no-cache-dir -r requirements.txt

# Sonra projenin geri kalanını kopyala
COPY . .

# FastAPI 8000 portunda çalışacak
# Bu satır sadece dokümantasyon — port açmaz, compose açar
EXPOSE 8000
EXPOSE 8501
# Container başladığında ne çalışsın
# --host 0.0.0.0 → container dışından erişim için zorunlu
# 127.0.0.1 yazarsan container dışından erişemezsin
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]