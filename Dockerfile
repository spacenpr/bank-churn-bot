FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скачиваем модель через скрипт
RUN python -c "import joblib; import pandas as pd; from catboost import CatBoostClassifier; print('Model will be trained on server')"

COPY retrain_in_docker.py .
RUN python retrain_in_docker.py

COPY model_api.py .
COPY telegram_bot_api.py .

EXPOSE 8000

CMD ["python", "model_api.py"]