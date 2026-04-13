FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем ВСЕ необходимые файлы
COPY bank_churn_dataset.csv .
COPY retrain_in_docker.py .
COPY model_api.py .
COPY telegram_bot_api.py .

# Обучаем модель
RUN python retrain_in_docker.py

EXPOSE 8000

CMD ["python", "model_api.py"]