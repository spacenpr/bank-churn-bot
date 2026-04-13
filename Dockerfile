FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем данные и скрипт обучения
COPY bank_churn_dataset.csv .
COPY train_on_server.py .

# Обучаем модель (это произойдет во время сборки)
RUN python train_on_server.py

# Копируем API и бота
COPY model_api.py .
COPY telegram_bot_api.py .

EXPOSE 8000

CMD ["python", "model_api.py"]