from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from typing import List

# ================================================================
# ЗАГРУЗКА МОДЕЛИ
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'advanced_ml_reports', 'best_model.pkl')

print(f"🔄 Загрузка модели из: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print("✅ Модель загружена")

# ТОЧНЫЙ порядок признаков (должен совпадать с обучением)
FEATURES = [
    'credit_sco', 'gender', 'age', 'occupation', 'balance',
    'tenure_ye', 'married', 'nums_card', 'nums_service', 'active_member',
    'last_transaction_month', 'customer_segment', 'engagement_score',
    'loyalty_level', 'digital_behavior', 'risk_score', 'risk_segment', 'cluster_group'
]


# ----------------------------------------------------------------
# МОДЕЛЬ ДАННЫХ ДЛЯ ЗАПРОСА
# ----------------------------------------------------------------
class ClientData(BaseModel):
    credit_sco: float
    age: int
    balance: float
    tenure_ye: int
    nums_card: int
    nums_service: int
    engagement_score: float
    risk_score: float
    gender: str
    married: int
    active_member: int
    customer_segment: str
    loyalty_level: str
    digital_behavior: str
    occupation: str
    risk_segment: str
    cluster_group: int
    last_transaction_month: float = 0


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    recommendation: str


# ----------------------------------------------------------------
# ФУНКЦИЯ КОДИРОВАНИЯ
# ----------------------------------------------------------------
def encode_categories(data: dict) -> dict:
    """Преобразует категориальные признаки в числа"""
    data['gender'] = 0 if data['gender'] == 'Male' else 1 if data['gender'] == 'Female' else 2

    segment_map = {'Standard': 0, 'Premium': 1, 'Mass': 2, 'Emerging': 3, 'Affluent': 4, 'Priority': 5}
    data['customer_segment'] = segment_map.get(data['customer_segment'], 0)

    loyalty_map = {'Bronze': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
    data['loyalty_level'] = loyalty_map.get(data['loyalty_level'], 0)

    behavior_map = {'Low': 0, 'Medium': 1, 'High': 2}
    data['digital_behavior'] = behavior_map.get(data['digital_behavior'], 1)

    occ_map = {'Professional': 0, 'Manager': 1, 'Student': 2, 'Retired': 3, 'Other': 4}
    data['occupation'] = occ_map.get(data['occupation'], 0)

    risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
    data['risk_segment'] = risk_map.get(data['risk_segment'], 1)

    return data


# ----------------------------------------------------------------
# API ЭНДПОИНТЫ
# ----------------------------------------------------------------
app = FastAPI(title="Bank Churn Prediction API", version="1.0.0")


@app.get("/")
def root():
    return {"message": "Bank Churn Prediction API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientData):
    try:
        data = client.dict()
        data = encode_categories(data)

        # Создаем DataFrame с правильным порядком колонок
        df = pd.DataFrame([data])
        df = df[FEATURES]

        # Все значения в float
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(0)

        print(f"📊 Данные для модели: {df.iloc[0].to_dict()}")

        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])

        if probability >= 0.7:
            risk_level = "Высокий"
            recommendation = "Немедленно предложить скидку или персональные условия"
        elif probability >= 0.4:
            risk_level = "Средний"
            recommendation = "Отправить персонализированное предложение по email"
        else:
            risk_level = "Низкий"
            recommendation = "Стандартное обслуживание, наблюдение"

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            risk_level=risk_level,
            recommendation=recommendation
        )

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------
# ЗАПУСК (исправлен для Render)
# ----------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)