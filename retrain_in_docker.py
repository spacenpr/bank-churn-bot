import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

print("🔄 Training model on Render...")

df = pd.read_csv('bank_churn_dataset.csv')

exclude_cols = ['id', 'full_name', 'address', 'origin_province',
                'last_active_date', 'created_date', 'exit', 'monthly_ir']

feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].copy()
y = df['exit']

categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_seed=42, verbose=False)
model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

joblib.dump(model, 'advanced_ml_reports/best_model_v3.pkl')
print("✅ Model trained and saved")