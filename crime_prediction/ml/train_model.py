import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Загрузка данных
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'ml', 'crimedata_cleaned.csv')  # Укажите путь к файлу

# Загрузка данных
crimedata_cleaned = pd.read_csv(DATA_PATH)
features = ['population', 'pctUrban', 'racePctWhite']  # Example features
X = crimedata_cleaned[features]  # Feature set
y = crimedata_cleaned['ViolentCrimesPerPop']  # Target variable (number of violent crimes)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
print('MSE:', mean_squared_error(y_test, y_pred))

# Сохранение модели
joblib.dump(model, r'c:/Users/askar/Code/Crime-predictor/crime_prediction/model/crime_model.pkl')
