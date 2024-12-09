from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
import joblib
import numpy as np

def predict_crime(request):
    # Получение данных из запроса
    factor1 = float(request.GET.get('factor1', 0))
    factor2 = float(request.GET.get('factor2', 0))
    factor3 = float(request.GET.get('factor3', 0))

    # Загрузка модели
    model = joblib.load('model/crime_model.pkl')

    # Прогноз
    prediction = model.predict(np.array([[factor1, factor2, factor3]]))[0]

    return JsonResponse({'crime_rate': prediction})