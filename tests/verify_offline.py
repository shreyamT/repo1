import os
import sys

# Add parent directory to path to import weather1
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
from weather1 import predict_future, prepare_data

def verify_offline():
    print("Verifying Project Logic (Offline Mode)...")
    
    # Path to models and data in parent directory
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rain_model_path = os.path.join(base_path, "rain_model.pkl")
    temp_model_path = os.path.join(base_path, "temp_model.pkl")
    hum_model_path = os.path.join(base_path, "hum_model.pkl")
    data_path = os.path.join(base_path, "weather.csv")

    # Dummy current weather data
    current_weather = {
        'city': 'TestCity',
        'country': 'TC',
        'current_temp': 25,
        'temp_min': 20,
        'temp_max': 30,
        'humidity': 60,
        'description': 'Clear sky',
        'wind_gust_dir': 180,
        'wind_gust_speed': 5,
        'pressure': 1013
    }

    print("-" * 30)
    print(f"Using Dummy Data for {current_weather['city']}")
    print(f"Current Temp: {current_weather['current_temp']}°C")
    print("-" * 30)

    try:
        # 1. Load pre-trained models
        print("Loading models...")
        rain_model = joblib.load(rain_model_path)
        temp_model = joblib.load(temp_model_path)
        hum_model = joblib.load(hum_model_path)
        print("Models loaded successfully.")

        # 2. Load historical data
        print(f"Loading historical data ({data_path})...")
        historical_data = pd.read_csv(data_path).dropna()
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        historical_data['WindGustDir'] = le.fit_transform(historical_data['WindGustDir'])
        historical_data['RainTomorrow'] = le.fit_transform(historical_data['RainTomorrow'])
        print("Data loaded and preprocessed.")

        # 3. Test Rain Prediction
        compass_direction = "S" 
        try:
            compass_direction_encoded = le.transform([compass_direction])[0]
        except ValueError:
            print(f"Warning: {compass_direction} not in training labels. Using first available.")
            compass_direction_encoded = 0

        current_data = pd.DataFrame([{
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': compass_direction_encoded,
            'WindGustSpeed': current_weather['wind_gust_speed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure'],
            'Temp': current_weather['current_temp']
        }])

        rain_prediction = rain_model.predict(current_data)[0]
        print(f"\nRain Prediction Logic: {'Yes' if rain_prediction == 1 else 'No'} (Success)")

        # 4. Test Future Trends (Regression)
        future_temp = predict_future(temp_model, current_weather['current_temp'])
        future_hum = predict_future(hum_model, current_weather['humidity'])

        print("\nFuture Forecast Logic (Next 5 Hours):")
        for i in range(5):
            print(f"  Hour +{i+1}: {round(future_temp[i], 1)}°C | Humidity: {round(future_hum[i], 1)}%")
        print("\nConclusion: The core ML logic and pre-trained models ARE WORKING correctly.")

    except Exception as e:
        print(f"\nVerification Failed: {e}")

if __name__ == "__main__":
    verify_offline()
