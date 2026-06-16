import os
import sys

# Add parent directory to path to import weather1
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
from weather1 import get_current_weather, predict_future, prepare_data
from datetime import datetime, timedelta
import pytz

def cli_weather():
    city = input("Enter city (e.g., Dehradun): ").strip()
    if not city:
        print("Please enter a valid city name.")
        return

    print(f"\nFetching current weather for {city}...")
    current_weather = get_current_weather(city)

    if "error" in current_weather:
        print(f"Error: {current_weather['error']}")
        return

    print("-" * 30)
    print(f"City: {current_weather['city']}, {current_weather['country']}")
    print(f"Current Temp: {current_weather['current_temp']}°C")
    print(f"Feels Like: {current_weather['feels_like']}°C")
    print(f"Humidity: {current_weather['humidity']}%")
    print(f"Conditions: {current_weather['description']}")
    print("-" * 30)

    try:
        # Path to models and data in parent directory
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rain_model_path = os.path.join(base_path, "rain_model.pkl")
        temp_model_path = os.path.join(base_path, "temp_model.pkl")
        hum_model_path = os.path.join(base_path, "hum_model.pkl")
        data_path = os.path.join(base_path, "weather.csv")

        # Load pre-trained models
        rain_model = joblib.load(rain_model_path)
        temp_model = joblib.load(temp_model_path)
        hum_model = joblib.load(hum_model_path)

        # Load historical data to get the LabelEncoder for wind direction
        historical_data = pd.read_csv(data_path).dropna()
        _, _, le = prepare_data(historical_data)

        # Handle Wind Direction encoding
        wind_deg = current_weather.get('wind_gust_dir', 0) % 360
        compass_points = [
            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75)
        ]
        
        compass_direction = "N"
        for point, start, end in compass_points:
            if start <= wind_deg < end:
                compass_direction = point
                break
        
        compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else le.transform([le.classes_[0]])[0]

        # Prepare data for rain prediction
        current_data = pd.DataFrame([{
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': compass_direction_encoded,
            'WindGustSpeed': current_weather.get('wind_gust_speed', 0),
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather.get('pressure', 1013),
            'Temp': current_weather['current_temp']
        }])

        rain_prediction = rain_model.predict(current_data)[0]
        print(f"Rain Prediction for tomorrow: {'Yes' if rain_prediction == 1 else 'No'}")

        # Future predictions
        future_temp = predict_future(temp_model, current_weather['current_temp'])
        future_hum = predict_future(hum_model, current_weather['humidity'])

        print("\nFuture Forecast (Next 5 Hours):")
        for i in range(5):
            print(f"  Hour +{i+1}: {round(future_temp[i], 1)}°C | Humidity: {round(future_hum[i], 1)}%")

    except Exception as e:
        print(f"\nPrediction Error: {e}")

if __name__ == "__main__":
    cli_weather()
