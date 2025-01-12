import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime, timedelta
import pytz


BASE_URL= 'https://api.openweathermap.org/data/2.5/'
API_KEY='cb6888afcc7062ec5277c708a99bf566'
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    
    if response.status_code != 200:
        return {"error": f"Failed to fetch data: {response.status_code}, {response.text}"}
    
    data = response.json()
    
    if 'main' not in data or 'weather' not in data:
        return {"error": "Incomplete data received from API"}
    
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'wind_gust_speed': data['wind']['speed']
    }
def read_historical_data(filename):
    try:
        df = pd.read_csv(filename)
        df = df.dropna()
        df = df.drop_duplicates()
        return df
    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file {filename} is empty or corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame as fallback
def prepare_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

    x = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    y = data['RainTomorrow']
    
    return x, y, le

    
    
def train_rain_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

    return model
def prepare_regression_data(data, feature):
    x, y = [], []

    for i in range(len(data)-1):
        x.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i+1])

    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    return x, y
def train_regression_model(x, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x, y)
    return model
def predict_future(model, current_value):
    predictions = [current_value]

    for i in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])

    return predictions[1:]
def weather_view():
    city = input("Enter city: ")
    current_weather = get_current_weather(city)

    # Read historical data and prepare models
    historical_data = read_historical_data('https://github.com/shreyamT/repo1/blob/main/weather.csv')

    x, y, le = prepare_data(historical_data)
    rain_model = train_rain_model(x, y)

    # Process wind direction and encode it
    wind_deg = current_weather['wind_gust_dir'] % 360
    compass_points = [
        ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75)
    ]
    compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)
    compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

    # Prepare current weather data for prediction
    current_data = {
        'MinTemp': current_weather['temp_min'],
        'MaxTemp': current_weather['temp_max'],
        'WindGustDir': compass_direction_encoded,
        'WindGustSpeed': current_weather['wind_gust_speed'],
        'Humidity': current_weather['humidity'],
        'Pressure': current_weather['pressure'],
        'Temp': current_weather['current_temp']
    }

    current_df = pd.DataFrame([current_data])

    # Make rain prediction
    rain_prediction = rain_model.predict(current_df)[0]

    # Train models for temperature and humidity predictions
    x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
    x_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')

    temp_model = train_regression_model(x_temp, y_temp)
    hum_model = train_regression_model(x_hum, y_hum)

    # Make future predictions for temperature and humidity
    future_temp = predict_future(temp_model, current_weather['temp_min'])
    future_hum = predict_future(hum_model, current_weather['humidity'])

    # Get current time and generate future times
    timezone = pytz.timezone('Asia/Kolkata')
    now = datetime.now(timezone)
    next_hour = now + timedelta(hours=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)

    future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

    # Display current and future weather details
    print(f"City: {city}, {current_weather['country']}")
    print(f"Current Temp: {current_weather['current_temp']}C")
    print(f"Feels Like: {current_weather['feels_like']}C")
    print(f"Min Temp: {current_weather['temp_min']}C")
    print(f"Max Temp: {current_weather['temp_max']}C")
    print(f"Humiity: {current_weather['humidity']}%")
    print(f"Weather Prediction: {current_weather['description']}")
    print(f"Rain Prediction: {'Yes' if rain_prediction else 'No'}")

    print("\nFuture Temp: ")
    for time, temp in zip(future_times, future_temp):
        print(f"{time}: {round(temp, 1)}C")
    
    print("\nFuture Humidity: ")
    for time, humidity in zip(future_times, future_hum):
        print(f"{time}: {round(humidity, 1)}%")

  weather_view()
