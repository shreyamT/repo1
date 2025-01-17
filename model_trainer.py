import pandas as pd
import joblib
from weather1 import prepare_data, train_rain_model, prepare_regression_data, train_regression_model

def main():
    # Load historical weather data
    data_path = "weather.csv" 
    historical_data = pd.read_csv(data_path).dropna().drop_duplicates()
    
    # Train Rain Prediction Model
    x, y, _ = prepare_data(historical_data)
    print("Training Rain Prediction Model...")
    rain_model = train_rain_model(x, y)
    joblib.dump(rain_model, "rain_model.pkl")
    print("Rain Prediction Model saved as 'rain_model.pkl'.")

    # Train Temperature Regression Model
    print("Training Temperature Regression Model...")
    x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
    temp_model = train_regression_model(x_temp, y_temp)
    joblib.dump(temp_model, "temp_model.pkl")
    print("Temperature Regression Model saved as 'temp_model.pkl'.")

    # Train Humidity Regression Model
    print("Training Humidity Regression Model...")
    x_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
    hum_model = train_regression_model(x_hum, y_hum)
    joblib.dump(hum_model, "hum_model.pkl")
    print("Humidity Regression Model saved as 'hum_model.pkl'.")

if __name__ == "__main__":
    main()