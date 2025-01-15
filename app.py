# app.py
import streamlit as st
from weather1 import get_current_weather, read_historical_data, prepare_data, train_rain_model, prepare_regression_data, train_regression_model, predict_future

def main():
    st.title("Weather Prediction App")

    # Input for city
    city = st.text_input("Enter city:", "New York")

    if st.button("Get Current Weather"):
        current_weather = get_current_weather(city)
        if "error" in current_weather:
            st.error(current_weather["error"])
        else:
            st.write(f"City: {current_weather['city']}, {current_weather['country']}")
            st.write(f"Current Temp: {current_weather['current_temp']}C")
            st.write(f"Feels Like: {current_weather['feels_like']}C")
            st.write(f"Min Temp: {current_weather['temp_min']}C")
            st.write(f"Max Temp: {current_weather['temp_max']}C")
            st.write(f"Humidity: {current_weather['humidity']}%")
            st.write(f"Weather Prediction: {current_weather['description']}")

    # Input for historical data file
    data_file = st.file_uploader("Upload historical weather data (CSV)", type=["csv"])

    if data_file is not None:
        historical_data = read_historical_data(data_file)

        # Prepare data and train models
        x, y, le = prepare_data(historical_data)
        rain_model = train_rain_model(x, y)
        x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
        x_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
        temp_model = train_regression_model(x_temp, y_temp)
        hum_model = train_regression_model(x_hum, y_hum)

        st.success("Models trained successfully")

        # Make future predictions
        if st.button("Predict Future Weather"):
            future_temp = predict_future(temp_model, current_weather['temp_min'])
            future_hum = predict_future(hum_model, current_weather['humidity'])

            # Display future predictions
            st.write("\nFuture Temp:")
            for i, temp in enumerate(future_temp):
                st.write(f"Hour {i+1}: {round(temp, 1)}C")

            st.write("\nFuture Humidity:")
            for i, humidity in enumerate(future_hum):
                st.write(f"Hour {i+1}: {round(humidity, 1)}%")

if __name__ == "__main__":
    main()
