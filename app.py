import streamlit as st
import joblib
import pandas as pd
from weather1 import get_current_weather, prepare_regression_data, predict_future

# Load pre-trained models
@st.cache_resource
def load_models():
    rain_model = joblib.load("rain_model.pkl")
    temp_model = joblib.load("temp_model.pkl")
    hum_model = joblib.load("hum_model.pkl")
    return rain_model, temp_model, hum_model

def main():
    st.title("Weather Prediction App")

    # Load models
    st.info("Loading pre-trained models...")
    rain_model, temp_model, hum_model = load_models()
    st.success("Models loaded successfully!")

    # Input for city
    city = st.text_input("Enter city:", "New York")

    if st.button("Get Current Weather"):
        with st.spinner("Fetching current weather..."):
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
        with st.spinner("Processing uploaded data..."):
            try:
                historical_data = pd.read_csv(data_file)
                historical_data = historical_data.dropna().drop_duplicates()

                # Prepare data for predictions
                x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
                x_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')

                st.success("Data processed successfully!")

                # Make predictions
                if st.button("Predict Future Weather"):
                    future_temp = predict_future(temp_model, historical_data['Temp'].iloc[-1])
                    future_hum = predict_future(hum_model, historical_data['Humidity'].iloc[-1])

                    # Display future predictions
                    st.write("\nFuture Temp:")
                    for i, temp in enumerate(future_temp):
                        st.write(f"Hour {i + 1}: {round(temp, 1)}C")

                    st.write("\nFuture Humidity:")
                    for i, humidity in enumerate(future_hum):
                        st.write(f"Hour {i + 1}: {round(humidity, 1)}%")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()