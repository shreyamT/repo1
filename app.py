import streamlit as st
import joblib
import pandas as pd
from weather1 import get_current_weather, prepare_data, prepare_regression_data, predict_future

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
    st.warning("Please enter a valid city or state name.")

    # Input city from user
    city = st.text_input("Enter city:", "Dehradun")

    if st.button("Show Weather"):
        if not city.strip():
            st.warning("Please enter a valid city or state name.")
            return

        st.write(f"Fetching data for: {city}")

        # Fetch current weather
        with st.spinner("Fetching current weather..."):
            current_weather = get_current_weather(city)

        if "error" in current_weather:
            st.error(current_weather["error"])
            return

        # Display current weather
        st.write(f"City: {current_weather['city']}, {current_weather['country']}")
        st.write(f"Current Temp: {current_weather['current_temp']}C")
        st.write(f"Feels Like: {current_weather['feels_like']}C")
        st.write(f"Min Temp: {current_weather['temp_min']}C")
        st.write(f"Max Temp: {current_weather['temp_max']}C")
        st.write(f"Humidity: {current_weather['humidity']}%")
        st.write(f"Weather Details: {current_weather['description']}")

        # Load historical weather data
        st.write("Processing historical weather...")
        try:
            historical_data = pd.read_csv("weather.csv").dropna().drop_duplicates()

            # Prepare data for rain prediction
            if not historical_data.empty:
                x_rain, y_rain, _ = prepare_data(historical_data)

                # Ensure x_rain has valid data
                if len(x_rain) > 0:
                    last_instance = x_rain.iloc[-1].values.reshape(1, -1)  # Safely get the last instance
                    try:
                        rain_prediction = rain_model.predict(last_instance)[0]
                        rain_status = "Yes" if rain_prediction == 1 else "No"
                        st.write(f"Rain Prediction: {rain_status}")
                    except Exception as e:
                        st.error(f"Rain prediction failed: {e}")
                else:
                    st.warning("Insufficient data for rain prediction.")
            else:
                st.warning("Historical data is empty.")

            # Prepare data for future predictions
            x_temp, y_temp = prepare_regression_data(historical_data, "Temp")
            x_hum, y_hum = prepare_regression_data(historical_data, "Humidity")

            st.success("Historical data processed successfully!")

            # Predict future weather
            future_temp = predict_future(temp_model, current_weather["temp_min"])
            future_hum = predict_future(hum_model, current_weather["humidity"])

            # Display future predictions
            st.write("\nFuture Temp:")
            for i, temp in enumerate(future_temp):
                st.write(f"Hour {i + 1}: {round(temp, 1)}C")

            st.write("\nFuture Humidity:")
            for i, humidity in enumerate(future_hum):
                st.write(f"Hour {i + 1}: {round(humidity, 1)}%")

        except Exception as e:
            st.error(f"An error occurred while processing historical data: {str(e)}")

if __name__ == "__main__":
    main()
