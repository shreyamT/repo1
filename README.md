# TheWeather APP

A beginner-level Machine Learning project designed for weather forecasting and prediction. The application uses historical weather data to predict rain (classification) and future temperature/humidity trends (regression).

## Features

- **Current Weather**: Fetches real-time data using the OpenWeatherMap API.
- **Rain Prediction**: Predicts if it will rain tomorrow based on historical patterns.
- **Future Trends**: Provides a 5-hour forecast for temperature and humidity.
- **Interactive UI**: Built with Streamlit for a seamless web experience.

## Getting Started

### Prerequisites
Get you own API key
- Python 3.8+
- [OpenWeatherMap API Key](https://openweathermap.org/api)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shreyamT/TheWeather-.git
   cd TheWeather-
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Open `.env` and replace `your_api_key_here` with your actual OpenWeatherMap API key.

### Running the Application

To start the Streamlit web interface:
```bash
streamlit run app.py
```

### Running Tests

To verify the core ML logic (works even without an API key):
```bash
python tests/verify_offline.py
```

To run unit tests:
```bash
pytest tests/test_weather_logic.py
```

## Project Structure

- `app.py`: The main Streamlit web application.
- `weather1.py`: Core logic for data fetching, preprocessing, and model training.
- `model_trainer.py`: Script to train and save models.
- `tests/`: Contains unit tests and verification scripts.
- `weather.csv`: Historical dataset used for training.
