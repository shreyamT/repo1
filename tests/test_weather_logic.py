import os
import sys
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

# Add parent directory to path to import weather1
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weather1 import prepare_data, prepare_regression_data, predict_future

def test_prepare_data():
    # Create dummy data
    data = pd.DataFrame({
        'MinTemp': [10, 12, 11],
        'MaxTemp': [20, 22, 21],
        'WindGustDir': ['N', 'S', 'N'],
        'WindGustSpeed': [15, 20, 18],
        'Humidity': [50, 60, 55],
        'Pressure': [1010, 1012, 1011],
        'Temp': [15, 17, 16],
        'RainTomorrow': ['No', 'Yes', 'No']
    })
    
    x, y, le = prepare_data(data)
    
    assert x.shape == (3, 7)
    assert len(y) == 3
    assert 'WindGustDir' in x.columns
    # Check if encoding happened
    assert isinstance(x['WindGustDir'].iloc[0], (int, np.integer))
    assert isinstance(y.iloc[0], (int, np.integer))

def test_prepare_regression_data():
    data = pd.DataFrame({
        'Temp': [20, 21, 22, 23, 24]
    })
    
    x, y = prepare_regression_data(data, 'Temp')
    
    # Shifts data: [20, 21, 22, 23] -> x, [21, 22, 23, 24] -> y
    assert len(x) == 4
    assert len(y) == 4
    assert x[0][0] == 20
    assert y[0] == 21
    assert x[3][0] == 23
    assert y[3] == 24

def test_predict_future():
    # Mock model
    mock_model = MagicMock()
    # If input is [[x]], return [[x + 1]]
    mock_model.predict.side_effect = lambda x: np.array([x[0][0] + 1])
    
    current_value = 20
    predictions = predict_future(mock_model, current_value)
    
    assert len(predictions) == 5
    assert predictions[0] == 21
    assert predictions[4] == 25

def test_read_historical_data_local(tmp_path):
    from weather1 import read_historical_data
    
    # Create a dummy CSV file
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test_weather.csv"
    p.write_text("Date,Temp,Humidity\n2023-01-01,20,50\n2023-01-01,20,50\n2023-01-02,,60")
    
    # read_historical_data should drop duplicates and NaNs
    df = read_historical_data(str(p))
    
    # Original: 3 rows. Duplicate dropped -> 2. NaN in Temp dropped -> 1.
    assert len(df) == 1
    assert df['Temp'].iloc[0] == 20
