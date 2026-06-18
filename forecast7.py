import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# Load model
model = joblib.load("models/temperature_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Load data
df = pd.read_csv(
    "data/Palakkad data1.csv",
    skiprows=2,
    on_bad_lines="skip"
)

df["time"] = pd.to_datetime(df["time"])

weather_cols = [
    'temperature_2m_max (°C)',
    'temperature_2m_min (°C)',
    'rain_sum (mm)',
    'wind_speed_10m_max (km/h)',
    'temperature_2m_mean (°C)'
]

# Create lag features
lags = [1, 2, 3, 7, 14, 30]

for col in weather_cols:
    for lag in lags:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# Rolling averages
df['temp_7day_avg'] = (
    df['temperature_2m_mean (°C)']
    .rolling(7)
    .mean()
)

df['temp_30day_avg'] = (
    df['temperature_2m_mean (°C)']
    .rolling(30)
    .mean()
)

# Seasonal features
df["month"] = df["time"].dt.month
df["dayofyear"] = df["time"].dt.dayofyear

df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df = df.dropna()

# Last available day
latest = df.iloc[-1]

future_predictions = []

for i in range(1, 8):

    X_new = pd.DataFrame(
        [latest[feature_names].values],
        columns=feature_names
    )

    pred = model.predict(X_new)[0]

    forecast_date = latest["time"] + timedelta(days=i)

    future_predictions.append(
        [forecast_date.date(), round(pred, 2)]
    )

forecast_df = pd.DataFrame(
    future_predictions,
    columns=["Date", "Predicted Temperature"]
)

print("\n7-Day Forecast\n")
print(forecast_df)