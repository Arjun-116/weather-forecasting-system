import pandas as pd
import numpy as np
import joblib

model = joblib.load("models/temperature_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

df= pd.read_csv('data/Palakkad data1.csv', skiprows=2, on_bad_lines="skip")
df["time"]= pd.to_datetime(df['time'])

weather_cols = ['temperature_2m_max (°C)', 'temperature_2m_min (°C)', 'rain_sum (mm)', 'wind_speed_10m_max (km/h)', 'temperature_2m_mean (°C)']

lags = [1, 2, 3, 7, 14, 30]

for col in weather_cols:
    for lag in lags:
         df[f'{col}_lag{lag}'] = df[col].shift(lag)

df['temp_7day_avg'] = (df['temperature_2m_mean (°C)'].rolling(7).mean())
df['temp_30day_avg'] = (df['temperature_2m_mean (°C)'].rolling(30).mean())

df["month"] = df['time'].dt.month
df['dayofyear'] = df['time'].dt.dayofyear

df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df= df.dropna()
latest = df.iloc[-1]

X_new= pd.DataFrame([latest[feature_names].values], columns= feature_names)

prediction = model.predict(X_new)[0]

last_date = latest["time"]

print(df[['time', 'temperature_2m_mean (°C)']].tail())

from datetime import timedelta
next_date = last_date + timedelta(days=1)
print(f"Predicted mean temperature for "f"{next_date.date()}: {prediction:.2f} °C")