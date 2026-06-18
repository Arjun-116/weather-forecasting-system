import pandas as pd

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

lags = [1, 2, 3, 7, 14, 30]

for col in weather_cols:
    for lag in lags:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

df['temp_7day_avg'] = (df['temperature_2m_mean (°C)'].rolling(7).mean())

df['temp_30day_avg'] = (df['temperature_2m_mean (°C)'].rolling(30).mean())

df = df.dropna()

import numpy as np

df["month"] = df['time'].dt.month
df['dayofyear'] = df['time'].dt.dayofyear
df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)

df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
X = df.drop(columns=['time','temperature_2m_mean (°C)','temperature_2m_max (°C)','temperature_2m_min (°C)','rain_sum (mm)','wind_speed_10m_max (km/h)', 'month','dayofyear'])
y= df['temperature_2m_mean (°C)']
split= int(len(df)*0.8)
X_train= X[:split]
X_test= X[split:]
y_train= y[:split]
y_test= y[split:]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
model = LinearRegression()

model.fit(X_train, y_train)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = root_mean_squared_error(y_test, preds)

print("MAE:", mae)
print("RMSE:", rmse)   

best_model = LinearRegression()
best_model.fit(X_train, y_train)

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": best_model.coef_
})

print(
    coef_df.sort_values(
        by="Coefficient",
        key=lambda s: abs(s),
        ascending=False
    )
)
import joblib

joblib.dump(best_model, "temperature_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("Model saved!")