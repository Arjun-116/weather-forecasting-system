import pandas as pd
import joblib

from forecast_utils import recursive_forecast

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

forecast_df = recursive_forecast(df, model, feature_names, horizon=7)

print("\n7-Day Forecast\n")
print(forecast_df)
