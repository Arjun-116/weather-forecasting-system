import streamlit as st
import pandas as pd
import joblib

from forecast_utils import recursive_forecast

st.set_page_config(page_title="Weather Forecasting System")

st.title("Weather Forecasting System")
st.write("Machine Learning Based Temperature Prediction")

model = joblib.load("models/temperature_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

df = pd.read_csv("data/Palakkad data1.csv", skiprows=2, on_bad_lines="skip")
df["time"] = pd.to_datetime(df["time"])

forecast_df = recursive_forecast(df, model, feature_names, horizon=7)

next_date = forecast_df["Date"].iloc[0]
prediction = forecast_df["Predicted Temperature (°C)"].iloc[0]
st.metric(f"Predicted Mean Temperature for {next_date}", f"{prediction:.2f} °C")

st.subheader("7-Day Forecast")
st.caption(
    "Each day's forecast is generated recursively from the previous day's "
    "prediction, so accuracy decreases the further out the forecast goes."
)
st.dataframe(forecast_df)

st.subheader("Recent Temperatures")
st.line_chart(df.set_index("time")["temperature_2m_mean (°C)"].tail(30))

st.subheader("Latest Weather Records")
latest_records = df[['time', 'temperature_2m_mean (°C)']].tail(10).copy()
latest_records['time'] = latest_records['time'].dt.date
st.dataframe(latest_records)