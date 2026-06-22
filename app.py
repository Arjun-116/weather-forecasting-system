import streamlit as st
import pandas as pd
import joblib

from forecast_utils import recursive_forecast, build_tomorrow_features
from update_data import update_csv

st.set_page_config(page_title="Weather Forecasting System")

st.title("Weather Forecasting System")
st.write("Machine Learning Based Temperature Prediction")

mean_model = joblib.load("models/temperature_model.pkl")
max_model = joblib.load("models/max_temperature_model.pkl")
min_model = joblib.load("models/min_temperature_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")


@st.cache_data(ttl=3600)  # re-check for new data at most once an hour
def get_fresh_data():
    result = update_csv()
    df = pd.read_csv("data/Palakkad data1.csv", skiprows=2, on_bad_lines="skip")
    df["time"] = pd.to_datetime(df["time"])
    return df, result


col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Refresh data"):
        st.cache_data.clear()

df, update_result = get_fresh_data()

with col1:
    if update_result["status"] == "updated":
        st.success(
            f"Fetched {update_result['rows_added']} new day(s). "
            f"Data now current through {update_result['last_date']}."
        )
    elif update_result["status"] == "up_to_date":
        st.info(f"Data is already current through {update_result['last_date']}.")
    elif update_result["status"] == "no_new_data":
        st.warning(
            f"No new data available yet from the weather API "
            f"(still showing through {update_result['last_date']})."
        )
    elif update_result["status"] == "error":
        st.error(
            f"Couldn't reach the weather API ({update_result['error']}). "
            f"Showing existing data through {update_result['last_date']}."
        )

forecast_df = recursive_forecast(df, mean_model, feature_names, horizon=7)
X_new = build_tomorrow_features(df, feature_names)

mean_prediction = mean_model.predict(X_new)[0]
max_prediction = max_model.predict(X_new)[0]
min_prediction = min_model.predict(X_new)[0]

next_date = df["time"].iloc[-1].date() + pd.Timedelta(days=1)

st.subheader(f"Forecast for {next_date}")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Mean Temperature",f"{mean_prediction:.2f} °C")

with col2:
    st.metric("Maximum Temperature",f"{max_prediction:.2f} °C")

with col3:
    st.metric("Minimum Temperature",f"{min_prediction:.2f} °C")

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