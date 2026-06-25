import streamlit as st
import pandas as pd
import joblib
import numpy as np

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

forecast_df = recursive_forecast(df, mean_model, feature_names, target_col='temperature_2m_mean (°C)', horizon=7)
max_forecast_df = recursive_forecast(df,max_model,feature_names, target_col='temperature_2m_max (°C)', horizon=7)

min_forecast_df = recursive_forecast(df,min_model,feature_names, target_col='temperature_2m_min (°C)', horizon=7)

full_forecast_df = pd.DataFrame({
    "Date": forecast_df["Date"],
    "Mean Temp (°C)": forecast_df["Predicted Temperature (°C)"],
    "Max Temp (°C)": max_forecast_df["Predicted Temperature (°C)"],
    "Min Temp (°C)": min_forecast_df["Predicted Temperature (°C)"]
})

X_new = build_tomorrow_features(df, feature_names)

mean_prediction = mean_model.predict(X_new)[0]
max_prediction = max_model.predict(X_new)[0]
min_prediction = min_model.predict(X_new)[0]

next_date = df["time"].iloc[-1].date() + pd.Timedelta(days=1)

st.subheader(f"Forecast for {next_date}")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Mean Temperature", f"{mean_prediction:.2f} °C")

with col2:
    st.metric("Maximum Temperature", f"{max_prediction:.2f} °C")

with col3:
    st.metric("Minimum Temperature", f"{min_prediction:.2f} °C")

with st.expander("📅 7-Day Forecast", expanded=True):

    st.caption(
        "Each day's forecast is generated recursively from the previous day's "
        "prediction, so accuracy decreases the further out the forecast goes."
    )

    st.dataframe(full_forecast_df)
    st.line_chart(
        full_forecast_df.set_index("Date")[
            ["Mean Temp (°C)",
             "Max Temp (°C)",
             "Min Temp (°C)"]
             ])

with st.expander("📊 Historical Forecast Validation", expanded=False):

    test_horizon = 7

    # Create all features exactly as during training

    feature_df = df.copy()

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
            feature_df[f'{col}_lag{lag}'] = feature_df[col].shift(lag)

    feature_df['temp_7day_avg'] = (
        feature_df['temperature_2m_mean (°C)'].rolling(7).mean()
    )

    feature_df['temp_30day_avg'] = (
        feature_df['temperature_2m_mean (°C)'].rolling(30).mean()
    )

    feature_df['max_7day_avg'] = (
        feature_df['temperature_2m_max (°C)'].rolling(7).mean()
    )

    feature_df['max_30day_avg'] = (
        feature_df['temperature_2m_max (°C)'].rolling(30).mean()
    )

    feature_df['min_7day_avg'] = (
        feature_df['temperature_2m_min (°C)'].rolling(7).mean()
    )

    feature_df['min_30day_avg'] = (
        feature_df['temperature_2m_min (°C)'].rolling(30).mean()
    )

    feature_df["month"] = feature_df['time'].dt.month
    feature_df["dayofyear"] = feature_df['time'].dt.dayofyear

    feature_df['day_sin'] = np.sin(
        2 * np.pi * feature_df['dayofyear'] / 365
    )

    feature_df['day_cos'] = np.cos(
        2 * np.pi * feature_df['dayofyear'] / 365
    )

    feature_df['month_sin'] = np.sin(
        2 * np.pi * feature_df['month'] / 12
    )

    feature_df['month_cos'] = np.cos(
        2 * np.pi * feature_df['month'] / 12
    )

    feature_df = feature_df.dropna()

    # Use the last 7 days for validation
    actual_df = feature_df.tail(test_horizon).copy()

    mean_preds = []
    max_preds = []
    min_preds = []

    # Predict each day using ACTUAL historical data
    for _, row in actual_df.iterrows():

        X_test = pd.DataFrame(
            [row[feature_names].values],
            columns=feature_names
        )

        mean_preds.append(
            round(mean_model.predict(X_test)[0], 2)
        )

        max_preds.append(
            round(max_model.predict(X_test)[0], 2)
        )

        min_preds.append(
            round(min_model.predict(X_test)[0], 2)
        )

    # Comparison table
    comparison_df = pd.DataFrame({
        "Date": actual_df["time"].dt.date.values,

        "Pred Mean (°C)": mean_preds,
        "Actual Mean (°C)":
            actual_df["temperature_2m_mean (°C)"].round(2).values,

        "Pred Max (°C)": max_preds,
        "Actual Max (°C)":
            actual_df["temperature_2m_max (°C)"].round(2).values,

        "Pred Min (°C)": min_preds,
        "Actual Min (°C)":
            actual_df["temperature_2m_min (°C)"].round(2).values
    })

    # Errors
    comparison_df["Mean Error"] = abs(
        comparison_df["Pred Mean (°C)"] -
        comparison_df["Actual Mean (°C)"]
    ).round(2)

    comparison_df["Max Error"] = abs(
        comparison_df["Pred Max (°C)"] -
        comparison_df["Actual Max (°C)"]
    ).round(2)

    comparison_df["Min Error"] = abs(
        comparison_df["Pred Min (°C)"] -
        comparison_df["Actual Min (°C)"]
    ).round(2)

    st.dataframe(comparison_df)

    st.subheader("Backtesting Accuracy")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Mean Temp MAE",
            f"{comparison_df['Mean Error'].mean():.2f} °C"
        )

    with col2:
        st.metric(
            "Max Temp MAE",
            f"{comparison_df['Max Error'].mean():.2f} °C"
        )

    with col3:
        st.metric(
            "Min Temp MAE",
            f"{comparison_df['Min Error'].mean():.2f} °C"
        )

st.subheader("📈 Recent Temperature Trends")
recent_df = df.set_index("time")[[
    "temperature_2m_mean (°C)",
    "temperature_2m_max (°C)",
    "temperature_2m_min (°C)"
]].tail(30)

recent_df = recent_df.rename(columns={
    "temperature_2m_mean (°C)": "Mean Temperature",
    "temperature_2m_max (°C)": "Maximum Temperature",
    "temperature_2m_min (°C)": "Minimum Temperature"
})

st.line_chart(recent_df)
st.subheader("Latest Weather Records")
latest_records = df[[
    'time',
    'temperature_2m_mean (°C)',
    'temperature_2m_max (°C)',
    'temperature_2m_min (°C)']].tail(10).copy()
latest_records['time'] = latest_records['time'].dt.date
latest_records = latest_records.rename(columns={
    'time': 'Date',
    'temperature_2m_mean (°C)': 'Mean Temp (°C)',
    'temperature_2m_max (°C)': 'Max Temp (°C)',
    'temperature_2m_min (°C)': 'Min Temp (°C)'
})
with st.expander("Latest Weather Records"):
    st.dataframe(latest_records)