"""
Shared forecasting logic used by forecast7.py and app.py.

Implements a *recursive* multi-day forecast: each day's prediction is fed
back in as history so the next day's lag features are computed from the
previous day's forecast (not the same static row repeated 7 times).

Important limitation: the trained model only predicts temperature_2m_mean.
It does not predict max/min temperature, rain, or wind. For those columns
we cannot forecast forward, so we carry the last known actual value forward
(persistence assumption) purely so lag features stay defined. This means
day 1 lag1/2/3 for temperature_2m_mean come from real data, but by day 4+
the lag1/2/3 temperature_2m_mean values come from the model's own earlier
predictions, while max/min/rain/wind lag values stay frozen at their last
real reading. Treat predictions beyond ~2-3 days as increasingly rough.
"""

import numpy as np
import pandas as pd

WEATHER_COLS = [
    'temperature_2m_max (°C)',
    'temperature_2m_min (°C)',
    'rain_sum (mm)',
    'wind_speed_10m_max (km/h)',
    'temperature_2m_mean (°C)',
]

MEAN_TEMP_COL = 'temperature_2m_mean (°C)'

LAGS = [1, 2, 3, 7, 14, 30]


def load_history(df: pd.DataFrame) -> dict:
    """Turn the raw dataframe into a dict of plain python lists, one per
    weather column, in chronological order. Lists grow as we forecast."""
    return {col: df[col].tolist() for col in WEATHER_COLS}


def build_feature_row(history: dict, forecast_date: pd.Timestamp, feature_names: list) -> pd.DataFrame:
    """Build a single feature row matching the model's expected feature_names,
    using the most recent values in `history` for lag/rolling features and
    `forecast_date` for the seasonal sin/cos features."""
    feat = {}

    for col in WEATHER_COLS:
        series = history[col]
        for lag in LAGS:
            feat[f'{col}_lag{lag}'] = series[-lag]

    mean_series = history[MEAN_TEMP_COL]
    feat['temp_7day_avg'] = float(np.mean(mean_series[-7:]))
    feat['temp_30day_avg'] = float(np.mean(mean_series[-30:]))

    month = forecast_date.month
    dayofyear = forecast_date.dayofyear
    feat['day_sin'] = np.sin(2 * np.pi * dayofyear / 365)
    feat['day_cos'] = np.cos(2 * np.pi * dayofyear / 365)
    feat['month_sin'] = np.sin(2 * np.pi * month / 12)
    feat['month_cos'] = np.cos(2 * np.pi * month / 12)

    # Reindex to guarantee the exact column order the model was trained on
    return pd.DataFrame([feat])[feature_names]


def recursive_forecast(df: pd.DataFrame, model, feature_names: list, horizon: int = 7) -> pd.DataFrame:
    """Forecast `horizon` days ahead, recursively feeding each day's
    predicted mean temperature back into the lag features for the next day.

    df must already have a parsed datetime 'time' column and at least 30
    rows of real history (for the 30-day lag/rolling features).
    """
    history = load_history(df)
    last_date = df['time'].iloc[-1]

    results = []
    for i in range(1, horizon + 1):
        forecast_date = last_date + pd.Timedelta(days=i)

        X_new = build_feature_row(history, forecast_date, feature_names)
        pred = model.predict(X_new)[0]

        # Feed the prediction back in so tomorrow's lag1 = today's forecast
        history[MEAN_TEMP_COL].append(pred)

        # We can't forecast these, so persist the last known actual reading
        # purely to keep their lag features defined for later days.
        for col in WEATHER_COLS:
            if col != MEAN_TEMP_COL:
                history[col].append(history[col][-1])

        results.append([forecast_date.date(), round(float(pred), 2)])

    return pd.DataFrame(results, columns=['Date', 'Predicted Temperature (°C)'])

def build_tomorrow_features(df, feature_names):
    history = load_history(df)
    forecast_date = df["time"].iloc[-1] + pd.Timedelta(days=1)

    return build_feature_row(
        history,
        forecast_date,
        feature_names
    )