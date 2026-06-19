"""
Keeps data/Palakkad data1.csv up to date by fetching any missing days
from the Open-Meteo API and appending them.

Why Open-Meteo: the CSV's column names (temperature_2m_max (°C), etc.)
and the coordinates in its header (10.790861, 76.6313) are an exact match
for Open-Meteo's daily weather API, which is free and needs no API key.
This script just continues the same dataset rather than switching sources.

Usage:
    python update_data.py            # fetch + append, print summary
    from update_data import update_csv; update_csv()   # use from app.py
"""

import io

import pandas as pd
import requests

CSV_PATH = "data/Palakkad data1.csv"
LATITUDE = 10.790861
LONGITUDE = 76.6313

# Must match the order of weather_cols used everywhere else in the project
API_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "rain_sum",
    "wind_speed_10m_max",
    "temperature_2m_mean",
]

CSV_COLUMNS = [
    "time",
    "temperature_2m_max (°C)",
    "temperature_2m_min (°C)",
    "rain_sum (mm)",
    "wind_speed_10m_max (km/h)",
    "temperature_2m_mean (°C)",
]


def _read_preamble_and_data():
    """Return (preamble_lines, dataframe) where preamble_lines are the first
    3 lines of the CSV (lat/lon header, lat/lon values, blank line) kept
    as-is, and dataframe is the parsed weather data."""
    with open(CSV_PATH, encoding="utf-8") as f:
        all_lines = f.readlines()
    preamble_lines = all_lines[:3]

    df = pd.read_csv(CSV_PATH, skiprows=2, on_bad_lines="skip")
    df["time"] = pd.to_datetime(df["time"])
    return preamble_lines, df


def _fetch_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Call Open-Meteo for [start_date, end_date] inclusive (YYYY-MM-DD) and
    return a dataframe with the same column names/order as the CSV."""
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "daily": ",".join(API_DAILY_VARS),
            "timezone": "GMT",
            "start_date": start_date,
            "end_date": end_date,
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()

    daily = payload.get("daily")
    if not daily or not daily.get("time"):
        return pd.DataFrame(columns=CSV_COLUMNS)

    new_df = pd.DataFrame(
        {
            "time": daily["time"],
            "temperature_2m_max (°C)": daily["temperature_2m_max"],
            "temperature_2m_min (°C)": daily["temperature_2m_min"],
            "rain_sum (mm)": daily["rain_sum"],
            "wind_speed_10m_max (km/h)": daily["wind_speed_10m_max"],
            "temperature_2m_mean (°C)": daily["temperature_2m_mean"],
        }
    )
    new_df["time"] = pd.to_datetime(new_df["time"])
    return new_df


def update_csv(today: pd.Timestamp | None = None) -> dict:
    """Fetch any days missing between the CSV's last entry and yesterday,
    append them, and rewrite the CSV. Returns a small summary dict so
    callers (e.g. the Streamlit app) can show a status message.

    We only fetch up to *yesterday* (not today) because today's reading
    is incomplete until the day finishes, and the project's feature
    engineering assumes one full day's mean/max/min/rain/wind per row.
    """
    today = today or pd.Timestamp.now("UTC").normalize().tz_localize(None)
    yesterday = today - pd.Timedelta(days=1)

    preamble_lines, existing_df = _read_preamble_and_data()
    last_date = existing_df["time"].max()

    if last_date >= yesterday:
        return {
            "status": "up_to_date",
            "last_date": last_date.date(),
            "rows_added": 0,
        }

    start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = yesterday.strftime("%Y-%m-%d")

    try:
        new_df = _fetch_range(start_date, end_date)
    except requests.RequestException as e:
        return {
            "status": "error",
            "last_date": last_date.date(),
            "rows_added": 0,
            "error": str(e),
        }

    if new_df.empty:
        return {
            "status": "no_new_data",
            "last_date": last_date.date(),
            "rows_added": 0,
        }

    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="time", keep="last")
    combined = combined.sort_values("time").reset_index(drop=True)

    _write_csv(preamble_lines, combined)

    return {
        "status": "updated",
        "last_date": combined["time"].max().date(),
        "rows_added": len(new_df),
    }


def _write_csv(preamble_lines: list, df: pd.DataFrame) -> None:
    out = df.copy()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d")
    out["temperature_2m_max (°C)"] = out["temperature_2m_max (°C)"].round(1)
    out["temperature_2m_min (°C)"] = out["temperature_2m_min (°C)"].round(1)
    out["rain_sum (mm)"] = out["rain_sum (mm)"].round(2)
    out["wind_speed_10m_max (km/h)"] = out["wind_speed_10m_max (km/h)"].round(1)
    out["temperature_2m_mean (°C)"] = out["temperature_2m_mean (°C)"].round(1)

    buf = io.StringIO()
    out.to_csv(buf, index=False, columns=CSV_COLUMNS)

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        f.writelines(preamble_lines)
        f.write(buf.getvalue())


if __name__ == "__main__":
    result = update_csv()
    print(result)
