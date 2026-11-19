"""
Download the day-ahead weather FORECASTS (made 24 hours in advance) from the Open-Meteo
API, for the same two series as download_weather.py.

Why should we do this? - the model's weather features use ACTUAL weather, which isn't known when
forecasting the day before. At prediction time we swap in these forecasts instead.
The Previous Runs API only has data from January 2024, which covers the
validation and test years.

Output: data/raw/weather_forecast_hourly.csv   (columns akl_temp_fc, mnw_wind_fc)

Run from the repo root:  python src/download_weather_forecasts.py
"""
import pandas as pd
import requests

# The Previous Runs data is served from one of these hosts; try the first, then the second
URLS = ["https://previous-runs-api.open-meteo.com/v1/forecast",
        "https://historical-forecast-api.open-meteo.com/v1/forecast"]
START, END = "2023-12-31", "2026-06-30"     # one day early, as times come back in UTC
OUT = "data/raw/weather_forecast_hourly.csv"

# column name: (latitude, longitude, Open-Meteo variable)
LOCATIONS = {
    "akl_temp_fc": (-36.91, 174.74, "temperature_2m"),    # Mt Roskill, degrees C
    "mnw_wind_fc": (-40.35, 175.75, "wind_speed_100m"),   # Manawatu wind farms, km/h
}


def get_json(params):
    """Request from the first host; if that fails, try the second."""
    try:
        r = requests.get(URLS[0], params=params, timeout=120)
        r.raise_for_status()
    except requests.RequestException:
        r = requests.get(URLS[1], params=params, timeout=120)
        r.raise_for_status()
    return r.json()


def fetch(name, spec):
    """Download the 1-day-ahead forecast of one variable at one location."""
    lat, lon, variable = spec
    forecast_var = f"{variable}_previous_day1"      # = the forecast made 24 hours earlier
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START, "end_date": END,
        "hourly": forecast_var,
        "timezone": "GMT",
    }
    hourly = get_json(params)["hourly"]
    print(f"{name}: downloaded {len(hourly['time']):,} hours")
    return pd.Series(hourly[forecast_var], index=pd.to_datetime(hourly["time"]), name=name)


forecasts = pd.concat(map(fetch, LOCATIONS.keys(), LOCATIONS.values()), axis=1)
forecasts.index = forecasts.index.tz_localize("UTC").tz_convert("Pacific/Auckland")
forecasts.index.name = "time_nz"

print("\nMissing values per column:\n" + forecasts.isna().sum().to_string())
forecasts.to_csv(OUT)
print(f"\nSaved {len(forecasts):,} hourly rows to {OUT}")
print(f"From {forecasts.index.min()} to {forecasts.index.max()}")