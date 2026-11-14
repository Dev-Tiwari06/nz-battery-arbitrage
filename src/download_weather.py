"""
Download hourly historical weather from the Open-Meteo archive API for two
locations that drive Auckland electricity prices, and save it as one CSV.

  - Mt Roskill, Auckland: air temperature (drives heating demand)
  - Manawatu: wind speed at 100 m, roughly turbine hub height (drives wind generation)

Run from the repo root:  python src/download_weather.py
"""
import pandas as pd
import requests

# --- Settings -----------------------------------------------------------
URL = "https://archive-api.open-meteo.com/v1/archive"
# Times come back in UTC, and NZ is 12-13 hours ahead, so start one day early
# to make sure NZ local time is covered from midnight on 1 Jan 2023
START, END = "2022-12-31", "2026-06-30"
OUT = "data/raw/weather_hourly.csv"

# column name: (latitude, longitude, Open-Meteo variable)
LOCATIONS = {
    "akl_temp": (-36.91, 174.74, "temperature_2m"),    # Mt Roskill, degrees C
    "mnw_wind": (-40.35, 175.75, "wind_speed_100m"),   # Manawatu wind farms, km/h
}


def fetch(name, spec):
    """Download one variable at one location and return it as a time-indexed column."""
    lat, lon, variable = spec                  # unpack the tuple into three variables
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START, "end_date": END,
        "hourly": variable,
        "timezone": "GMT",   # UTC: Open-Meteo ignores daylight saving if given a local zone
    }
    r = requests.get(URL, params=params, timeout=120)   # requests builds the ?a=..&b=.. URL
    r.raise_for_status()                # stop with a clear error if the request failed
    hourly = r.json()["hourly"]         # JSON -> dict with a "time" list and a values list
    print(f"{name}: downloaded {len(hourly['time']):,} hours")
    # One column of values, indexed by time, named e.g. "akl_temp"
    return pd.Series(hourly[variable], index=pd.to_datetime(hourly["time"]), name=name)


# --- Download both locations and put them side by side ------------------
# map() runs fetch on each (name, spec) pair; concat(axis=1) joins them as columns
weather = pd.concat(map(fetch, LOCATIONS.keys(), LOCATIONS.values()), axis=1)

# --- Convert UTC to NZ time ---------------------------------------------
# Label the times as UTC, then convert; pandas handles NZST/NZDT switches correctly
weather.index = weather.index.tz_localize("UTC").tz_convert("Pacific/Auckland")
weather.index.name = "time_nz"

# --- Check and save -----------------------------------------------------
print("\nMissing values per column:\n" + weather.isna().sum().to_string())
weather.to_csv(OUT)
print(f"\nSaved {len(weather):,} hourly rows to {OUT}")
print(f"From {weather.index.min()} to {weather.index.max()}")