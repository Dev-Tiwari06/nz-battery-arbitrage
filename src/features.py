"""
The aim here is to build the model's features from the merged half-hourly dataset.

Input:   data/processed/merged.parquet     (from build_dataset.py)
Output:  data/processed/features.parquet   (one row per trading period: target + features)

The target is price[D, t]: the price on day D in trading period t.
Every price feature uses only days before D, so the model never sees the answer.
Exception: the weather columns are day D's ACTUAL weather (a "perfect weather
forecast"). That's a known shortcut we test and fix on Day 3.

Run from the repo root:  python src/features.py
"""
import holidays
import pandas as pd

data = pd.read_parquet("data/processed/merged.parquet")

# 1. PRICE GRID 
# Build a day x trading-period price matrix (rows = days, columns = periods 1..48)
# Then reindex to EVERY calendar day: the 8 dropped days come back as empty rows,
# so "shift down 1 row" always means "go back exactly 1 day".
grid = data.pivot(index="date", columns="tp", values="price")
grid = grid.reindex(pd.date_range(grid.index.min(), grid.index.max(), freq="D"))
grid.index.name = "date"
past = grid.shift(1)    # row D now holds day D-1's prices: the latest info at forecast time


def to_long(wide, name):
    """Turn a day x trading-period grid back into a single column indexed by (date, tp)."""
    return (wide.reset_index()
                .melt(id_vars="date", var_name="tp", value_name=name)
                .set_index(["date", "tp"])[name])


# 2. PRICE FEATURES 
features = pd.concat([
    to_long(grid, "price"),                                  # TARGET: price on day D, period t
    to_long(past, "lag_1d"),                                 # same period, 1 day earlier
    to_long(grid.shift(2), "lag_2d"),                        # same period, 2 days earlier
    to_long(grid.shift(7), "lag_7d"),                        # same period, 1 week earlier
    to_long(past.rolling(7, min_periods=5).mean(), "tp_mean_7d"),   # same period, average of last 7 days
], axis=1)

# Whole-day summaries of day D-1, copied onto all 48 rows of day D
prev_day = pd.DataFrame({"prev_day_mean": past.mean(axis=1),
                         "prev_day_max": past.max(axis=1)})
features = features.join(prev_day, on="date")

# 3. CALENDAR FEATURES 
dates = features.index.get_level_values("date")
features["dow"] = dates.day_of_week                            # Monday = 0 ... Sunday = 6
features["month"] = dates.month
features["is_weekend"] = (features["dow"] >= 5).astype(int)
# NZ public holidays, including Auckland Anniversary Day (subdiv="AUK")
nz_holidays = holidays.NZ(subdiv="AUK", years=range(2023, 2027))
features["is_holiday"] = dates.isin(pd.to_datetime(list(nz_holidays.keys()))).astype(int)

# 4. WEATHER FEATURES 
# Line up the weather with each (date, tp); keep the real timestamp too
weather = data.set_index(["date", "tp"])[["akl_temp", "mnw_wind"]]
features = features.join(weather)
features = features.join(data.reset_index().set_index(["date", "tp"])["time_nz"])

# Day-ahead weather FORECASTS (only available from Jan 2024 onwards).
# Same steps as the actual weather in build_dataset.py: read via UTC, then hourly -> half-hourly.
# limit=1 fills only the single half-hour between two forecast hours, so the archive's
# gaps (Jan to Feb 2024) stay empty instead of being bridged with a straight line
forecast = pd.read_csv("data/raw/weather_forecast_hourly.csv")
forecast.index = pd.DatetimeIndex(pd.to_datetime(forecast.pop("time_nz"), utc=True)).tz_convert("Pacific/Auckland")
forecast_hh = (forecast.reindex(forecast.index.union(data.index))
                       .interpolate(method="time", limit=1)
                       .reindex(data.index))
forecast_hh.index = pd.MultiIndex.from_arrays([data["date"], data["tp"]])   # label by (date, tp)
features = features.join(forecast_hh)

# 5. TIDY AND SAVE 
# Drop rows with no target (the 8 empty days). Rows with missing FEATURES are kept:
# the modelling notebook will decide which rows to use.
features = features.dropna(subset=["price"]).reset_index()   # date and tp become normal columns
features["tp"] = features["tp"].astype(int)
features = features.sort_values(["date", "tp"], ignore_index=True)   # back into time order

print(f"Rows: {len(features):,}")
print("Missing values per column:\n" + features.isna().sum().to_string())
print("\nExample rows:\n" + features.iloc[2000:2003].to_string())

features.to_parquet("data/processed/features.parquet", index=False)
print("\nSaved to data/processed/features.parquet")