"""
Clean the raw price and weather data and merge them into one half-hourly table.

Inputs:  data/raw/prices_mt_roskill.csv   EMI prices, half-hourly, two Mt Roskill nodes
         data/raw/weather_hourly.csv      Open-Meteo weather, hourly, NZ time
Output:  data/processed/merged.parquet    one row per trading period, node ROS1101

Run from the repo root:  python src/build_dataset.py
"""
import pandas as pd

NODE = "ROS1101"                        # Mt Roskill 110 kV: the node the model uses
OUT = "data/processed/merged.parquet"

# ============================== 1. PRICES ==============================
raw = pd.read_csv("data/raw/prices_mt_roskill.csv", parse_dates=["TradingDate"])

# --- Quick check: how different are the two Mt Roskill nodes? ---
# pivot_table puts each node in its own column so they line up row by row
both = raw.pivot_table(index=["TradingDate", "TradingPeriod"],
                       columns="PointOfConnection", values="DollarsPerMegawattHour")
gap = (both["ROS0221"] - both["ROS1101"]).abs()
print(f"ROS0221 vs ROS1101: mean gap ${gap.mean():.2f}/MWh, max gap ${gap.max():.2f}/MWh")

# --- Keep only our node, with short column names ---
prices = (raw[raw["PointOfConnection"] == NODE]
          .rename(columns={"TradingDate": "date", "TradingPeriod": "tp",
                           "DollarsPerMegawattHour": "price"})
          [["date", "tp", "price"]])

# --- Drop daylight-saving change days ---
# A normal day has 48 trading periods; the DST change days have 46 or 50.
# transform("size") writes each day's period count onto every row of that day
prices["n_tp"] = prices.groupby("date")["tp"].transform("size")
dst_days = pd.DatetimeIndex(prices.loc[prices["n_tp"] != 48, "date"].unique())
print(f"\nDropping {len(dst_days)} daylight-saving days: {dst_days.strftime('%d/%m/%Y').tolist()}")
prices = prices[prices["n_tp"] == 48].drop(columns="n_tp")

# --- Timestamp for the start of each trading period, in NZ time ---
# Period 1 starts at 00:00, period 2 at 00:30, ..., period 48 at 23:30
midnight = prices["date"].dt.tz_localize("Pacific/Auckland")
prices["time_nz"] = midnight + pd.to_timedelta((prices["tp"] - 1) * 30, unit="min")
prices = prices.set_index("time_nz").sort_index()

# ============================== 2. WEATHER ==============================
weather = pd.read_csv("data/raw/weather_hourly.csv")
# The file mixes +13:00 (summer) and +12:00 (winter) offsets, so read the
# times as UTC first, then convert back to NZ time
weather.index = pd.DatetimeIndex(pd.to_datetime(weather.pop("time_nz"), utc=True)).tz_convert("Pacific/Auckland")

# --- Hourly -> half-hourly ---
# 1. add the half-hour timestamps from the price data (they start out empty)
# 2. fill them by straight-line interpolation between the neighbouring hours
# 3. keep only the timestamps that match a trading period
all_times = weather.index.union(prices.index)
weather_hh = weather.reindex(all_times).interpolate(method="time").reindex(prices.index)

# ============================== 3. MERGE ==============================
# join lines the two tables up on their shared time index
data = prices.join(weather_hh)

# ============================== 4. CHECK AND SAVE ==============================
print(f"\nRows: {len(data):,}  ({data['date'].nunique():,} days x 48 periods)")
print(f"From {data.index.min()} to {data.index.max()}")
print("Missing values per column:\n" + data.isna().sum().to_string())
print("\nFirst rows:\n" + data.head(3).to_string())

data.to_parquet(OUT)
print(f"\nSaved to {OUT}")