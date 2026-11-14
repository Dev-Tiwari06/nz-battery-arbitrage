"""
Download half-hourly final energy prices for selected Auckland grid nodes
from the Electricity Authority's EMI website, and save them as a single CSV.

Node kept: every Mt Roskill node (codes starting "ROS", e.g. the 110 kV and 22 kV supply points)

Run from the repo root:  python src/download_prices.py
"""
import pandas as pd

# Settings
# EMI folder holding one CSV per month (every node, every trading period)
BASE = "https://www.emi.ea.govt.nz/Wholesale/Datasets/DispatchAndPricing/FinalEnergyPrices/ByMonth"
OUT = "data/raw/prices_mt_roskill.csv"   # where the filtered data gets saved

# Every month we want, as pandas "periods" (Jan 2023, Feb 2023, ..., Jun 2026)
MONTHS = pd.period_range("2023-01", "2026-06", freq="M")


def load_month(month):
    """Download one month's file and return only the Mt Roskill rows."""
    # EMI's filenames need YYYYMM, e.g. 202301_FinalEnergyPrices.csv
    url = f"{BASE}/{month.strftime('%Y%m')}_FinalEnergyPrices.csv"
    try:
        df = pd.read_csv(url)                   # pandas reads a CSV straight from a web address
    except Exception:
        # One file has a different name (December 2025 ends in "_incomplete"),
        # so if the normal name fails, try that version instead
        df = pd.read_csv(url.replace(".csv", "_incomplete.csv"))
    # Progress update, with the month displayed as MM/YYYY
    print(f"{month.strftime('%m/%Y')}: downloaded {len(df):,} rows (all nodes)")
    # Keep a row only if its node code starts with "ROS" (Mt Roskill)
    return df[df["PointOfConnection"].str.startswith("ROS")]


# --- Download every month and stack them into one table -----------------
# map() applies load_month to each month; concat() stacks the results on top of each other
prices = pd.concat(map(load_month, MONTHS), ignore_index=True)

# --- Which node codes did we actually get? -------------------------------
print("\nNodes found:", sorted(prices["PointOfConnection"].unique()))

# --- Sanity check: rows per month per node, about 48 x (days in month) ---
prices["TradingDate"] = pd.to_datetime(prices["TradingDate"])
month = prices["TradingDate"].dt.to_period("M").dt.strftime("%m/%Y")   # MM/YYYY labels
# One row per month, one column per node, each cell = number of half-hours found
rows_per_month = prices.groupby([month, prices["PointOfConnection"]], sort=False).size().unstack()
print("\nRows per month:\n" + rows_per_month.to_string())

# --- Save ---------------------------------------------------------------
prices.to_csv(OUT, index=False)
print(f"\nSaved {len(prices):,} rows to {OUT}")