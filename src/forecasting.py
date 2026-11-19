"""
Shared forecasting tools, so every notebook uses identical definitions:
feature lists, date ranges, data loading, scoring and walk-forward validation.
"""
import numpy as np
import pandas as pd

# Feature groups
LAGS = ["lag_1d", "lag_2d", "lag_7d", "tp_mean_7d", "prev_day_mean", "prev_day_max"]
CALENDAR = ["tp", "dow", "month", "is_weekend", "is_holiday"]
WEATHER = ["akl_temp", "mnw_wind"]
FEATURES = LAGS + CALENDAR + WEATHER
FEATURES_NO_WEATHER = LAGS + CALENDAR

# Date ranges (inclusive months)
VAL_START, VAL_END = "2024-07", "2025-06"     # validation year, used to compare and tune models
TEST_START, TEST_END = "2025-07", "2026-06"   # test year. Only touched once, at the very end


def load_model_data(path="../data/processed/features.parquet"):
    """Load the feature matrix, keeping only rows where every feature exists,
    so all models are scored on identical rows."""
    feats = pd.read_parquet(path)
    return feats.dropna(subset=FEATURES).reset_index(drop=True)


def evaluate(actual, pred, dates):
    """Score one model: MAE, RMSE, and average within-day Spearman rank correlation."""
    err = actual - pred
    table = pd.DataFrame({"actual": actual, "pred": pred, "date": dates})
    # For each day: correlation between the RANKS of actual and forecast prices, then average over days
    spearman = (table.groupby("date")
                     .apply(lambda d: d["actual"].corr(d["pred"], method="spearman"), include_groups=False)
                     .mean())
    return pd.Series({"MAE": err.abs().mean(),
                      "RMSE": np.sqrt((err ** 2).mean()),
                      "Spearman": spearman})


def walk_forward(df, fit_predict, start, end):
    """Expanding-window walk-forward: at the start of each month from `start` to `end`,
    train on all rows before that month, then predict that month.

    fit_predict(train, test) must return one prediction per row of `test`.
    """
    def one_month(month):
        train = df[df["date"] < month.start_time]               # everything before this month
        test = df[df["date"].dt.to_period("M") == month]        # just this month
        result = pd.Series(fit_predict(train, test), index=test.index)
        print(f"  {month.strftime('%m/%Y')} done")              # progress update
        return result
    # map() runs one_month for each month; concat() joins the predictions
    return pd.concat(map(one_month, pd.period_range(start, end, freq="M")))

def with_forecast_weather(fit_predict):
    """Wrap any fit_predict so it TRAINS on actual weather but PREDICTS using the
    day-ahead weather forecasts, as it would have to in real use."""
    def wrapped(train, test):
        test_with_forecasts = test.assign(akl_temp=test["akl_temp_fc"], mnw_wind=test["mnw_wind_fc"])
        return fit_predict(train, test_with_forecasts)
    return wrapped    