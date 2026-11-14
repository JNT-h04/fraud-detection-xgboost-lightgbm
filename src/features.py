"""Cleaning and feature engineering for the ULB credit-card fraud dataset.

The dataset has 28 anonymised PCA components (V1..V28), the seconds since the
first transaction (Time) and the amount. It has no card or customer ID, so
per-card velocity features are impossible. What we can engineer honestly:

* time-of-day features (the data covers two days; fraud skews to night hours)
* amount transforms (log, round-number flags, zero-amount "card test" flag)
* global activity windows: how busy the stream was just before a transaction,
  and how this amount compares with recent amounts

Every rolling feature looks strictly backwards in time, so nothing leaks from
the future into a row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    n0 = len(df)
    df = df.drop_duplicates().sort_values("Time", kind="stable").reset_index(drop=True)
    df.attrs["duplicates_dropped"] = n0 - len(df)
    assert df.isna().sum().sum() == 0, "unexpected missing values"
    return df


def _backward_window_stats(t: np.ndarray, amount: np.ndarray, window: float):
    """Count and mean amount of transactions in (t - window, t), excluding the row itself
    and any row with the same timestamp (we cannot know their order)."""
    left = np.searchsorted(t, t - window, side="right")
    right = np.searchsorted(t, t, side="left")  # first row with this exact timestamp
    count = right - left
    csum = np.concatenate([[0.0], np.cumsum(amount)])
    total = csum[right] - csum[left]
    mean = np.divide(total, count, out=np.full(len(t), np.nan), where=count > 0)
    return count, mean


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t = out["Time"].to_numpy(dtype=float)
    amt = out["Amount"].to_numpy(dtype=float)

    # time of day (Time is seconds from the first transaction; the period starts near midnight)
    hour = (t % 86400) / 3600
    out["hour"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["is_night"] = ((hour < 6)).astype(int)

    # amount
    out["log_amount"] = np.log1p(amt)
    out["amount_zero"] = (amt == 0).astype(int)
    out["amount_small"] = (amt <= 1).astype(int)  # card-testing style micro charges
    out["amount_round"] = ((amt > 0) & (np.mod(amt, 1) == 0)).astype(int)

    # global stream activity just before each transaction
    for name, w in [("1m", 60), ("10m", 600), ("1h", 3600)]:
        cnt, mean = _backward_window_stats(t, amt, w)
        out[f"txn_count_{name}"] = cnt
        out[f"amount_mean_{name}"] = mean
    out["amount_vs_1h_mean"] = np.log1p(amt) - np.log1p(out["amount_mean_1h"].fillna(amt.mean()))
    # same-second bursts (several transactions sharing one timestamp)
    out["same_second_count"] = out.groupby("Time")["Time"].transform("size") - 1
    return out


ENGINEERED_FEATURES = [
    "hour_sin", "hour_cos", "is_night",
    "log_amount", "amount_zero", "amount_small", "amount_round",
    "txn_count_1m", "txn_count_10m", "txn_count_1h",
    "amount_mean_1m", "amount_mean_10m", "amount_mean_1h",
    "amount_vs_1h_mean", "same_second_count",
]


def time_split(df: pd.DataFrame, train=0.6, val=0.2):
    """Chronological split: train on the past, tune on the next slice, test on the latest.
    A random split would let the model see transactions from the same fraud bursts on both sides."""
    n = len(df)
    a, b = int(n * train), int(n * (train + val))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]
