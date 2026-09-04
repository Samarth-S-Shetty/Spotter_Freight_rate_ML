"""
Feature engineering for the freight rate prediction task.
Shared between training and inference so train/val/test all get identical transforms,
and medians used for imputation are always fit on TRAIN only (no leakage).
"""
import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points (vectorized)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def fit_medians(df: pd.DataFrame) -> dict:
    """Fit imputation medians on a training set only."""
    medians = {}
    for col in ["weight", "market_index"]:
        if col in df.columns:
            medians[col] = df[col].median()
    return medians


def build_feature_frame(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """Turn raw load rows into the final feature matrix used by the model."""
    df = df.copy()
    has_latlon = "pickup_lat" in df.columns

    # missing-value flags computed BEFORE imputation, then fill with train medians
    # (market_index/quote_signal aren't present in the December-style forecast inputs,
    # since those are live market signals unavailable ahead of time -- handled by the
    # caller selecting a reduced column set, not by faking these columns here)
    for col in ["weight", "market_index"]:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(medians[col])

    # date-derived seasonality
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # geography: straight-line distance as a supplement to the provided routed `distance`
    if has_latlon:
        df["haversine_km"] = haversine_km(
            df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"]
        )
        df["distance_ratio"] = df["distance"] / df["haversine_km"].replace(0, np.nan)
        df["distance_ratio"] = df["distance_ratio"].fillna(1.0)

    # weight per mile (density-ish signal)
    df["weight_per_mile"] = df["weight"] / df["distance"].replace(0, np.nan)
    df["weight_per_mile"] = df["weight_per_mile"].fillna(0)

    feature_cols = [
        "distance", "weight",
        "month_sin", "month_cos", "day_of_week",
        "weight_missing", "weight_per_mile",
    ]
    if "market_index" in df.columns:
        feature_cols += ["market_index", "market_index_missing"]
    if "quote_signal" in df.columns:
        feature_cols += ["quote_signal"]
    if has_latlon:
        feature_cols += ["haversine_km", "distance_ratio"]

    X = df[feature_cols].copy()
    X["equipment"] = df["equipment"].astype("category")
    return X