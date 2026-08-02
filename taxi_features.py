import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
CATEGORICAL_FEATURES = ["pickup_dayofweek", "pickup_month", "pickup_hour", "passenger_count"]
DROP_COLS = ["id", "pickup_datetime"]


def extract_datetime_features(df, datetime_cols):
    df = df.copy()
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        prefix = col.replace("_datetime", "") if "_datetime" in col else col

        df[f"{prefix}_dayofweek"] = df[col].dt.dayofweek
        df[f"{prefix}_month"] = df[col].dt.month
        df[f"{prefix}_hour"] = df[col].dt.hour
        df[f"{prefix}_dayofyear"] = df[col].dt.dayofyear
    return df


def haversine_miles(lat1, lat2, lon1, lon2):
    R = 3958.8
    lat1, lat2, lon1, lon2 = map(np.radians, [lat1, lat2, lon1, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c * R


def engineer_features(df):
    df = extract_datetime_features(df, ["pickup_datetime"])
    df["distance"] = haversine_miles(
        df["pickup_latitude"], df["dropoff_latitude"],
        df["pickup_longitude"], df["dropoff_longitude"],
    )
    df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)
    return df


def prepare_model_frame(df, fill_medians=None):
    df = df.drop(columns=DROP_COLS).drop_duplicates()
    medians = fill_medians if fill_medians is not None else df.median(numeric_only=True)
    df = df.fillna(medians)
    df["distance"] = np.log1p(df["distance"])
    return df, medians


def split_features_and_target(df):
    X = df.drop(columns=["trip_duration"])
    y = np.log1p(df["trip_duration"])
    return X, y
