import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
                    "pickup_jfk_dist", "dropoff_jfk_dist", "pickup_lga_dist", "dropoff_lga_dist"
                    ]
CATEGORICAL_FEATURES = ["pickup_dayofweek", "pickup_month", "pickup_hour", "passenger_count"]
DROP_COLS = ["id", "pickup_datetime"]

JFK = (40.6413, -73.7781)
LGA = (40.7769, -73.8740)


def extract_datetime_features(df, datetime_cols):
    df = df.copy()
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        prefix = col.replace("_datetime", "") if "_datetime" in col else col

        df[f"{prefix}_dayofweek"] = df[col].dt.dayofweek
        df[f"{prefix}_month"] = df[col].dt.month
        df[f"{prefix}_hour"] = df[col].dt.hour
        df[f"{prefix}_dayofyear"] = df[col].dt.dayofyear
        df[f"{prefix}_is_peak_hour"] = df[col].dt.hour.isin([7, 8, 9, 16, 17, 18])
    return df


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def calculateBearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y))


def add_airport_distances(df):
    df["pickup_jfk_dist"] = np.log1p(haversine(df["pickup_latitude"], df["pickup_longitude"], JFK[0], JFK[1]))
    df["dropoff_jfk_dist"] = np.log1p(haversine(df["dropoff_latitude"], df["dropoff_longitude"], JFK[0], JFK[1]))
    df["pickup_lga_dist"] = np.log1p(haversine(df["pickup_latitude"], df["pickup_longitude"], LGA[0], LGA[1]))
    df["dropoff_lga_dist"] = np.log1p(haversine(df["dropoff_latitude"], df["dropoff_longitude"], LGA[0], LGA[1]))
    return df


def engineer_features(df):
    df = extract_datetime_features(df, ["pickup_datetime"])
    df["distance"] = np.log1p((
        haversine(df["pickup_latitude"], df["pickup_longitude"], df["dropoff_latitude"], df["pickup_longitude"]) +
        haversine(df["dropoff_latitude"], df["pickup_longitude"], df["dropoff_latitude"], df["dropoff_longitude"])
    ))
    df["speed_kph"] = df["distance"] / (df["trip_duration"] / 3600).replace(0, np.nan)
    df = add_airport_distances(df)
    df["delta_lat"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["delta_lon"] = df["dropoff_longitude"] - df["pickup_longitude"]
    df["bearing"] = calculateBearing(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )
    df["bearing_sin"] = np.sin(np.radians(df["bearing"]))
    df["bearing_cos"] = np.cos(np.radians(df["bearing"]))
    df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)
    return df


def prepare_model_frame(df, fill_medians=None, drop_duplicates=True):
    df = df.drop(columns=DROP_COLS)
    if drop_duplicates:
        df = df.drop_duplicates()
    medians = fill_medians if fill_medians is not None else df.median(numeric_only=True)
    df = df.fillna(medians)
    return df, medians


def split_features_and_target(df):
    X = df.drop(columns=["trip_duration"])
    y = np.log1p(df["trip_duration"])
    return X, y
