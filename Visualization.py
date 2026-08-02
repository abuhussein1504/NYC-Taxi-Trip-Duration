# NYC Taxi Trip Duration - Exploratory Data Analysis
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io import DownloadWarning
    HAS_CARTOPY = True
    warnings.filterwarnings('ignore', category=DownloadWarning, module='cartopy.io')
except ImportError:
    HAS_CARTOPY = False
    print("cartopy not installed.")


DATA_PATH = "/mnt/DATA/Courses/DR_Mostafa_Saad/My Tasks/ML Tasks/Project 1/nyc-taxi-trip-duration/train/train.csv"

DAY_NAMES = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday",
}

BOROUGHS = {
    "Manhattan": (-73.98, 40.75),
    "Brooklyn": (-73.95, 40.65),
    "Queens": (-73.77, 40.72),
    "Bronx": (-73.85, 40.85),
    "Staten Island": (-74.15, 40.58),
}

MAP_EXTENT = [-74.3, -73.4, 40.5, 41.1]
NYC_BBOX = {"lon": (-74.3, -73.4), "lat": (40.5, 41.1)}


# Data loading / feature engineering
def load_raw(path):
    return pd.read_csv(path, parse_dates=["pickup_datetime"])


def data_quality_report(df):
    df.info()
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum(), "\n")


def filter_geo_outliers(df, bbox=NYC_BBOX):
    outside = df[
        ~df["pickup_longitude"].between(*bbox["lon"]) |
        ~df["pickup_latitude"].between(*bbox["lat"]) |
        ~df["dropoff_longitude"].between(*bbox["lon"]) |
        ~df["dropoff_latitude"].between(*bbox["lat"])
    ]
    print(f"{len(outside)} rows ({len(outside) / len(df) * 100:.3f}%) "
          f"have coordinates outside the NYC area\n")
    return df.drop(outside.index)


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles between two lat/lon points."""
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def engineer_features(df):
    df = df.copy()

    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["distance"] = haversine_miles(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )

    ordered_days = pd.api.types.CategoricalDtype(categories=DAY_NAMES.values(), ordered=True)
    df["day_name"] = df["pickup_dayofweek"].map(DAY_NAMES).astype(ordered_days)

    return df


def prepare_viz_df(df, max_duration=3600):
    viz_df = df[df["trip_duration"] < max_duration].copy()
    viz_df["log_duration"] = np.log1p(viz_df["trip_duration"])
    viz_df["speed_mph"] = viz_df["distance"] / (viz_df["trip_duration"] / 3600).replace(0, np.nan)
    return viz_df


def duration_and_speed_diagnostics(df, viz_df):
    print(f"Trips >= 1hr: {(df['trip_duration'] >= 3600).sum()} "
          f"({(df['trip_duration'] >= 3600).mean() * 100:.2f}%)\n")

    implausible = viz_df[(viz_df["speed_mph"] > 80) | (viz_df["speed_mph"] == 0)]
    print(f"{len(implausible)} trips with implausible speed (0 or >80mph)")


# Plots
def plot_categorical_breakdowns(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    df["vendor_id"].value_counts().plot(kind="bar", ax=axes[0], title="Vendor ID")
    df["passenger_count"].value_counts().sort_index().plot(kind="bar", ax=axes[1], title="Passenger Count")
    df["store_and_fwd_flag"].value_counts().plot(kind="bar", ax=axes[2], title="Store and Fwd Flag")
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df):
    num_cols = ["trip_duration", "distance", "passenger_count", "pickup_hour", "pickup_dayofweek"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.show()


def plot_pickup_dropoff_map(viz_df):
    if not HAS_CARTOPY:
        return

    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection=projection)
    ax2 = fig.add_subplot(1, 2, 2, projection=projection)

    optional_features = [
        (cfeature.BORDERS, dict(linestyle=":", edgecolor="gray")),
        (cfeature.STATES, dict(edgecolor="gray")),
        (cfeature.RIVERS, dict(edgecolor="steelblue")),
        (cfeature.LAKES, dict(edgecolor="steelblue")),
    ]

    for ax in (ax1, ax2):
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightyellow")
        ax.add_feature(cfeature.OCEAN, facecolor="skyblue")
        ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.5)
        for feature, kwargs in optional_features:
            try:
                ax.add_feature(feature, **kwargs)
            except Exception as e:
                print(f"Skipping a map feature (likely no internet for shapefile download): {e}")

    sc1 = ax1.scatter(
        viz_df["pickup_longitude"], viz_df["pickup_latitude"],
        c=viz_df["log_duration"], cmap="viridis", s=2, alpha=0.6,
        transform=ccrs.PlateCarree(), zorder=1,
    )
    ax1.set_title("Pickup Locations Colored by Trip Duration (log scale)")

    sc2 = ax2.scatter(
        viz_df["dropoff_longitude"], viz_df["dropoff_latitude"],
        c=viz_df["log_duration"], cmap="viridis", s=2, alpha=0.6,
        transform=ccrs.PlateCarree(), zorder=1,
    )
    ax2.set_title("Dropoff Locations Colored by Trip Duration (log scale)")

    for ax in (ax1, ax2):
        for borough, (lon, lat) in BOROUGHS.items():
            ax.text(lon, lat, borough, fontsize=10, color="black", ha="center", va="center", zorder=2)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cbar = plt.colorbar(sc2, cax=cax)
    cbar.set_label("log(1 + Trip Duration)", rotation=270, labelpad=15)

    plt.suptitle("Pickup and Dropoff Locations Over Map (New York Area)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.show()


def plot_duration_histogram(viz_df):
    plt.figure(figsize=(10, 5))
    sns.histplot(viz_df["log_duration"], bins=100, kde=True, color="orange")
    plt.title("Log-Transformed Trip Duration")
    plt.xlabel("log(Trip Duration + 1)")
    plt.ylabel("Frequency")
    plt.show()


def plot_time_heatmaps(viz_df, i):
    metrics = [
        ("trip_duration", "Average Trip Duration by Day and Hour", "Avg Duration (sec)"),
        ("distance", "Average Trip Distance by Day and Hour", "Avg Distance (miles)"),
        ("speed_mph", "Average Trip Speed by Day and Hour", "Avg Speed (mph)"),
    ]

    value_col, title, cbar_label = metrics[i]

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    pivot = viz_df.pivot_table(
        index="day_name", columns="pickup_hour", values=value_col,
        aggfunc="mean", observed=True,
    )
    sns.heatmap(pivot, cmap="YlGnBu", annot=False, ax=ax, cbar_kws={"label": cbar_label})
    ax.set_title(title)
    ax.set_xlabel("Hour of Day (0 = midnight)")
    ax.set_ylabel("Day of Week")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.show()


def plot_distance_vs_duration(viz_df):
    plt.figure(figsize=(10, 6))
    plt.scatter(viz_df["distance"], viz_df["trip_duration"], alpha=0.6)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Trip Duration vs Distance (Log Scale)")
    plt.xlabel("Log Distance (miles)")
    plt.ylabel("Log Trip Duration (sec)")
    plt.grid(True)
    plt.show()


def main():
    train_raw = load_raw(DATA_PATH)

    data_quality_report(train_raw)
    train_clean = filter_geo_outliers(train_raw)

    train = engineer_features(train_clean)
    viz_df = prepare_viz_df(train)

    duration_and_speed_diagnostics(train, viz_df)

    plot_categorical_breakdowns(train)
    plot_correlation_matrix(train)
    plot_pickup_dropoff_map(viz_df)
    plot_duration_histogram(viz_df)
    plot_time_heatmaps(viz_df, 0)
    plot_time_heatmaps(viz_df, 1)
    plot_time_heatmaps(viz_df, 2)
    plot_distance_vs_duration(viz_df)


if __name__ == "__main__":
    main()