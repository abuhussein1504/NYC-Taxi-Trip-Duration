# NYC Taxi Trip Duration

Predicting the duration of NYC taxi trips, based on the [Kaggle NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) competition. Data for this project was provided pre-split (train/val/test) and pre-cleaned by the course instructor rather than downloaded directly from Kaggle.

## Project structure

```
.
├── NYC_Taxi_Trip_Duration_EDA.ipynb   # Exploratory data analysis notebook
├── taxi_features.py                   # Shared feature engineering (used by both scripts below)
├── train_model.py                     # Trains and saves the model
├── predict_model.py                   # Loads the saved model and scores/predicts on a CSV
└── README.md
```

## Setup

```bash
pip install numpy pandas scikit-learn seaborn matplotlib cartopy joblib
```

`cartopy` is only required for the map visualization in the EDA notebook — everything else runs without it.

## Data

This project expects three CSV files (train/validation/test), each with the standard NYC Taxi Trip Duration columns (`id`, `vendor_id`, `pickup_datetime`, `dropoff_datetime`, `passenger_count`, pickup/dropoff coordinates, `store_and_fwd_flag`, `trip_duration`). Update the path constants at the top of `train_model.py` and `predict_model.py` (or pass a path as a command-line argument to `predict_model.py`) to point at your local copies.

**Note:** the real, official test set is only evaluated once, at the very end, after all feature and model decisions are finalized — see [Methodology](#methodology) below.

## Exploratory Data Analysis

`NYC_Taxi_Trip_Duration_EDA.ipynb` covers:
- Data quality checks (missing values, duplicates, geographic outliers)
- Vendor / passenger count / store-and-forward flag distributions
- Correlation between numeric features and trip duration
- Pickup/dropoff location maps colored by trip duration
- Trip duration distribution (raw and log-transformed)
- Trip duration, distance, and speed patterns by day of week and hour
- Distance vs. duration relationship

Each figure is followed by a short written interpretation, and the notebook closes with a summary of key findings.

## Features

Engineered in `taxi_features.py`, shared identically between training and prediction:

| Feature | Description |
|---|---|
| `pickup_dayofweek`, `pickup_month`, `pickup_hour` | Extracted from `pickup_datetime` |
| `pickup_is_peak_hour` | Whether pickup falls in a weekday rush-hour window (7-9am, 4-6pm) |
| `distance` | Manhattan-style distance (sum of latitude-only and longitude-only haversine legs), in km |
| `pickup_jfk_dist`, `dropoff_jfk_dist` | Haversine distance (km) from pickup/dropoff to JFK airport |
| `pickup_lga_dist`, `dropoff_lga_dist` | Haversine distance (km) from pickup/dropoff to LaGuardia airport |
| `delta_lat`, `delta_lon` | Raw latitude/longitude difference between pickup and dropoff |
| `bearing_sin`, `bearing_cos` | Sine/cosine of the initial compass bearing from pickup to dropoff (encodes direction without the 0°/360° wraparound discontinuity) |
| `store_and_fwd_flag` | Converted to a numeric 0/1 flag |

Categorical features (`pickup_dayofweek`, `pickup_month`, `pickup_hour`, `passenger_count`) are one-hot encoded; numeric features are standard-scaled. Both are handled inside a single `ColumnTransformer`, fit only on the training set and reused (never re-fit) on validation/test data to avoid leakage.

Target variable: `np.log1p(trip_duration)`.

## Model

Model choice and hyperparameter are fixed rather than tuned, per the assignment: **`Ridge(alpha=1)`**. The intent is to put engineering effort into features rather than hyperparameter search.

## Usage

**Train and save the model:**
```bash
python train_model.py
```
This fits the pipeline on `train.csv`, reports train/validation R², and saves the fitted pipeline (`ridge_trip_duration_model.joblib`) and the training-set medians used for missing-value imputation (`train_medians.joblib`).

**Score or predict on a CSV:**
```bash
python predict_model.py path/to/data.csv --output predictions.csv
```
Loads the saved model and applies the exact same feature engineering used at training time. If the input CSV has a `trip_duration` column, it also reports R². If not, it just writes out predictions.

## Results

| Split | R² (log1p trip_duration) |
|---|---|
| Train | 0.7569 |
| Validation | 0.7558 |
| Test | 0.7984 |

## Methodology

- **Train/validation/test split** comes entirely from the instructor-provided files — no data was downloaded from Kaggle directly.
- **No leakage**: all fitted preprocessing (imputation medians, one-hot categories, scaling) is fit only on the training set and reused unchanged on validation and test data.
- **Test set evaluated once**: the official test set is scored a single time, after the model and features are finalized, and that result is reported as-is rather than used to guide further tuning.

## Author

Built as part of an ML course project (NYC Taxi Trip Duration regression task).