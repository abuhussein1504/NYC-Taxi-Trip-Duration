import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from taxi_features import engineer_features, prepare_model_frame

MODEL_PATH = "ridge_trip_duration_model.joblib"
MEDIANS_PATH = "train_medians.joblib"


def score_csv(data_path):
    pipeline = joblib.load(MODEL_PATH)
    train_medians = joblib.load(MEDIANS_PATH)

    raw = pd.read_csv(data_path)
    engineered = engineer_features(raw)
    prepared, _ = prepare_model_frame(engineered, fill_medians=train_medians, drop_duplicates=False)

    has_labels = "trip_duration" in prepared.columns
    X = prepared.drop(columns=["trip_duration"]) if has_labels else prepared

    log_preds = pipeline.predict(X)
    preds = np.expm1(log_preds)

    output = pd.DataFrame({"id": raw["id"], "predicted_trip_duration": preds})

    if has_labels:
        y_true = np.log1p(prepared["trip_duration"])
        r2 = r2_score(y_true, log_preds)
        print(f"R² score (log1p trip_duration space): {r2:.4f}") # 0.6199
    else:
        print("No trip_duration column found - predictions only, no R² to report.")

    return output


def main():
    parser = argparse.ArgumentParser(description="Score the saved trip-duration model on a CSV file.")
    parser.add_argument("data_path", help="Path to the CSV to score (e.g. test.csv)")
    parser.add_argument("--output", default="predictions.csv", help="Where to write predictions")
    args = parser.parse_args()

    output = score_csv(args.data_path)
    output.to_csv(args.output, index=False)
    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
