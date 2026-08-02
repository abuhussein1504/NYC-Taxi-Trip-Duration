import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from taxi_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    engineer_features,
    prepare_model_frame,
    split_features_and_target,
)

TRAIN_PATH = r"/mnt/DATA/Courses/DR_Mostafa_Saad/My Tasks/ML Tasks/Project 1/split/train.csv"
VAL_PATH = r"/mnt/DATA/Courses/DR_Mostafa_Saad/My Tasks/ML Tasks/Project 1/split/val.csv"

MODEL_OUTPUT_PATH = "ridge_trip_duration_model.joblib"
MEDIANS_OUTPUT_PATH = "train_medians.joblib"

RIDGE_ALPHA = 1.0


def build_pipeline():
    column_transformer = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("scaling", StandardScaler(), NUMERIC_FEATURES),
    ], remainder="passthrough")

    return Pipeline([
        ("preprocessing", column_transformer),
        ("model", Ridge(alpha=RIDGE_ALPHA)),
    ])


def main():
    train_raw = pd.read_csv(TRAIN_PATH)
    val_raw = pd.read_csv(VAL_PATH)

    train = engineer_features(train_raw)
    val = engineer_features(val_raw)

    train_prepared, train_medians = prepare_model_frame(train)
    val_prepared, _ = prepare_model_frame(val, fill_medians=train_medians)

    X_train, y_train = split_features_and_target(train_prepared)
    X_val, y_val = split_features_and_target(val_prepared)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    train_r2 = r2_score(y_train, pipeline.predict(X_train))
    val_r2 = r2_score(y_val, pipeline.predict(X_val))

    print(f"Ridge(alpha={RIDGE_ALPHA}) Train R²:      {train_r2:.4f}")
    print(f"Ridge(alpha={RIDGE_ALPHA}) Validation R²: {val_r2:.4f}")

    # joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    # joblib.dump(train_medians, MEDIANS_OUTPUT_PATH)
    # print(f"\nSaved fitted pipeline to {MODEL_OUTPUT_PATH}")
    # print(f"Saved training medians to {MEDIANS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
