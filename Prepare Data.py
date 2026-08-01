import numpy as np
import pandas as pd
import seaborn as sns
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable

train = pd.read_csv(r"X:\DR_Mostafa_Saad\My Tasks\ML Tasks\Project 1\nyc-taxi-trip-duration\train\train.csv")
test = pd.read_csv(r"X:\DR_Mostafa_Saad\My Tasks\ML Tasks\Project 1\nyc-taxi-trip-duration\test\test.csv") # test_val
sample_submission = pd.read_csv(r"X:\DR_Mostafa_Saad\My Tasks\ML Tasks\Project 1\nyc-taxi-trip-duration\sample_submission\sample_submission.csv") # test_val

# Extract Datetime Features
def extract_datetime_features(df, datetime_cols):

    df = df.copy()
    
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        prefix = col.replace('_datetime', '') if '_datetime' in col else col
        
        df[f'{prefix}_second'] = df[col].dt.second              # 0 --> 59
        df[f'{prefix}_minute'] = df[col].dt.minute              # 0 --> 59
        df[f'{prefix}_hour'] = df[col].dt.hour                  # 0 --> 23
        df[f'{prefix}_dayofweek'] = df[col].dt.dayofweek        # 0 --> 6 (Monday-Sunday)
        df[f'{prefix}_day'] = df[col].dt.day                    # 1 --> 31
        df[f'{prefix}_month'] = df[col].dt.month                # 1 --> 12
        df[f'{prefix}_year'] = df[col].dt.year                  # e.g. 2016
    
    return df

# create distance column
def haversine_formula(lat1, lat2, lon1, lon2):
    
    R = 3958.8
    lat1, lat2, lon1, lon2 = map(np.radians, [lat1, lat2, lon1, lon2])

    a = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return c * R 


train = extract_datetime_features(train, ['pickup_datetime'])
test = extract_datetime_features(test, ['pickup_datetime'])

train['distance'] = haversine_formula(train['pickup_latitude'], train['dropoff_latitude'],
                                       train['pickup_longitude'], train['dropoff_longitude'])

test['distance'] = haversine_formula(test['pickup_latitude'], test['dropoff_latitude'],
                                       test['pickup_longitude'], test['dropoff_longitude'])

# handle store_and_fwd_flag column
train['store_and_fwd_flag'] = (train['store_and_fwd_flag'] == 'Y').astype(int)
test['store_and_fwd_flag'] = (test['store_and_fwd_flag'] == 'Y').astype(int)
#################################################################################################################
# Modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, accuracy_score

from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
#################################################################################################################
train_copy = train.copy()
test_copy = test.copy()
sample_submission_copy = sample_submission.copy()


train_copy = train_copy.drop(columns=['id','pickup_datetime', 'dropoff_datetime'])
                            #  'pickup_latitude', 'dropoff_latitude', 'pickup_longitude', 'dropoff_longitude'])
test_copy = test_copy.drop(columns=['id', 'pickup_datetime'])
                        #    'pickup_latitude', 'dropoff_latitude', 'pickup_longitude', 'dropoff_longitude'])
sample_submission_copy = sample_submission_copy.drop(columns=["id"])

train_copy = train_copy.drop_duplicates()
train_copy = train_copy.fillna(train_copy.median())
test_copy = test_copy.fillna(test_copy.median())
#################################################################################################################
# from ydata_profiling import ProfileReport
# from ydata_profiling.utils.cache import cache_file

# report = ProfileReport(train_copy, title="Trip_Duration EDA")
# report.to_file("report.html")

# train_copy.info()
#################################################################################################################
X = train_copy.drop(columns=['trip_duration'])
y = train_copy['trip_duration']

X['distance'] = np.log1p(X['distance'])
test_copy['distance'] = np.log1p(test_copy['distance'])
y = np.log1p(y)
sample_submission_copy = np.log1p(sample_submission_copy)

X_valtest = test_copy.reset_index(drop=True)
y_valtest = sample_submission_copy.reset_index(drop=True)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)

X_train = X.reset_index(drop=True)
y_train = y.reset_index(drop=True)
#################################################################################################################
poly = PolynomialFeatures(degree=2, include_bias=True)
X_train = poly.fit_transform(X_train)
X_val = poly.fit_transform(X_val)
X_test = poly.fit_transform(X_test)
#################################################################################################################
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
print("Scaler Done!")
#################################################################################################################
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np


# Step 2: Use RidgeCV to find the optimal alpha
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
ridge_cv.fit(X_train, y_train)

# Step 3: Evaluate the model
train_r2 = ridge_cv.score(X_train, y_train)
val_r2 = ridge_cv.score(X_val, y_val)

print("Best Alpha:", ridge_cv.alpha_)
print("Train R²:", train_r2)
print("Validation R²:", val_r2)

# Step 4: Cross-Validation
cv_scores = cross_val_score(ridge_cv, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))
#################################################################################################################
bst = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
bst = Ridge(alpha=0.01)
bst.fit(X_train, y_train)
bst_pred = bst.predict(X_val)
bst_r2 = r2_score(y_val, bst_pred)
print(f"XGB Validation R²: {bst_r2:.4f}")
bst_pred_train = bst.predict(X_train)
bst_r2_train = r2_score(y_train, bst_pred_train)
print(f"XGB Train R²: {bst_r2_train:.4f}")
#################################################################################################################