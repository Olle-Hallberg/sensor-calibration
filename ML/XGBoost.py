import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# Read the dataset
data = pd.read_csv("gas_sensor_data.csv")


# Choose target variable: "methane_ppm" or "co2_ppm"
target_column = "methane_ppm"


# Select input features and target
feature_columns = ["sensorsignal", "pressure", "humidity", "temperature"]
X = data[feature_columns]
y = data[target_column]


# Remove rows with missing values
valid_idx = X.notna().all(axis=1) & y.notna()
X = X.loc[valid_idx]
y = y.loc[valid_idx]


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=1
)


# Define the XGBoost regressor
xgb_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=1,
    n_jobs=-1
)


# Define the hyperparameter search space
param_grid = {
    "n_estimators": [50, 100, 150, 200],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4, 5, 6],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "gamma": [0, 0.1, 0.3]
}


# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=1)


# Perform grid search
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    verbose=1
)


# Train all model combinations
grid_search.fit(X_train, y_train)


# Print the best hyperparameters
print("\nBest hyperparameters:")
print(grid_search.best_params_)


# Print the best cross-validation score
print("\nBest CV score (negative MAE):")
print(grid_search.best_score_)


# Extract the best model
best_model = grid_search.best_estimator_


# Predict on the test set
y_pred = best_model.predict(X_test)


# Evaluate the final model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


print(f"\nResults for target: {target_column}")
print(f"MAE  = {mae:.3f} ppm")
print(f"RMSE = {rmse:.3f} ppm")
print(f"R²   = {r2:.3f}")


# Store true and predicted values
results = pd.DataFrame({
    "True_ppm": y_test.values,
    "Predicted_ppm": y_pred
})


print("\nFirst 10 predictions:")
print(results.head(10))


# Save predictions to file
results.to_csv(f"predictions_xgboost_{target_column}.csv", index=False)