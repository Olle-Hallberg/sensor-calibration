import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# #seed
np.random.seed(1)


# #model
def build_linear_model(params):
    model = LinearRegression(
        fit_intercept=params["fit_intercept"],
        positive=params["positive"]
    )
    return model


# #train
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


# #predict
def predict_model(model, X):
    return model.predict(X)


# #load
data = pd.read_csv("gas_sensor_data.csv")


# #target
target_column = "methane_ppm"
feature_columns = ["sensorsignal", "pressure", "humidity", "temperature"]


# #select
X = data[feature_columns]
y = data[target_column]


# #clean
valid_idx = X.notna().all(axis=1) & y.notna()
X = X.loc[valid_idx].copy()
y = y.loc[valid_idx].copy()


# #split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=1
)


# #final_scale
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

X_train_final = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
X_test_final = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)


# #grid
param_grid = {
    "fit_intercept": [True, False],
    "positive": [False, True]
}


# #cv
cv = KFold(n_splits=5, shuffle=True, random_state=1)

best_score = np.inf
best_params = None
search_results = []

all_param_combinations = list(ParameterGrid(param_grid))
total_runs = len(all_param_combinations)

print("Starting Linear Regression hyperparameter search...\n")


# #search
for run_number, params in enumerate(all_param_combinations, start=1):
    cv_scores = []

    for train_idx, val_idx in cv.split(X_train):
        # #fold_split
        X_cv_train_raw = X_train.iloc[train_idx]
        X_cv_val_raw = X_train.iloc[val_idx]
        y_cv_train = y_train.iloc[train_idx]
        y_cv_val = y_train.iloc[val_idx]

        # #fold_scaleX
        x_scaler_cv = StandardScaler()
        X_cv_train_scaled = x_scaler_cv.fit_transform(X_cv_train_raw)
        X_cv_val_scaled = x_scaler_cv.transform(X_cv_val_raw)

        X_cv_train = pd.DataFrame(
            X_cv_train_scaled,
            columns=feature_columns,
            index=X_cv_train_raw.index
        )

        X_cv_val = pd.DataFrame(
            X_cv_val_scaled,
            columns=feature_columns,
            index=X_cv_val_raw.index
        )

        model = build_linear_model(params)

        train_model(
            model=model,
            X_train=X_cv_train,
            y_train=y_cv_train
        )

        y_val_pred = predict_model(model, X_cv_val)
        y_val_true = y_cv_val.values

        fold_mae = mean_absolute_error(y_val_true, y_val_pred)
        cv_scores.append(fold_mae)

    mean_cv_score = np.mean(cv_scores)

    search_results.append({
        "fit_intercept": params["fit_intercept"],
        "positive": params["positive"],
        "cv_mae": mean_cv_score
    })

    if mean_cv_score < best_score:
        best_score = mean_cv_score
        best_params = params
        print(
            f"[{run_number}/{total_runs}] New best -> "
            f"MAE = {best_score:.3f} ppm, params = {best_params}"
        )


# #savegrid
search_results_df = pd.DataFrame(search_results)
search_results_df = search_results_df.sort_values("cv_mae", ascending=True)
search_results_df.to_csv("hyperparameter_results_linear_regression.csv", index=False)

print("\nBest hyperparameters:")
print(best_params)
print(f"\nBest CV MAE: {best_score:.3f} ppm")


# #finalmodel
best_model = build_linear_model(best_params)

train_model(
    model=best_model,
    X_train=X_train_final,
    y_train=y_train
)


# #testpred
y_pred = predict_model(best_model, X_test_final)


# #metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nResults for target: {target_column}")
print(f"MAE  = {mae:.3f} ppm")
print(f"RMSE = {rmse:.3f} ppm")
print(f"R²   = {r2:.3f}")


# #savetest
test_results = pd.DataFrame({
    "True_ppm": y_test.values,
    "Predicted_ppm": y_pred
})

print("\nFirst 10 test predictions:")
print(test_results.head(10))

test_results.to_csv(
    f"predictions_test_{target_column}_linear_regression.csv",
    index=False
)


# #saveall
X_all_scaled = x_scaler.transform(X)
X_all_final = pd.DataFrame(X_all_scaled, columns=feature_columns, index=X.index)

y_all_pred = predict_model(best_model, X_all_final)

all_results = pd.DataFrame({
    "sensorsignal": X["sensorsignal"].values,
    "pressure": X["pressure"].values,
    "humidity": X["humidity"].values,
    "temperature": X["temperature"].values,
    "True_ppm": y.values,
    "Predicted_ppm": y_all_pred
})

all_results.to_csv(
    f"predictions_all_{target_column}_linear_regression.csv",
    index=False
)


# #plot_timeseries (NO temperature axis)
all_plot_df = all_results.copy()
all_plot_df = all_plot_df.sort_index()

x_axis = np.arange(len(all_plot_df))

plt.figure(figsize=(14, 6))

plt.plot(
    x_axis,
    all_plot_df["True_ppm"].values,
    label="reference"
)

plt.plot(
    x_axis,
    all_plot_df["Predicted_ppm"].values,
    label="calibrated CH4 concentration"
)

plt.xlabel("Sample index")
plt.ylabel("CH4 concentration (ppm)")
plt.title(f"Temperature Compensation (Linear Regression)\nR² = {r2:.3f}")

plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig(f"time_series_all_{target_column}_linear_regression.png", dpi=300)
plt.show()