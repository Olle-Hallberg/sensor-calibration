import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# #seed
np.random.seed(1)


# #model
def build_svr_model(params):
    model = SVR(
        kernel=params["kernel"],
        C=params["C"],
        epsilon=params["epsilon"],
        gamma=params["gamma"]
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

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()


# #grid
param_grid = {
    # kernel -> SVR kernel function
    # C -> regularization strength
    # epsilon -> width of epsilon-insensitive tube
    # gamma -> kernel coefficient
    "kernel": ["rbf", "linear", "poly"],
    "C": [0.1, 1, 10, 100],
    "epsilon": [0.01, 0.05, 0.1, 0.2],
    "gamma": ["scale", "auto"]
}


# #cv
cv = KFold(n_splits=5, shuffle=True, random_state=1)

best_score = np.inf
best_params = None
search_results = []

all_param_combinations = list(ParameterGrid(param_grid))
total_runs = len(all_param_combinations)

print("Starting Support Vector Regression hyperparameter search...\n")


# #search
for run_number, params in enumerate(all_param_combinations, start=1):
    cv_scores = []

    for train_idx, val_idx in cv.split(X_train):
        # #fold_split
        X_cv_train_raw = X_train.iloc[train_idx]
        X_cv_val_raw = X_train.iloc[val_idx]
        y_cv_train_raw = y_train.iloc[train_idx]
        y_cv_val_raw = y_train.iloc[val_idx]

        # #fold_scaleX
        x_scaler_cv = StandardScaler()
        X_cv_train_scaled = x_scaler_cv.fit_transform(X_cv_train_raw)
        X_cv_val_scaled = x_scaler_cv.transform(X_cv_val_raw)

        # #fold_scaleY
        y_scaler_cv = StandardScaler()
        y_cv_train_scaled = y_scaler_cv.fit_transform(
            y_cv_train_raw.values.reshape(-1, 1)
        ).flatten()
        y_cv_val_scaled = y_scaler_cv.transform(
            y_cv_val_raw.values.reshape(-1, 1)
        ).flatten()

        model = build_svr_model(params)

        train_model(
            model=model,
            X_train=X_cv_train_scaled,
            y_train=y_cv_train_scaled
        )

        y_val_pred_scaled = predict_model(model, X_cv_val_scaled)
        y_val_true_scaled = y_cv_val_scaled

        y_val_pred = y_scaler_cv.inverse_transform(
            y_val_pred_scaled.reshape(-1, 1)
        ).flatten()

        y_val_true = y_scaler_cv.inverse_transform(
            y_val_true_scaled.reshape(-1, 1)
        ).flatten()

        fold_mae = mean_absolute_error(y_val_true, y_val_pred)
        cv_scores.append(fold_mae)

    mean_cv_score = np.mean(cv_scores)

    search_results.append({
        "kernel": params["kernel"],
        "C": params["C"],
        "epsilon": params["epsilon"],
        "gamma": params["gamma"],
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
search_results_df.to_csv("hyperparameter_results_svr.csv", index=False)

print("\nBest hyperparameters:")
print(best_params)
print(f"\nBest CV MAE: {best_score:.3f} ppm")


# #finalmodel
best_model = build_svr_model(best_params)

train_model(
    model=best_model,
    X_train=X_train_scaled,
    y_train=y_train_scaled
)


# #testpred
y_pred_scaled = predict_model(best_model, X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()


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
    f"predictions_test_{target_column}_svr.csv",
    index=False
)


# #saveall
X_all_scaled = x_scaler.transform(X)
y_all_pred_scaled = predict_model(best_model, X_all_scaled)
y_all_pred = y_scaler.inverse_transform(y_all_pred_scaled.reshape(-1, 1)).flatten()

all_results = pd.DataFrame({
    "sensorsignal": X["sensorsignal"].values,
    "pressure": X["pressure"].values,
    "humidity": X["humidity"].values,
    "temperature": X["temperature"].values,
    "True_ppm": y.values,
    "Predicted_ppm": y_all_pred
})

all_results.to_csv(
    f"predictions_all_{target_column}_svr.csv",
    index=False
)


# #plot_timeseries
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
plt.title(f"Temperature Compensation (SVR)\nR² = {r2:.3f}")

plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig(f"time_series_all_{target_column}_svr.png", dpi=300)
plt.show()