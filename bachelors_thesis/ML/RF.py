import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# #seed
np.random.seed(1)


# #model
def build_rf_model(params):
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        bootstrap=params["bootstrap"],
        random_state=1,
        n_jobs=-1
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


# #grid
param_grid = {
    # n_estimators -> number of trees
    # max_depth -> maximum tree depth
    # min_samples_split -> minimum samples required to split
    # min_samples_leaf -> minimum samples required in leaf
    # max_features -> number of features considered at each split
    # bootstrap -> whether bootstrap samples are used
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]
}


# #cv
cv = KFold(n_splits=5, shuffle=True, random_state=1)

best_score = np.inf
best_params = None
search_results = []

all_param_combinations = list(ParameterGrid(param_grid))
total_runs = len(all_param_combinations)

print("Starting Random Forest hyperparameter search...\n")


# #search
for run_number, params in enumerate(all_param_combinations, start=1):
    cv_scores = []

    for train_idx, val_idx in cv.split(X_train):
        # #fold_split
        X_cv_train = X_train.iloc[train_idx]
        X_cv_val = X_train.iloc[val_idx]
        y_cv_train = y_train.iloc[train_idx]
        y_cv_val = y_train.iloc[val_idx]

        model = build_rf_model(params)

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
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
        "min_samples_split": params["min_samples_split"],
        "min_samples_leaf": params["min_samples_leaf"],
        "max_features": params["max_features"],
        "bootstrap": params["bootstrap"],
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
search_results_df.to_csv("hyperparameter_results_random_forest.csv", index=False)

print("\nBest hyperparameters:")
print(best_params)
print(f"\nBest CV MAE: {best_score:.3f} ppm")


# #finalmodel
best_model = build_rf_model(best_params)

train_model(
    model=best_model,
    X_train=X_train,
    y_train=y_train
)


# #testpred
y_pred = predict_model(best_model, X_test)


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
    f"predictions_test_{target_column}_random_forest.csv",
    index=False
)


# #saveall
y_all_pred = predict_model(best_model, X)

all_results = pd.DataFrame({
    "sensorsignal": X["sensorsignal"].values,
    "pressure": X["pressure"].values,
    "humidity": X["humidity"].values,
    "temperature": X["temperature"].values,
    "True_ppm": y.values,
    "Predicted_ppm": y_all_pred
})

all_results.to_csv(
    f"predictions_all_{target_column}_random_forest.csv",
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
plt.title(f"Temperature Compensation (Random Forest)\nR² = {r2:.3f}")

plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig(f"time_series_all_{target_column}_random_forest.png", dpi=300)
plt.show()