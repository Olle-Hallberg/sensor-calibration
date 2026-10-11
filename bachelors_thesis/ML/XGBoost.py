import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


# ============================================================
# Seed
# ============================================================

np.random.seed(1)
random.seed(1)


# ============================================================
# File names
# ============================================================

TRAIN_FILE = "training_data.csv"
TEST_FILE = "test_data.csv"


# ============================================================
# Settings for random search
# ============================================================

N_ITER_STAGE1 = 500
N_ITER_STAGE2 = 500


# ============================================================
# Settings for final train/test split
# ============================================================

TEST_SIZE = 0.20
SHUFFLE_RANDOM_STATE = 1


# ============================================================
# Plot settings
# ============================================================

PLOT_N_POINTS = 200
PLOT_RANDOM_SUBSET = False
PLOT_RANDOM_STATE = 1


# ============================================================
# Settings for test-data filtering before combining
# ============================================================

CH4_MIN = 1.0
CH4_MAX = 3.0

CO2_MIN = 300.0
CO2_MAX = 800.0

ROLLING_WINDOW = 101

CH4_MAX_LOCAL_DEVIATION = 0.08
CO2_MAX_LOCAL_DEVIATION = 20.0

USE_FEATURE_RANGE_FILTER = True
FEATURE_RANGE_MARGIN_FACTOR = 0.25


# ============================================================
# Function for training one XGBoost model per target
# ============================================================

def train_xgboost_multioutput(X_train, y_train, params):
    models = {}

    for target in y_train.columns:
        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            gamma=params["gamma"],
            max_delta_step=params["max_delta_step"],
            objective="reg:squarederror",
            random_state=1,
            n_jobs=-1
        )

        model.fit(X_train, y_train[target])
        models[target] = model

    return models


# ============================================================
# Prediction function
# ============================================================

def predict_xgboost_multioutput(models, X, target_columns):
    predictions = []

    for target in target_columns:
        pred = models[target].predict(X)
        predictions.append(pred)

    return np.column_stack(predictions)


# ============================================================
# Test-data filtering function before combining
# ============================================================

def filter_test_data(
    X_test,
    y_test,
    X_train,
    feature_columns,
    target_columns
):
    X_test = X_test.copy()
    y_test = y_test.copy()

    mask = pd.Series(True, index=X_test.index)

    finite_mask = (
        np.isfinite(X_test[feature_columns]).all(axis=1)
        & np.isfinite(y_test[target_columns]).all(axis=1)
    )

    mask &= finite_mask

    print("\nFiltering original test_data.csv before combining:")
    print(f"Initial test samples: {len(mask)}")
    print(f"Removed by NaN/inf filter: {(~finite_mask).sum()}")

    ch4_col = target_columns[0]
    co2_col = target_columns[1]

    physical_mask = (
        (y_test[ch4_col] >= CH4_MIN)
        & (y_test[ch4_col] <= CH4_MAX)
        & (y_test[co2_col] >= CO2_MIN)
        & (y_test[co2_col] <= CO2_MAX)
    )

    removed_physical = (mask & ~physical_mask).sum()
    mask &= physical_mask

    print(f"Removed by physical LGR limits: {removed_physical}")

    ch4_rolling_median = y_test[ch4_col].rolling(
        window=ROLLING_WINDOW,
        center=True,
        min_periods=1
    ).median()

    ch4_local_deviation = np.abs(y_test[ch4_col] - ch4_rolling_median)
    ch4_spike_mask = ch4_local_deviation <= CH4_MAX_LOCAL_DEVIATION

    removed_ch4_spikes = (mask & ~ch4_spike_mask).sum()
    mask &= ch4_spike_mask

    print(f"Removed by CH4 local spike filter: {removed_ch4_spikes}")

    co2_rolling_median = y_test[co2_col].rolling(
        window=ROLLING_WINDOW,
        center=True,
        min_periods=1
    ).median()

    co2_local_deviation = np.abs(y_test[co2_col] - co2_rolling_median)
    co2_spike_mask = co2_local_deviation <= CO2_MAX_LOCAL_DEVIATION

    removed_co2_spikes = (mask & ~co2_spike_mask).sum()
    mask &= co2_spike_mask

    print(f"Removed by CO2 local spike filter: {removed_co2_spikes}")

    if USE_FEATURE_RANGE_FILTER:
        feature_mask = pd.Series(True, index=X_test.index)

        for col in feature_columns:
            train_min = X_train[col].min()
            train_max = X_train[col].max()
            train_range = train_max - train_min

            lower_limit = train_min - FEATURE_RANGE_MARGIN_FACTOR * train_range
            upper_limit = train_max + FEATURE_RANGE_MARGIN_FACTOR * train_range

            col_mask = (
                (X_test[col] >= lower_limit)
                & (X_test[col] <= upper_limit)
            )

            feature_mask &= col_mask

        removed_feature_outliers = (mask & ~feature_mask).sum()
        mask &= feature_mask

        print(f"Removed by feature range filter: {removed_feature_outliers}")

    X_filtered = X_test.loc[mask].reset_index(drop=True)
    y_filtered = y_test.loc[mask].reset_index(drop=True)

    print(f"Final test_data samples kept before combining: {len(X_filtered)}")
    print(f"Final test_data samples removed before combining: {len(mask) - len(X_filtered)}")

    return X_filtered, y_filtered


# ============================================================
# Random search hyperparameter evaluation with normalization
# ============================================================

def evaluate_random_search(
    X_train_raw,
    y_train_raw,
    target_columns,
    param_grid,
    cv,
    stage_name,
    n_iter,
    random_seed=1
):
    best_score = np.inf
    best_params = None
    search_results = []

    all_param_combinations = list(ParameterGrid(param_grid))
    total_possible_runs = len(all_param_combinations)

    random.seed(random_seed)

    if n_iter >= total_possible_runs:
        sampled_param_combinations = all_param_combinations
    else:
        sampled_param_combinations = random.sample(
            all_param_combinations,
            k=n_iter
        )

    total_runs = len(sampled_param_combinations)

    print(f"\nStarting {stage_name} random hyperparameter search...")
    print(f"Total possible hyperparameter combinations: {total_possible_runs}")
    print(f"Randomly selected combinations: {total_runs}")
    print(f"Number of folds: {cv.get_n_splits()}")
    print(
        f"Maximum number of trained XGBoost model sets in {stage_name}: "
        f"{total_runs * cv.get_n_splits()}\n"
    )

    for run_number, params in enumerate(sampled_param_combinations, start=1):
        print(f"Running {stage_name} combination {run_number}/{total_runs}: {params}")

        cv_scores = []
        cv_scores_ch4 = []
        cv_scores_co2 = []

        for fold_number, (train_idx, val_idx) in enumerate(
            cv.split(X_train_raw),
            start=1
        ):
            X_cv_train_raw = X_train_raw.iloc[train_idx].copy()
            X_cv_val_raw = X_train_raw.iloc[val_idx].copy()

            y_cv_train = y_train_raw.iloc[train_idx].copy()
            y_cv_val = y_train_raw.iloc[val_idx].copy()

            # ============================================================
            # Normalize features inside each CV fold
            # Important: fit scaler only on the CV training fold
            # ============================================================

            scaler = StandardScaler()

            X_cv_train_scaled = scaler.fit_transform(X_cv_train_raw)
            X_cv_val_scaled = scaler.transform(X_cv_val_raw)

            X_cv_train_scaled = pd.DataFrame(
                X_cv_train_scaled,
                columns=X_train_raw.columns
            )

            X_cv_val_scaled = pd.DataFrame(
                X_cv_val_scaled,
                columns=X_train_raw.columns
            )

            models = train_xgboost_multioutput(
                X_train=X_cv_train_scaled,
                y_train=y_cv_train,
                params=params
            )

            y_val_pred = predict_xgboost_multioutput(
                models=models,
                X=X_cv_val_scaled,
                target_columns=target_columns
            )

            y_val_true = y_cv_val[target_columns].values

            mae_ch4 = mean_absolute_error(y_val_true[:, 0], y_val_pred[:, 0])
            mae_co2 = mean_absolute_error(y_val_true[:, 1], y_val_pred[:, 1])

            mean_mae = np.mean([mae_ch4, mae_co2])

            cv_scores.append(mean_mae)
            cv_scores_ch4.append(mae_ch4)
            cv_scores_co2.append(mae_co2)

            print(
                f"  Fold {fold_number}/{cv.get_n_splits()}: "
                f"CH4 MAE = {mae_ch4:.6f} ppm, "
                f"CO2 MAE = {mae_co2:.6f} ppm, "
                f"Mean MAE = {mean_mae:.6f} ppm"
            )

        mean_cv_score = np.mean(cv_scores)
        mean_cv_ch4 = np.mean(cv_scores_ch4)
        mean_cv_co2 = np.mean(cv_scores_co2)

        search_results.append({
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "learning_rate": params["learning_rate"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "min_child_weight": params["min_child_weight"],
            "reg_alpha": params["reg_alpha"],
            "reg_lambda": params["reg_lambda"],
            "gamma": params["gamma"],
            "max_delta_step": params["max_delta_step"],
            "cv_mae_mean_ppm": mean_cv_score,
            "cv_mae_ch4_ppm": mean_cv_ch4,
            "cv_mae_co2_ppm": mean_cv_co2,
            "stage": stage_name
        })

        print(
            f"  Mean CV MAE = {mean_cv_score:.6f} ppm "
            f"(CH4 = {mean_cv_ch4:.6f} ppm, CO2 = {mean_cv_co2:.6f} ppm)\n"
        )

        if mean_cv_score < best_score:
            best_score = mean_cv_score
            best_params = params

            print(
                f"[{run_number}/{total_runs}] New best -> "
                f"Mean MAE = {best_score:.6f} ppm, params = {best_params}\n"
            )

    return best_score, best_params, search_results


# ============================================================
# Load data
# ============================================================

train_data = pd.read_csv(TRAIN_FILE, sep=";")
test_data = pd.read_csv(TEST_FILE, sep=";")

print("Columns in training file:")
print(train_data.columns.tolist())

print("\nColumns in test file:")
print(test_data.columns.tolist())


# ============================================================
# Select features and targets
# ============================================================

feature_columns = [
    "spl_signal",
    "lpl_signal",
    "rh_temp_c",
    "mpx5500_pressure_hpa",
    "humidity_pct"
]

if "mpl_signal" in train_data.columns and "mpl_signal" in test_data.columns:
    feature_columns.append("mpl_signal")

target_columns = [
    "ch4_ppm",
    "co2_ppm"
]

test_target_columns = [
    "lgr_ch4_ppm",
    "lgr_co2_ppm"
]


required_train_columns = feature_columns + target_columns
required_test_columns = feature_columns + test_target_columns

for col in required_train_columns:
    if col not in train_data.columns:
        raise ValueError(f"Column '{col}' is missing from training file.")

for col in required_test_columns:
    if col not in test_data.columns:
        raise ValueError(f"Column '{col}' is missing from test file.")


# ============================================================
# Convert columns to numeric
# ============================================================

for col in required_train_columns:
    train_data[col] = pd.to_numeric(train_data[col], errors="coerce")

for col in required_test_columns:
    test_data[col] = pd.to_numeric(test_data[col], errors="coerce")


# ============================================================
# Prepare training_data.csv
# ============================================================

X_train_file = train_data[feature_columns].copy()
y_train_file = train_data[target_columns].copy()

valid_train_idx = (
    X_train_file.notna().all(axis=1)
    & y_train_file.notna().all(axis=1)
)

X_train_file = X_train_file.loc[valid_train_idx].reset_index(drop=True)
y_train_file = y_train_file.loc[valid_train_idx].reset_index(drop=True)

train_file_clean = pd.concat(
    [X_train_file, y_train_file],
    axis=1
)

train_file_clean["source_file"] = "training_data.csv"


# ============================================================
# Prepare test_data.csv and rename LGR targets
# ============================================================

X_test_file = test_data[feature_columns].copy()

y_test_file = test_data[test_target_columns].copy()
y_test_file = y_test_file.rename(columns={
    "lgr_ch4_ppm": "ch4_ppm",
    "lgr_co2_ppm": "co2_ppm"
})

valid_test_idx = (
    X_test_file.notna().all(axis=1)
    & y_test_file.notna().all(axis=1)
)

X_test_file = X_test_file.loc[valid_test_idx].reset_index(drop=True)
y_test_file = y_test_file.loc[valid_test_idx].reset_index(drop=True)

X_test_file, y_test_file = filter_test_data(
    X_test=X_test_file,
    y_test=y_test_file,
    X_train=X_train_file,
    feature_columns=feature_columns,
    target_columns=target_columns
)

test_file_clean = pd.concat(
    [X_test_file, y_test_file],
    axis=1
)

test_file_clean["source_file"] = "test_data.csv"


# ============================================================
# Combine training_data.csv and test_data.csv
# ============================================================

combined_data = pd.concat(
    [train_file_clean, test_file_clean],
    axis=0
).reset_index(drop=True)

print("\nCombined dataset:")
print(f"Samples from training_data.csv: {len(train_file_clean)}")
print(f"Samples from test_data.csv: {len(test_file_clean)}")
print(f"Total samples before shuffling: {len(combined_data)}")


# ============================================================
# Shuffle combined dataset and split into 80 % train, 20 % test
# ============================================================

combined_data_shuffled = combined_data.sample(
    frac=1.0,
    random_state=SHUFFLE_RANDOM_STATE
).reset_index(drop=True)

X_all = combined_data_shuffled[feature_columns].copy()
y_all = combined_data_shuffled[target_columns].copy()
source_all = combined_data_shuffled["source_file"].copy()

X_train_raw, X_test_raw, y_train_raw, y_test_raw, source_train, source_test = train_test_split(
    X_all,
    y_all,
    source_all,
    test_size=TEST_SIZE,
    shuffle=True,
    random_state=SHUFFLE_RANDOM_STATE
)

X_train_raw = X_train_raw.reset_index(drop=True)
X_test_raw = X_test_raw.reset_index(drop=True)
y_train_raw = y_train_raw.reset_index(drop=True)
y_test_raw = y_test_raw.reset_index(drop=True)
source_train = source_train.reset_index(drop=True)
source_test = source_test.reset_index(drop=True)

print("\nAfter shuffling and 80/20 split:")
print(f"Training samples: {len(X_train_raw)}")
print(f"Test samples: {len(X_test_raw)}")

print("\nSource distribution in final training set:")
print(source_train.value_counts())

print("\nSource distribution in final test set:")
print(source_test.value_counts())

print("\nSelected features:")
print(feature_columns)

print("\nTargets:")
print(target_columns)


# ============================================================
# Stage 1 hyperparameter grid
# ============================================================

param_grid_stage1 = {
    "n_estimators": [100, 200, 400, 600, 800, 1000],
    "max_depth": [2, 3, 4, 5, 6, 8],
    "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "subsample": [0.6, 0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.85, 1.0],
    "min_child_weight": [1, 3, 5, 10, 20],
    "reg_alpha": [0.0, 0.001, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    "gamma": [0.0, 0.01, 0.1, 1.0],
    "max_delta_step": [0, 1, 5]
}

cv_stage1 = KFold(
    n_splits=5,
    shuffle=True,
    random_state=1
)

best_score_stage1, best_params_stage1, search_results_stage1 = evaluate_random_search(
    X_train_raw=X_train_raw,
    y_train_raw=y_train_raw,
    target_columns=target_columns,
    param_grid=param_grid_stage1,
    cv=cv_stage1,
    stage_name="stage1",
    n_iter=N_ITER_STAGE1,
    random_seed=1
)

if best_params_stage1 is None:
    raise ValueError("No valid hyperparameter combinations found in stage 1.")

print("\nBest stage 1 hyperparameters:")
print(best_params_stage1)
print(f"\nBest stage 1 CV mean MAE: {best_score_stage1:.6f} ppm")


# ============================================================
# Stage 2 hyperparameter grid
# ============================================================

best_n_estimators = best_params_stage1["n_estimators"]
best_max_depth = best_params_stage1["max_depth"]
best_learning_rate = best_params_stage1["learning_rate"]
best_subsample = best_params_stage1["subsample"]
best_colsample = best_params_stage1["colsample_bytree"]
best_min_child_weight = best_params_stage1["min_child_weight"]
best_reg_alpha = best_params_stage1["reg_alpha"]
best_reg_lambda = best_params_stage1["reg_lambda"]
best_gamma = best_params_stage1["gamma"]
best_max_delta_step = best_params_stage1["max_delta_step"]

n_estimators_candidates = sorted(set([
    max(50, best_n_estimators - 300),
    max(50, best_n_estimators - 200),
    max(50, best_n_estimators - 100),
    best_n_estimators,
    best_n_estimators + 100,
    best_n_estimators + 200,
    best_n_estimators + 300
]))

max_depth_candidates = sorted(set([
    max(1, best_max_depth - 2),
    max(1, best_max_depth - 1),
    best_max_depth,
    best_max_depth + 1,
    best_max_depth + 2
]))

learning_rate_candidates = sorted(set([
    max(0.001, best_learning_rate * 0.5),
    max(0.001, best_learning_rate * 0.75),
    best_learning_rate,
    min(0.3, best_learning_rate * 1.25),
    min(0.3, best_learning_rate * 1.5),
    min(0.3, best_learning_rate * 2.0)
]))

subsample_candidates = sorted(set([
    max(0.5, best_subsample - 0.15),
    max(0.5, best_subsample - 0.10),
    best_subsample,
    min(1.0, best_subsample + 0.10),
    min(1.0, best_subsample + 0.15)
]))

colsample_candidates = sorted(set([
    max(0.5, best_colsample - 0.15),
    max(0.5, best_colsample - 0.10),
    best_colsample,
    min(1.0, best_colsample + 0.10),
    min(1.0, best_colsample + 0.15)
]))

min_child_weight_candidates = sorted(set([
    max(1, best_min_child_weight - 5),
    max(1, best_min_child_weight - 2),
    best_min_child_weight,
    best_min_child_weight + 2,
    best_min_child_weight + 5
]))

reg_alpha_candidates = sorted(set([
    0.0,
    max(0.0, best_reg_alpha * 0.5),
    best_reg_alpha,
    best_reg_alpha + 0.01,
    best_reg_alpha + 0.1,
    best_reg_alpha + 0.5
]))

reg_lambda_candidates = sorted(set([
    max(0.1, best_reg_lambda * 0.5),
    best_reg_lambda,
    best_reg_lambda * 1.5,
    best_reg_lambda * 2.0,
    best_reg_lambda + 5.0
]))

gamma_candidates = sorted(set([
    0.0,
    max(0.0, best_gamma * 0.5),
    best_gamma,
    best_gamma + 0.01,
    best_gamma + 0.1,
    best_gamma + 0.5
]))

max_delta_step_candidates = sorted(set([
    0,
    best_max_delta_step,
    max(0, best_max_delta_step - 1),
    best_max_delta_step + 1,
    best_max_delta_step + 3
]))

param_grid_stage2 = {
    "n_estimators": n_estimators_candidates,
    "max_depth": max_depth_candidates,
    "learning_rate": learning_rate_candidates,
    "subsample": subsample_candidates,
    "colsample_bytree": colsample_candidates,
    "min_child_weight": min_child_weight_candidates,
    "reg_alpha": reg_alpha_candidates,
    "reg_lambda": reg_lambda_candidates,
    "gamma": gamma_candidates,
    "max_delta_step": max_delta_step_candidates
}

cv_stage2 = KFold(
    n_splits=5,
    shuffle=True,
    random_state=2
)

best_score_stage2, best_params_stage2, search_results_stage2 = evaluate_random_search(
    X_train_raw=X_train_raw,
    y_train_raw=y_train_raw,
    target_columns=target_columns,
    param_grid=param_grid_stage2,
    cv=cv_stage2,
    stage_name="stage2",
    n_iter=N_ITER_STAGE2,
    random_seed=2
)

if best_params_stage2 is None:
    raise ValueError("No valid hyperparameter combinations found in stage 2.")

print("\nBest stage 2 hyperparameters:")
print(best_params_stage2)
print(f"\nBest stage 2 CV mean MAE: {best_score_stage2:.6f} ppm")


# ============================================================
# Select final hyperparameters
# ============================================================

all_stage_scores = [
    ("stage1", best_score_stage1, best_params_stage1),
    ("stage2", best_score_stage2, best_params_stage2)
]

best_stage, best_score, best_params = min(
    all_stage_scores,
    key=lambda item: item[1]
)

print("\nSelected final hyperparameters:")
print(best_params)
print(f"\nSelected from: {best_stage}")
print(f"Selected CV mean MAE: {best_score:.6f} ppm")


# ============================================================
# Save hyperparameter results
# ============================================================

search_results_all = search_results_stage1 + search_results_stage2

search_results_df = pd.DataFrame(search_results_all)

search_results_df = search_results_df.sort_values(
    ["cv_mae_mean_ppm"],
    ascending=[True]
)

search_results_df.to_csv(
    "hyperparameter_results_xgboost_combined_shuffled_80_20_normalized.csv",
    index=False
)

top_results = search_results_df.head(10)

print("\nTop 10 hyperparameter combinations:")
print(top_results.to_string(index=False))


# ============================================================
# Final normalization
# ============================================================
# Important:
# The scaler is fitted only on the final training set.
# The same scaler is then used to transform the held-out test set.

final_scaler = StandardScaler()

X_train_scaled = final_scaler.fit_transform(X_train_raw)
X_test_scaled = final_scaler.transform(X_test_raw)

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=feature_columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=feature_columns
)


# ============================================================
# Save normalized train/test features for inspection
# ============================================================

normalized_train_data = X_train_scaled.copy()
normalized_train_data[target_columns] = y_train_raw[target_columns].copy()
normalized_train_data["source_file"] = source_train.values

normalized_test_data = X_test_scaled.copy()
normalized_test_data[target_columns] = y_test_raw[target_columns].copy()
normalized_test_data["source_file"] = source_test.values

normalized_train_data.to_csv(
    "normalized_training_features_xgboost.csv",
    index=False
)

normalized_test_data.to_csv(
    "normalized_test_features_xgboost.csv",
    index=False
)

print("\nFeature normalization:")
print("StandardScaler was fitted on the final 80 % training set.")
print("The same scaler was used to transform the 20 % held-out test set.")

print("\nMean of normalized training features:")
print(X_train_scaled.mean())

print("\nStandard deviation of normalized training features:")
print(X_train_scaled.std())


# ============================================================
# Final training on normalized 80 % of combined scrambled dataset
# ============================================================

best_models = train_xgboost_multioutput(
    X_train=X_train_scaled,
    y_train=y_train_raw,
    params=best_params
)


# ============================================================
# Prediction on normalized 20 % held-out test set
# ============================================================

y_pred = predict_xgboost_multioutput(
    models=best_models,
    X=X_test_scaled,
    target_columns=target_columns
)

y_test_original = y_test_raw[target_columns].values


# ============================================================
# Test metrics
# ============================================================

mae_ch4 = mean_absolute_error(y_test_original[:, 0], y_pred[:, 0])
rmse_ch4 = np.sqrt(mean_squared_error(y_test_original[:, 0], y_pred[:, 0]))
r2_ch4 = r2_score(y_test_original[:, 0], y_pred[:, 0])

mae_co2 = mean_absolute_error(y_test_original[:, 1], y_pred[:, 1])
rmse_co2 = np.sqrt(mean_squared_error(y_test_original[:, 1], y_pred[:, 1]))
r2_co2 = r2_score(y_test_original[:, 1], y_pred[:, 1])

print("\nResults on normalized scrambled 20 % held-out test set:")

print(f"\nTarget: CH4 ppm")
print(f"MAE  = {mae_ch4:.6f} ppm")
print(f"RMSE = {rmse_ch4:.6f} ppm")
print(f"R²   = {r2_ch4:.6f}")

print(f"\nTarget: CO2 ppm")
print(f"MAE  = {mae_co2:.6f} ppm")
print(f"RMSE = {rmse_co2:.6f} ppm")
print(f"R²   = {r2_co2:.6f}")


# ============================================================
# Save test predictions
# ============================================================

test_results = pd.DataFrame({
    "test_sample_index": np.arange(len(X_test_raw)),
    "source_file": source_test.values,
    "True_CH4_ppm": y_test_original[:, 0],
    "Predicted_CH4_ppm": y_pred[:, 0],
    "True_CO2_ppm": y_test_original[:, 1],
    "Predicted_CO2_ppm": y_pred[:, 1]
})

# Save original, unnormalized feature values
for col in feature_columns:
    test_results[f"{col}_original"] = X_test_raw[col].values

# Save normalized feature values
for col in feature_columns:
    test_results[f"{col}_normalized"] = X_test_scaled[col].values

print("\nFirst 10 test predictions:")
print(test_results.head(10))

test_results.to_csv(
    "predictions_xgboost_combined_shuffled_20_percent_test_normalized.csv",
    index=False
)


# ============================================================
# Create smaller subset for plotting only
# ============================================================

if PLOT_RANDOM_SUBSET:
    plot_results_subset = test_results.sample(
        n=min(PLOT_N_POINTS, len(test_results)),
        random_state=PLOT_RANDOM_STATE
    ).sort_values("test_sample_index").reset_index(drop=True)
else:
    plot_results_subset = test_results.head(
        min(PLOT_N_POINTS, len(test_results))
    ).copy().reset_index(drop=True)

x_axis_full = test_results["test_sample_index"].values
x_axis_subset = np.arange(len(plot_results_subset))


# ============================================================
# Plot 1: CH4 on full test set
# ============================================================

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    x_axis_full,
    test_results["True_CH4_ppm"],
    label="True CH4",
    linewidth=1.4
)

ax.plot(
    x_axis_full,
    test_results["Predicted_CH4_ppm"],
    label="Predicted CH4",
    linewidth=1.4
)

ax.set_xlabel("Full shuffled test sample index")
ax.set_ylabel("CH4 ppm")
ax.set_title(
    f"XGBoost CH4 prediction on normalized full shuffled 20 % test set\n"
    f"R² = {r2_ch4:.4f}"
)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("xgboost_ch4_full_test_plot_normalized.png", dpi=300)
plt.show()


# ============================================================
# Plot 2: CO2 on full test set
# ============================================================

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    x_axis_full,
    test_results["True_CO2_ppm"],
    label="True CO2",
    linewidth=1.4
)

ax.plot(
    x_axis_full,
    test_results["Predicted_CO2_ppm"],
    label="Predicted CO2",
    linewidth=1.4
)

ax.set_xlabel("Full shuffled test sample index")
ax.set_ylabel("CO2 ppm")
ax.set_title(
    f"XGBoost CO2 prediction on normalized full shuffled 20 % test set\n"
    f"R² = {r2_co2:.4f}"
)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("xgboost_co2_full_test_plot_normalized.png", dpi=300)
plt.show()


# ============================================================
# Plot 3: CH4 on smaller subset of test set
# ============================================================

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    x_axis_subset,
    plot_results_subset["True_CH4_ppm"],
    label="True CH4",
    linewidth=1.8,
    marker="o",
    markersize=3
)

ax.plot(
    x_axis_subset,
    plot_results_subset["Predicted_CH4_ppm"],
    label="Predicted CH4",
    linewidth=1.8,
    marker="o",
    markersize=3
)

ax.set_xlabel("Displayed subset test sample index")
ax.set_ylabel("CH4 ppm")
ax.set_title(
    f"XGBoost CH4 prediction on normalized displayed subset of test set "
    f"({len(plot_results_subset)} points)\n"
    f"Full test set R² = {r2_ch4:.4f}"
)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("xgboost_ch4_subset_test_plot_normalized.png", dpi=300)
plt.show()


# ============================================================
# Plot 4: CO2 on smaller subset of test set
# ============================================================

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    x_axis_subset,
    plot_results_subset["True_CO2_ppm"],
    label="True CO2",
    linewidth=1.8,
    marker="o",
    markersize=3
)

ax.plot(
    x_axis_subset,
    plot_results_subset["Predicted_CO2_ppm"],
    label="Predicted CO2",
    linewidth=1.8,
    marker="o",
    markersize=3
)

ax.set_xlabel("Displayed subset test sample index")
ax.set_ylabel("CO2 ppm")
ax.set_title(
    f"XGBoost CO2 prediction on normalized displayed subset of test set "
    f"({len(plot_results_subset)} points)\n"
    f"Full test set R² = {r2_co2:.4f}"
)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("xgboost_co2_subset_test_plot_normalized.png", dpi=300)
plt.show()