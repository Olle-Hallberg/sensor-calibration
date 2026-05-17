import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# Seed and device
# ============================================================

np.random.seed(1)
random.seed(1)
torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# File names
# ============================================================

TRAIN_FILE = "training_data.csv"
TEST_FILE = "test_data.csv"


# ============================================================
# Settings for random search
# ============================================================

N_ITER_STAGE1 = 5
N_ITER_STAGE2 = 5


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
# Neural network model
# ============================================================

class GasNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, dropout):
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================
# Training function
# ============================================================

def train_neural_network(
    X_train,
    y_train,
    X_val,
    y_val,
    params,
    verbose=False
):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)

    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)

    X_train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32
    ).to(device)

    y_train_tensor = torch.tensor(
        y_train_scaled,
        dtype=torch.float32
    ).to(device)

    X_val_tensor = torch.tensor(
        X_val_scaled,
        dtype=torch.float32
    ).to(device)

    y_val_tensor = torch.tensor(
        y_val_scaled,
        dtype=torch.float32
    ).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True
    )

    model = GasNeuralNetwork(
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_layers=params["hidden_layers"],
        dropout=params["dropout"]
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"]
    )

    best_val_loss = np.inf
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, params["epochs"] + 1):
        model.train()

        train_losses = []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            y_batch_pred = model(X_batch)

            loss = criterion(y_batch_pred, y_batch)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()

        with torch.no_grad():
            y_val_pred_scaled = model(X_val_tensor)
            val_loss = criterion(y_val_pred_scaled, y_val_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and epoch % 50 == 0:
            mean_train_loss = np.mean(train_losses)

            print(
                f"Epoch {epoch}/{params['epochs']} - "
                f"Train loss = {mean_train_loss:.6f}, "
                f"Val loss = {val_loss:.6f}"
            )

        if patience_counter >= params["patience"]:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, x_scaler, y_scaler


# ============================================================
# Prediction function
# ============================================================

def predict_neural_network(model, x_scaler, y_scaler, X):
    X_scaled = x_scaler.transform(X)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    ).to(device)

    model.eval()

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()

    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    return y_pred


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
# Random search hyperparameter evaluation
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
        f"Maximum number of trained Neural Network models in {stage_name}: "
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
            X_cv_train = X_train_raw.iloc[train_idx].copy()
            X_cv_val = X_train_raw.iloc[val_idx].copy()

            y_cv_train = y_train_raw.iloc[train_idx].copy()
            y_cv_val = y_train_raw.iloc[val_idx].copy()

            model, x_scaler, y_scaler = train_neural_network(
                X_train=X_cv_train.values,
                y_train=y_cv_train.values,
                X_val=X_cv_val.values,
                y_val=y_cv_val.values,
                params=params,
                verbose=False
            )

            y_val_pred = predict_neural_network(
                model=model,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                X=X_cv_val.values
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
            "hidden_layers": str(params["hidden_layers"]),
            "dropout": params["dropout"],
            "learning_rate": params["learning_rate"],
            "weight_decay": params["weight_decay"],
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "patience": params["patience"],
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
    "hidden_layers": [
        (32, 16),
        (64, 32),
        (128, 64),
        (128, 64, 32),
        (256, 128, 64)
    ],
    "dropout": [0.0, 0.05, 0.1, 0.2],
    "learning_rate": [0.0003, 0.0005, 0.001, 0.003],
    "weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
    "batch_size": [32, 64, 128, 256],
    "epochs": [150, 250, 400],
    "patience": [25, 40]
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

best_hidden_layers = best_params_stage1["hidden_layers"]
best_dropout = best_params_stage1["dropout"]
best_learning_rate = best_params_stage1["learning_rate"]
best_weight_decay = best_params_stage1["weight_decay"]
best_batch_size = best_params_stage1["batch_size"]
best_epochs = best_params_stage1["epochs"]
best_patience = best_params_stage1["patience"]

hidden_layer_candidates = sorted(set([
    best_hidden_layers,
    (32, 16),
    (64, 32),
    (128, 64),
    (128, 64, 32),
    (256, 128, 64),
    (256, 128, 64, 32)
]))

dropout_candidates = sorted(set([
    max(0.0, best_dropout - 0.05),
    best_dropout,
    min(0.4, best_dropout + 0.05),
    min(0.4, best_dropout + 0.1)
]))

learning_rate_candidates = sorted(set([
    max(0.00005, best_learning_rate * 0.5),
    max(0.00005, best_learning_rate * 0.75),
    best_learning_rate,
    min(0.01, best_learning_rate * 1.25),
    min(0.01, best_learning_rate * 1.5)
]))

weight_decay_candidates = sorted(set([
    0.0,
    best_weight_decay,
    best_weight_decay * 0.5,
    best_weight_decay * 2.0,
    best_weight_decay + 1e-5
]))

batch_size_candidates = sorted(set([
    32,
    64,
    128,
    256,
    best_batch_size
]))

epochs_candidates = sorted(set([
    max(50, best_epochs - 100),
    best_epochs,
    best_epochs + 100
]))

patience_candidates = sorted(set([
    max(10, best_patience - 10),
    best_patience,
    best_patience + 10
]))

param_grid_stage2 = {
    "hidden_layers": hidden_layer_candidates,
    "dropout": dropout_candidates,
    "learning_rate": learning_rate_candidates,
    "weight_decay": weight_decay_candidates,
    "batch_size": batch_size_candidates,
    "epochs": epochs_candidates,
    "patience": patience_candidates
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
    "hyperparameter_results_neural_network_combined_shuffled_80_20.csv",
    index=False
)

top_results = search_results_df.head(10)

print("\nTop 10 hyperparameter combinations:")
print(top_results.to_string(index=False))


# ============================================================
# Final train/validation split inside the 80 % training set
# ============================================================

X_final_train, X_final_val, y_final_train, y_final_val = train_test_split(
    X_train_raw,
    y_train_raw,
    test_size=0.15,
    shuffle=True,
    random_state=1
)

X_final_train = X_final_train.reset_index(drop=True)
X_final_val = X_final_val.reset_index(drop=True)
y_final_train = y_final_train.reset_index(drop=True)
y_final_val = y_final_val.reset_index(drop=True)


# ============================================================
# Final training on 80 % training data
# ============================================================

print("\nTraining final Neural Network model...")

final_model, final_x_scaler, final_y_scaler = train_neural_network(
    X_train=X_final_train.values,
    y_train=y_final_train.values,
    X_val=X_final_val.values,
    y_val=y_final_val.values,
    params=best_params,
    verbose=True
)

print("Final Neural Network model trained.")


# ============================================================
# Prediction on 20 % held-out test set
# ============================================================

y_pred = predict_neural_network(
    model=final_model,
    x_scaler=final_x_scaler,
    y_scaler=final_y_scaler,
    X=X_test_raw.values
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

print("\nResults on scrambled 20 % held-out test set:")

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

for col in feature_columns:
    test_results[col] = X_test_raw[col].values

print("\nFirst 10 test predictions:")
print(test_results.head(10))

test_results.to_csv(
    "predictions_neural_network_combined_shuffled_20_percent_test.csv",
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
    f"Neural Network CH4 prediction on full shuffled 20 % test set\n"
    f"R² = {r2_ch4:.4f}"
)

ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("neural_network_ch4_full_test_plot.png", dpi=300)
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
    f"Neural Network CO2 prediction on full shuffled 20 % test set\n"
    f"R² = {r2_co2:.4f}"
)

ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("neural_network_co2_full_test_plot.png", dpi=300)
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
    f"Neural Network CH4 prediction on displayed subset of test set "
    f"({len(plot_results_subset)} points)\n"
    f"Full test set R² = {r2_ch4:.4f}"
)

ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("neural_network_ch4_subset_test_plot.png", dpi=300)
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
    f"Neural Network CO2 prediction on displayed subset of test set "
    f"({len(plot_results_subset)} points)\n"
    f"Full test set R² = {r2_co2:.4f}"
)

ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("neural_network_co2_subset_test_plot.png", dpi=300)
plt.show()


# ============================================================
# Plot 5: CH4 true vs predicted scatter plot
# ============================================================

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    test_results["True_CH4_ppm"],
    test_results["Predicted_CH4_ppm"],
    s=8,
    alpha=0.5
)

min_value = min(
    test_results["True_CH4_ppm"].min(),
    test_results["Predicted_CH4_ppm"].min()
)

max_value = max(
    test_results["True_CH4_ppm"].max(),
    test_results["Predicted_CH4_ppm"].max()
)

ax.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=1.5,
    label="Perfect prediction"
)

ax.set_xlabel("True CH4 ppm")
ax.set_ylabel("Predicted CH4 ppm")
ax.set_title(
    f"Neural Network CH4: true vs predicted\n"
    f"R² = {r2_ch4:.4f}"
)

ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("neural_network_ch4_true_vs_predicted_scatter.png", dpi=300)
plt.show()


# ============================================================
# Plot 6: CO2 true vs predicted scatter plot
# ============================================================

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    test_results["True_CO2_ppm"],
    test_results["Predicted_CO2_ppm"],
    s=8,
    alpha=0.5
)

min_value = min(
    test_results["True_CO2_ppm"].min(),
    test_results["Predicted_CO2_ppm"].min()
)

max_value = max(
    test_results["True_CO2_ppm"].max(),
    test_results["Predicted_CO2_ppm"].max()
)

ax.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=1.5,
    label="Perfect prediction"
)

ax.set_xlabel("True CO2 ppm")
ax.set_ylabel("Predicted CO2 ppm")
ax.set_title(
    f"Neural Network CO2: true vs predicted\n"
    f"R² = {r2_co2:.4f}"
)

ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("neural_network_co2_true_vs_predicted_scatter.png", dpi=300)
plt.show()