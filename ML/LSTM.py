import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# #seed
np.random.seed(1)
torch.manual_seed(1)

# #device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #seq
def create_sequences(X_array, y_array, sequence_length):
    X_sequences = []
    y_sequences = []

    # inkluderar nuvarande tidssteg X[i]
    for i in range(sequence_length - 1, len(X_array)):
        X_sequences.append(X_array[i - sequence_length + 1:i + 1])
        y_sequences.append(y_array[i])

    return np.array(X_sequences), np.array(y_sequences)

# #model
class GasLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout, output_dim=1):
        super(GasLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, X):
        lstm_out, _ = self.lstm(X)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

# #train
def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    learning_rate,
    batch_size,
    epochs,
    patience=10,
    min_delta=1e-4
):
    criterion = nn.HuberLoss()  # #loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # #adam

    dataset = torch.utils.data.TensorDataset(X_train, y_train)  # #dataset
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )  # #loader

    best_val_loss = np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):  # #epochs
        model.train()  # #trainmode

        for X_batch, y_batch in dataloader:  # #batches
            X_batch = X_batch.to(device)  # #xgpu
            y_batch = y_batch.to(device)  # #ygpu

            optimizer.zero_grad()  # #zero
            loss = criterion(model(X_batch), y_batch)  # #calc
            loss.backward()  # #backprop
            optimizer.step()  # #step

        model.eval()  # #eval
        with torch.no_grad():  # #nograd
            val_loss = criterion(
                model(X_val.to(device)),
                y_val.to(device)
            ).item()

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping after epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

# #predict
def predict_model(model, X):
    model.eval()  # #eval
    with torch.no_grad():  # #nograd
        return model(X.to(device)).cpu().numpy().flatten()  # #pred

# #cv_eval
def evaluate_param_grid(X_train_raw, y_train_raw, feature_columns, param_grid, cv, stage_name):
    best_score = np.inf
    best_params = None
    search_results = []

    all_param_combinations = list(ParameterGrid(param_grid))
    total_runs = len(all_param_combinations)

    print(f"Starting {stage_name} hyperparameter search...\n")

    for run_number, params in enumerate(all_param_combinations, start=1):
        print(f"Running {stage_name} combination {run_number}/{total_runs}: {params}")

        cv_scores = []
        sequence_length = params["sequence_length"]

        for fold_number, (train_idx, val_idx) in enumerate(cv.split(X_train_raw), start=1):
            # #fold_split
            X_cv_train_raw = X_train_raw.iloc[train_idx].copy()
            X_cv_val_raw = X_train_raw.iloc[val_idx].copy()

            y_cv_train_raw = y_train_raw.iloc[train_idx].copy()
            y_cv_val_raw = y_train_raw.iloc[val_idx].copy()

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

            # #fold_seq_train
            X_cv_train_seq, y_cv_train_seq = create_sequences(
                X_cv_train_scaled,
                y_cv_train_scaled,
                sequence_length=sequence_length
            )

            # #fold_seq_val_context
            if len(X_cv_train_scaled) < sequence_length:
                cv_scores = []
                break

            X_cv_val_with_context = np.vstack([
                X_cv_train_scaled[-(sequence_length - 1):],
                X_cv_val_scaled
            ])

            y_cv_val_with_context = np.concatenate([
                y_cv_train_scaled[-(sequence_length - 1):],
                y_cv_val_scaled
            ])

            X_cv_val_seq, y_cv_val_seq = create_sequences(
                X_cv_val_with_context,
                y_cv_val_with_context,
                sequence_length=sequence_length
            )

            if len(X_cv_train_seq) == 0 or len(X_cv_val_seq) == 0:
                cv_scores = []
                break

            # #fold_tensor
            X_cv_train_tensor = torch.tensor(X_cv_train_seq, dtype=torch.float32)
            X_cv_val_tensor = torch.tensor(X_cv_val_seq, dtype=torch.float32)

            y_cv_train_tensor = torch.tensor(
                y_cv_train_seq.reshape(-1, 1),
                dtype=torch.float32
            )
            y_cv_val_tensor = torch.tensor(
                y_cv_val_seq.reshape(-1, 1),
                dtype=torch.float32
            )

            # #fold_model
            model = GasLSTM(
                input_dim=len(feature_columns),
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
                output_dim=1
            ).to(device)

            # #fold_train
            train_model(
                model=model,
                X_train=X_cv_train_tensor,
                y_train=y_cv_train_tensor,
                X_val=X_cv_val_tensor,
                y_val=y_cv_val_tensor,
                learning_rate=params["learning_rate"],
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                patience=params["patience"]
            )

            # #fold_pred
            y_val_pred_scaled = predict_model(model, X_cv_val_tensor)
            y_val_true_scaled = y_cv_val_tensor.cpu().numpy().flatten()

            y_val_pred = y_scaler_cv.inverse_transform(
                y_val_pred_scaled.reshape(-1, 1)
            ).flatten()

            y_val_true = y_scaler_cv.inverse_transform(
                y_val_true_scaled.reshape(-1, 1)
            ).flatten()

            fold_mae = mean_absolute_error(y_val_true, y_val_pred)
            cv_scores.append(fold_mae)

            print(
                f"  Fold {fold_number}/{cv.get_n_splits()}: "
                f"MAE = {fold_mae:.3f} ppm"
            )

        if len(cv_scores) == 0:
            continue

        mean_cv_score = np.mean(cv_scores)

        search_results.append({
            "sequence_length": params["sequence_length"],
            "hidden_size": params["hidden_size"],
            "num_layers": params["num_layers"],
            "dropout": params["dropout"],
            "learning_rate": params["learning_rate"],
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "patience": params["patience"],
            "cv_mae": mean_cv_score,
            "stage": stage_name
        })

        print(f"  Mean CV MAE = {mean_cv_score:.3f} ppm\n")

        if mean_cv_score < best_score:
            best_score = mean_cv_score
            best_params = params
            print(
                f"[{run_number}/{total_runs}] New best -> "
                f"MAE = {best_score:.3f} ppm, params = {best_params}\n"
            )

    return best_score, best_params, search_results

# #load
data = pd.read_csv("gas_sensor_data.csv")

# #target
target_column = "methane_ppm"
feature_columns = ["sensorsignal", "pressure", "humidity", "temperature"]

# #select
X = data[feature_columns].copy()
y = data[target_column].copy()

# #clean
valid_idx = X.notna().all(axis=1) & y.notna()
X = X.loc[valid_idx].reset_index(drop=True)
y = y.loc[valid_idx].reset_index(drop=True)

# #split
train_fraction = 0.75
split_index = int(len(X) * train_fraction)

X_train_raw = X.iloc[:split_index].copy()
X_test_raw = X.iloc[split_index:].copy()

y_train_raw = y.iloc[:split_index].copy()
y_test_raw = y.iloc[split_index:].copy()

# #stage1_grid
param_grid_stage1 = {
    # sequence_length -> antal steg inklusive nuvarande tidpunkt
    # hidden_size -> storlek på hidden state
    # num_layers -> antal LSTM-lager
    # dropout -> regularisering mellan lager
    # learning_rate -> steglängd för optimizer
    # batch_size -> antal sekvenser per batch
    # epochs -> max antal epochs
    # patience -> early stopping
    "sequence_length": [8, 12, 20, 32],
    "hidden_size": [32, 64, 96],
    "num_layers": [1, 2],
    "dropout": [0.0, 0.1],
    "learning_rate": [0.001, 0.003, 0.005],
    "batch_size": [16, 32],
    "epochs": [60],
    "patience": [10]
}

# #stage1_cv
cv_stage1 = TimeSeriesSplit(n_splits=3)

# #stage1_search
best_score_stage1, best_params_stage1, search_results_stage1 = evaluate_param_grid(
    X_train_raw=X_train_raw,
    y_train_raw=y_train_raw,
    feature_columns=feature_columns,
    param_grid=param_grid_stage1,
    cv=cv_stage1,
    stage_name="stage1"
)

if best_params_stage1 is None:
    raise ValueError("No valid hyperparameter combinations found in stage 1.")

print("\nBest stage 1 hyperparameters:")
print(best_params_stage1)
print(f"\nBest stage 1 CV MAE: {best_score_stage1:.3f} ppm")

# #stage2_grid
best_seq = best_params_stage1["sequence_length"]
best_hidden = best_params_stage1["hidden_size"]
best_layers = best_params_stage1["num_layers"]
best_lr = best_params_stage1["learning_rate"]

seq_candidates = sorted(set([max(4, best_seq - 4), best_seq, best_seq + 4]))
hidden_candidates = sorted(set([max(16, best_hidden // 2), best_hidden, best_hidden * 2]))
layer_candidates = sorted(set([best_layers, min(best_layers + 1, 3)]))
lr_candidates = sorted(set([best_lr / 2, best_lr, min(best_lr * 2, 0.01)]))

param_grid_stage2 = {
    "sequence_length": seq_candidates,
    "hidden_size": hidden_candidates,
    "num_layers": layer_candidates,
    "dropout": [0.0, 0.1],
    "learning_rate": lr_candidates,
    "batch_size": [16],
    "epochs": [80],
    "patience": [12]
}

# #stage2_cv
cv_stage2 = TimeSeriesSplit(n_splits=4)

# #stage2_search
best_score_stage2, best_params_stage2, search_results_stage2 = evaluate_param_grid(
    X_train_raw=X_train_raw,
    y_train_raw=y_train_raw,
    feature_columns=feature_columns,
    param_grid=param_grid_stage2,
    cv=cv_stage2,
    stage_name="stage2"
)

if best_params_stage2 is None:
    raise ValueError("No valid hyperparameter combinations found in stage 2.")

print("\nBest stage 2 hyperparameters:")
print(best_params_stage2)
print(f"\nBest stage 2 CV MAE: {best_score_stage2:.3f} ppm")

# #savegrid
search_results_all = search_results_stage1 + search_results_stage2
search_results_df = pd.DataFrame(search_results_all)
search_results_df = search_results_df.sort_values(["stage", "cv_mae"], ascending=[True, True])
search_results_df.to_csv("hyperparameter_results_lstm.csv", index=False)

# #final_scale
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train_raw)
X_test_scaled = x_scaler.transform(X_test_raw)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train_raw.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test_raw.values.reshape(-1, 1)).flatten()

# #final_seq_train
seq_len = best_params_stage2["sequence_length"]

X_train_seq, y_train_seq = create_sequences(
    X_train_scaled,
    y_train_scaled,
    sequence_length=seq_len
)

# #final_seq_test_context
X_test_with_context = np.vstack([
    X_train_scaled[-(seq_len - 1):],
    X_test_scaled
])

y_test_with_context = np.concatenate([
    y_train_scaled[-(seq_len - 1):],
    y_test_scaled
])

X_test_seq, y_test_seq = create_sequences(
    X_test_with_context,
    y_test_with_context,
    sequence_length=seq_len
)

if len(X_train_seq) == 0 or len(X_test_seq) == 0:
    raise ValueError("Sequence length is too large for final train/test split.")

# #final_val_split
val_fraction = 0.15
val_size = max(1, int(len(X_train_seq) * val_fraction))

X_final_train_seq = X_train_seq[:-val_size]
y_final_train_seq = y_train_seq[:-val_size]

X_final_val_seq = X_train_seq[-val_size:]
y_final_val_seq = y_train_seq[-val_size:]

# #tensor_final
X_final_train_tensor = torch.tensor(X_final_train_seq, dtype=torch.float32)
y_final_train_tensor = torch.tensor(y_final_train_seq.reshape(-1, 1), dtype=torch.float32)

X_final_val_tensor = torch.tensor(X_final_val_seq, dtype=torch.float32)
y_final_val_tensor = torch.tensor(y_final_val_seq.reshape(-1, 1), dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_seq.reshape(-1, 1), dtype=torch.float32)

# #finalmodel
best_model = GasLSTM(
    input_dim=len(feature_columns),
    hidden_size=best_params_stage2["hidden_size"],
    num_layers=best_params_stage2["num_layers"],
    dropout=best_params_stage2["dropout"],
    output_dim=1
).to(device)

# #finaltrain
train_model(
    model=best_model,
    X_train=X_final_train_tensor,
    y_train=y_final_train_tensor,
    X_val=X_final_val_tensor,
    y_val=y_final_val_tensor,
    learning_rate=best_params_stage2["learning_rate"],
    batch_size=best_params_stage2["batch_size"],
    epochs=best_params_stage2["epochs"],
    patience=best_params_stage2["patience"]
)

# #testpred
y_pred_scaled = predict_model(best_model, X_test_tensor)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_original = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

# #metrics
mae = mean_absolute_error(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
r2 = r2_score(y_test_original, y_pred)

print(f"\nResults for target: {target_column}")
print(f"MAE  = {mae:.3f} ppm")
print(f"RMSE = {rmse:.3f} ppm")
print(f"R²   = {r2:.3f}")

# #savetest
test_results = pd.DataFrame({
    "True_ppm": y_test_original,
    "Predicted_ppm": y_pred
})

print("\nFirst 10 test predictions:")
print(test_results.head(10))

test_results.to_csv(f"predictions_test_lstm_{target_column}.csv", index=False)

# #saveall
X_all_scaled = x_scaler.transform(X)
y_all_scaled = y_scaler.transform(y.values.reshape(-1, 1)).flatten()

X_all_seq, y_all_seq = create_sequences(
    X_all_scaled,
    y_all_scaled,
    sequence_length=seq_len
)

X_all_tensor = torch.tensor(X_all_seq, dtype=torch.float32)

y_all_pred_scaled = predict_model(best_model, X_all_tensor)
y_all_pred = y_scaler.inverse_transform(y_all_pred_scaled.reshape(-1, 1)).flatten()
y_all_true = y_scaler.inverse_transform(y_all_seq.reshape(-1, 1)).flatten()

all_results = pd.DataFrame({
    "sensorsignal": X.iloc[seq_len - 1:]["sensorsignal"].values,
    "pressure": X.iloc[seq_len - 1:]["pressure"].values,
    "humidity": X.iloc[seq_len - 1:]["humidity"].values,
    "temperature": X.iloc[seq_len - 1:]["temperature"].values,
    "True_ppm": y_all_true,
    "Predicted_ppm": y_all_pred
})

all_results.to_csv(f"predictions_all_lstm_{target_column}.csv", index=False)

# #plot_timeseries
all_plot_df = all_results.copy().reset_index(drop=True)
x_axis = np.arange(len(all_plot_df))

fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(
    x_axis,
    all_plot_df["True_ppm"].values,
    label="reference",
    color="tab:blue",
    linewidth=1.6
)

ax1.plot(
    x_axis,
    all_plot_df["Predicted_ppm"].values,
    label="calibrated CH4 concentration",
    color="tab:orange",
    linewidth=1.6
)

ax1.set_xlabel("Sample index")
ax1.set_ylabel("CH4 concentration (ppm)")
ax1.set_title(f"Temperature Compensation - LSTM\nR² = {r2:.3f}")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(
    x_axis,
    all_plot_df["temperature"].values,
    label="Temperature",
    color="tab:green",
    alpha=0.7,
    linewidth=1.2
)
ax2.set_ylabel("Temperature (°C)")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.tight_layout()
plt.savefig(f"time_series_all_lstm_{target_column}.png", dpi=300)
plt.show()