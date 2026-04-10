import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# #seed
np.random.seed(1)
torch.manual_seed(1)

# #device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# #model
class GasNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim=1):
        super(GasNet, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))  # #linear
            layers.append(nn.ReLU())  # #relu
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))  # #output
        self.network = nn.Sequential(*layers)  # #stack

    def forward(self, X):
        return self.network(X)  # #forward


# #train
def train_model(model, X_train, y_train, learning_rate, batch_size, epochs):
    criterion = nn.MSELoss()  # #loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # #adam

    dataset = torch.utils.data.TensorDataset(X_train, y_train)  # #dataset
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )  # #loader

    model.train()  # #trainmode

    for _ in range(epochs):  # #epochs
        for X_batch, y_batch in dataloader:  # #batches
            X_batch = X_batch.to(device)  # #xgpu
            y_batch = y_batch.to(device)  # #ygpu

            optimizer.zero_grad()  # #zero
            loss = criterion(model(X_batch), y_batch)  # #calc
            loss.backward()  # #backprop
            optimizer.step()  # #step


# #predict
def predict_model(model, X):
    model.eval()  # #eval
    with torch.no_grad():  # #nograd
        return model(X.to(device)).cpu().numpy().flatten()  # #pred


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

# #tensor_final
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled.reshape(-1, 1), dtype=torch.float32)

# #grid
param_grid = {
    # hidden_layers -> model depth/width
    # learning_rate -> optimizer step size
    # batch_size -> samples per update
    # epochs -> full training passes
    # change hyperparameters to tighter intervall
    "hidden_layers": [[], [50], [100], [200], [200, 100], [200, 100, 60, 30]],
    "learning_rate": [0.001, 0.005, 0.01],
    "batch_size": [16, 32, 64],
    "epochs": [100, 200, 400]
}

# #cv
cv = KFold(n_splits=5, shuffle=True, random_state=1)

best_score = np.inf
best_params = None
search_results = []

all_param_combinations = list(ParameterGrid(param_grid))
total_runs = len(all_param_combinations)

print("Starting neural network hyperparameter search...\n")

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

        # #fold_tensor
        X_cv_train = torch.tensor(X_cv_train_scaled, dtype=torch.float32)
        X_cv_val = torch.tensor(X_cv_val_scaled, dtype=torch.float32)
        y_cv_train = torch.tensor(y_cv_train_scaled.reshape(-1, 1), dtype=torch.float32)
        y_cv_val = torch.tensor(y_cv_val_scaled.reshape(-1, 1), dtype=torch.float32)

        model = GasNet(
            input_dim=X_cv_train.shape[1],
            hidden_layers=params["hidden_layers"]
        ).to(device)

        train_model(
            model=model,
            X_train=X_cv_train,
            y_train=y_cv_train,
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            epochs=params["epochs"]
        )

        y_val_pred_scaled = predict_model(model, X_cv_val)
        y_val_true_scaled = y_cv_val.cpu().numpy().flatten()

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
        "hidden_layers": str(params["hidden_layers"]),
        "learning_rate": params["learning_rate"],
        "batch_size": params["batch_size"],
        "epochs": params["epochs"],
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
search_results_df.to_csv("hyperparameter_results.csv", index=False)

print("\nBest hyperparameters:")
print(best_params)
print(f"\nBest CV MAE: {best_score:.3f} ppm")

# #finalmodel
best_model = GasNet(
    input_dim=X_train_tensor.shape[1],
    hidden_layers=best_params["hidden_layers"]
).to(device)

train_model(
    model=best_model,
    X_train=X_train_tensor,
    y_train=y_train_tensor,
    learning_rate=best_params["learning_rate"],
    batch_size=best_params["batch_size"],
    epochs=best_params["epochs"]
)

# #testpred
y_pred_scaled = predict_model(best_model, X_test_tensor)
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

test_results.to_csv(f"predictions_test_{target_column}.csv", index=False)

# #saveall
X_all_scaled = x_scaler.transform(X)
X_all_tensor = torch.tensor(X_all_scaled, dtype=torch.float32)

y_all_pred_scaled = predict_model(best_model, X_all_tensor)
y_all_pred = y_scaler.inverse_transform(y_all_pred_scaled.reshape(-1, 1)).flatten()

all_results = pd.DataFrame({
    "sensorsignal": X["sensorsignal"].values,
    "pressure": X["pressure"].values,
    "humidity": X["humidity"].values,
    "temperature": X["temperature"].values,
    "True_ppm": y.values,
    "Predicted_ppm": y_all_pred
})

all_results.to_csv(f"predictions_all_{target_column}.csv", index=False)

# #plot_timeseries
all_plot_df = all_results.copy()
all_plot_df = all_plot_df.sort_index()

x_axis = np.arange(len(all_plot_df))

fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(
    x_axis,
    all_plot_df["True_ppm"].values,
    label="reference"
)

ax1.plot(
    x_axis,
    all_plot_df["Predicted_ppm"].values,
    label="calibrated CH4 concentration"
)

ax1.set_xlabel("Sample index")
ax1.set_ylabel("CH4 concentration (ppm)")
ax1.set_title(f"Temperature Compensation\nR² = {r2:.3f}")

ax2 = ax1.twinx()

ax2.plot(
    x_axis,
    all_plot_df["temperature"].values,
    label="Temperature"
)

ax2.set_ylabel("Temperature (°C)")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.tight_layout()
plt.savefig(f"time_series_all_{target_column}.png", dpi=300)
plt.show()