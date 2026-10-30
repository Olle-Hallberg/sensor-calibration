#!/usr/bin/env python3
"""Goal 5: GBM comparison, same train/holdout split as the physical model.

Important direction difference: the physical model (fit_joint_model.py) was
fit FORWARD - predicting spl_signal from (C, T, P, RH) - because that's the
direction the physics (Beer-Lambert) is naturally expressed in. The actual
deployment task is the INVERSE: given measured signals + environment,
estimate concentration. The physical model's forward fit is analytically
invertible (solve the exponential for C), so both models are compared in
concentration space, on the same held-out rows, for a fair comparison.

Features for the GBM: lpl_signal, mpl_signal, spl_signal, temp_c,
pressure_bar, humidity_pct, and an is_thesis flag - mirroring exactly the
information the physical model's g_thesis term uses to absorb the
cross-rig gain difference. Without that flag the GBM would have to infer
the gain switch from signal magnitude alone, confounding it with
concentration; the physical model gets this "for free" as an explicit term,
so the GBM gets the same information explicitly too, for a fair comparison.
Deployment caveat either way: a new, uncalibrated K96 unit needs its own
gain reference before either model's cross-rig handling means anything for
it.

Asymmetry worth noting: the physical model only ever used co2_ppm as its
concentration variable C (cont_dev's CH4/CO2 are ~collinear, so this was a
deliberate, discussed limitation) - it has no CH4 output at all. The GBM
can be trained on ch4_ppb directly, so that comparison is GBM-only, flagged
as such below.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fit_joint_model import model as physical_model


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


FEATURES = ["lpl_signal", "mpl_signal", "spl_signal", "temp_c", "pressure_bar", "humidity_pct"]


def invert_physical_model(df, popt):
    a0, a1, a1b, a2, a3, b, c0, c1, c1b, c2, c3, g_thesis = popt
    T = df["temp_c"].values
    P = df["pressure_bar"].values
    RH = df["humidity_pct"].values
    g = np.where(df["source"].values == "thesis", g_thesis, 1.0)
    span = a0 + a1 * T + a1b * T**2 + a2 * P + a3 * RH
    baseline = c0 + c1 * T + c1b * T**2 + c2 * P + c3 * RH
    ratio = (df["spl_signal"].values / g - baseline) / span
    C = np.full_like(ratio, np.nan)
    valid = ratio > 0
    C[valid] = -np.log(ratio[valid]) / b
    return C


def make_gbm_features(df):
    X = df[FEATURES].copy()
    X["is_thesis"] = (df["source"] == "thesis").astype(int)
    return X


def summarize(y_true, y_pred, label, unit):
    true_val = np.asarray(y_true)[0] if np.ptp(y_true) == 0 else None
    bias = float(np.mean(y_pred) - np.mean(y_true))
    spread = float(np.std(y_pred))
    print(f"{label}: RMSE={rmse(y_true, y_pred):.2f} {unit}  MAE={mean_absolute_error(y_true, y_pred):.2f} {unit}  "
          f"bias={bias:+.2f} {unit}  pred_std={spread:.2f} {unit}"
          + (f"  (true={true_val:.1f}, constant)" if true_val is not None else ""))


def main():
    train = pd.read_csv("train_data.csv")
    test = pd.read_csv("holdout_data.csv")
    popt = np.load("joint_model_params.npy")

    print(f"Train: {len(train)} rows, Holdout ({test['session'].iloc[0]}): {len(test)} rows")
    print("NOTE: this holdout session is cylinder_constant - co2_ppm/ch4_ppb have ZERO variance")
    print("here (true value is the same for every row), so R^2 is not a valid metric (it divides")
    print("by target variance). Using RMSE/MAE/bias instead, which stay meaningful against a")
    print("constant truth.")
    print()

    # --- Physical model, inverted to concentration space ---
    c_pred_phys = invert_physical_model(test, popt)
    valid = ~np.isnan(c_pred_phys)
    print(f"Physical model (inverted) on holdout: {valid.sum()}/{len(test)} rows gave a valid inversion")
    summarize(test["co2_ppm"][valid], c_pred_phys[valid], "  Physical (inverted) CO2", "ppm")
    print()

    # --- GBM, trained directly on signals+environment -> concentration ---
    X_train = make_gbm_features(train)
    X_test = make_gbm_features(test)

    for target, unit in [("co2_ppm", "ppm"), ("ch4_ppb", "ppb")]:
        gbm = HistGradientBoostingRegressor(random_state=0)
        gbm.fit(X_train, train[target])
        pred = gbm.predict(X_test)
        tag = "" if target == "co2_ppm" else "  (GBM-only - physical model has no CH4 output)"
        summarize(test[target], pred, f"GBM {target}{tag}", unit)
        imp = permutation_importance(gbm, X_test, test[target], n_repeats=10, random_state=0,
                                      scoring="neg_root_mean_squared_error")
        order = np.argsort(imp.importances_mean)[::-1]
        print("  permutation importance (RMSE increase when shuffled, ppm/ppb - higher = more important):")
        for i in order:
            print(f"    {X_test.columns[i]:14s} {-imp.importances_mean[i]:.3f} +/- {imp.importances_std[i]:.3f}")
        if target == "co2_ppm":
            gbm_co2_pred = pred

    print()
    print("Head-to-head on CO2, same held-out session (RMSE, lower is better):")
    print(f"  Physical (inverted): {rmse(test['co2_ppm'][valid], c_pred_phys[valid]):.2f} ppm")
    print(f"  GBM:                 {rmse(test['co2_ppm'], gbm_co2_pred):.2f} ppm")


if __name__ == "__main__":
    main()
