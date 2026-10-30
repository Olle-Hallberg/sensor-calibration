#!/usr/bin/env python3
"""Run the GBM through the same single-session-holdout panel used for the
physical model, so the Goal-5 comparison isn't based on just one holdout
choice (the same discipline applied earlier: CORE-2 alone was not enough
evidence either way)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from fit_joint_model import Xy, fit, model as physical_model

FEATURES = ["lpl_signal", "mpl_signal", "spl_signal", "temp_c", "pressure_bar", "humidity_pct"]

CANDIDATES = [
    ("thesis_CORE-1_dry_214017", "interchangeable"),
    ("thesis_CORE-2_dry_001851", "interchangeable (goal 4/5 original holdout)"),
    ("thesis_atm_luft_dry_164032", "interchangeable"),
    ("thesis_teknisk_luft_dry_142847", "interchangeable"),
    ("thesis_teknisk_luft_dry_195527", "known-hard (thin cold-T region)"),
    ("thesis_chem_N_dry_101014", "known-hard (unique C=0 anchor)"),
]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def invert_physical(df, popt):
    a0, a1, a1b, a2, a3, b, c0, c1, c1b, c2, c3, g_thesis = popt
    T, P, RH = df["temp_c"].values, df["pressure_bar"].values, df["humidity_pct"].values
    g = np.where(df["source"].values == "thesis", g_thesis, 1.0)
    span = a0 + a1 * T + a1b * T**2 + a2 * P + a3 * RH
    baseline = c0 + c1 * T + c1b * T**2 + c2 * P + c3 * RH
    ratio = (df["spl_signal"].values / g - baseline) / span
    C = np.full_like(ratio, np.nan)
    valid = ratio > 0
    C[valid] = -np.log(ratio[valid]) / b
    return C


def gbm_features(df):
    X = df[FEATURES].copy()
    X["is_thesis"] = (df["source"] == "thesis").astype(int)
    return X


def main():
    df = pd.read_csv("../pooled_data.csv")
    print(f"{'session':38s} {'note':38s} {'phys_valid%':>11s} {'phys_RMSE':>10s} {'gbm_RMSE':>9s}")
    for session, note in CANDIDATES:
        train = df[df["session"] != session].reset_index(drop=True)
        test = df[df["session"] == session].reset_index(drop=True)
        true_c = test["co2_ppm"].iloc[0]

        popt = fit(train)
        c_pred = invert_physical(test, popt)
        valid = ~np.isnan(c_pred)
        phys_rmse = rmse(test["co2_ppm"][valid], c_pred[valid]) if valid.sum() else float("nan")

        gbm = HistGradientBoostingRegressor(random_state=0)
        gbm.fit(gbm_features(train), train["co2_ppm"])
        gbm_pred = gbm.predict(gbm_features(test))
        gbm_rmse = rmse(test["co2_ppm"], gbm_pred)

        print(f"{session:38s} {note:38s} {100*valid.mean():10.1f}% {phys_rmse:10.1f} {gbm_rmse:9.1f}  (true C={true_c:.1f} ppm)")


if __name__ == "__main__":
    main()
