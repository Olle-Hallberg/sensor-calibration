#!/usr/bin/env python3
"""Stage 3 of the practical path: fit the T/P/RH environmental dependence
from thesis_raw's cylinder sessions (dataset 2's wide T/P/RH, narrow/fixed
concentration data).

The Stage-2 model (fit on cont_dev_raw alone) extrapolates badly onto
thesis's much wider T range (6.9-40.4 C vs cont_dev's 10.4-30.7 C) and
partially-overlapping P range - residuals were ~-3.5M to -6M against a
signal range of ~5-8.5M, i.e. not a small correction. So this re-fits the
same separable form fresh on thesis data alone, now adding a humidity term
(RH never varied in cont_dev, so this is the first fit that can see it):

    spl_signal = (a0 + a1*T + a2*P + a3*RH) * exp(-b*C) + (c0 + c1*T + c2*P + c3*RH)

C is each session's known certified cylinder concentration (co2_ppm) - held
fixed within a session, varying only across the ~6 cylinder levels, so this
fit's main job is pinning down the T/P/RH terms; b is weakly identified at
best from only 6 concentration points and gets properly resolved in Stage 4
(the joint refit on the full pooled data).

thesis_test is NOT used here - it's reserved as the held-out evaluation set
(goal 4), not for fitting.

Validation: leave-one-session-out across the 8 cylinder sessions.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def model(X, a0, a1, a2, a3, b, c0, c1, c2, c3):
    C, T, P, RH = X
    span = a0 + a1 * T + a2 * P + a3 * RH
    baseline = c0 + c1 * T + c2 * P + c3 * RH
    return span * np.exp(-b * C) + baseline


def fit(df):
    X = (df["co2_ppm"].values, df["temp_c"].values, df["pressure_bar"].values, df["humidity_pct"].values)
    y = df["spl_signal"].values
    p0 = [y.max() - y.min(), 0.0, 0.0, 0.0, 1.0 / 200.0, y.min(), 0.0, 0.0, 0.0]
    popt, _ = curve_fit(model, X, y, p0=p0, maxfev=40000)
    return popt


def main():
    df = pd.read_csv("../pooled_data.csv")
    th = df[(df["source"] == "thesis") & (df["gas_source"] == "cylinder_constant")].reset_index(drop=True)
    sessions = sorted(th["session"].unique())

    print(f"Fitting on {len(th)} thesis cylinder rows across {len(sessions)} sessions")
    print(f"T range: {th['temp_c'].min():.1f}-{th['temp_c'].max():.1f} C, "
          f"P range: {th['pressure_bar'].min():.2f}-{th['pressure_bar'].max():.2f} bar, "
          f"RH range: {th['humidity_pct'].min():.1f}-{th['humidity_pct'].max():.1f} %")
    print()

    popt_full = fit(th)
    X_full = (th["co2_ppm"].values, th["temp_c"].values, th["pressure_bar"].values, th["humidity_pct"].values)
    pred_full = model(X_full, *popt_full)
    print(f"In-sample R^2: {r2_score(th['spl_signal'], pred_full):.4f}, "
          f"RMSE: {rmse(th['spl_signal'], pred_full):,.0f} counts "
          f"(signal range ~{th['spl_signal'].min():,.0f}-{th['spl_signal'].max():,.0f})")
    print()
    names = ["a0", "a1(T)", "a2(P)", "a3(RH)", "b", "c0", "c1(T)", "c2(P)", "c3(RH)"]
    for n, v in zip(names, popt_full):
        print(f"  {n}: {v:.6g}")
    print()

    print("Leave-one-session-out cross-validation:")
    for held_out in sessions:
        train = th[th["session"] != held_out]
        test = th[th["session"] == held_out]
        popt = fit(train)
        X_test = (test["co2_ppm"].values, test["temp_c"].values, test["pressure_bar"].values, test["humidity_pct"].values)
        pred = model(X_test, *popt)
        r2 = r2_score(test["spl_signal"], pred)
        test_rmse = rmse(test["spl_signal"], pred)
        print(f"  held out {held_out}: R^2={r2:.4f}  RMSE={test_rmse:,.0f}  "
              f"(C={test['co2_ppm'].iloc[0]:.1f} ppm, n={len(test)})")

    np.save("environmental_correction_params.npy", popt_full)


if __name__ == "__main__":
    main()
