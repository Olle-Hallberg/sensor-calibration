#!/usr/bin/env python3
"""Stage 2 of the practical path: fit the K96's intrinsic concentration
response from cont_dev_raw (dataset 1, the wide concentration sweep).

Dataset 1's segments are NOT at fixed T/P as their filenames suggest - each
one ramps from ~10-15 C (room temp at power-on) up to its T25/T30 setpoint
over the first chunk of the file, simultaneously with the concentration
sweep. Hand-filtering to "steady" rows either leaves the T/P confound in, or
restricts the concentration range enough to wash out the correlation (both
were tried and checked). So this fits T and P jointly with concentration
from the start, using a physically-motivated separable form:

    spl_signal = (a0 + a1*T + a2*P) * exp(-b*C) + (c0 + c1*T + c2*P)

C is co2_ppm - CH4 and CO2 are ~collinear in this dataset (single mixed
calibration gas), so this curve describes the response to "the calibration
gas mixture", not CO2 specifically; see prior discussion on why CH4/CO2
can't be separated from this dataset alone.

Validation is leave-one-segment-out: holding out an entire session (not
random rows, which are autocorrelated) and predicting it from a fit on the
other four.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

SESSIONS = [
    "2026-09-22_12-21-30_P3_T30_ppm0-0.4",
    "2026-09-22_13-18-42_P3_T30_ppm0.5-0.9",
    "2026-09-22_15-14-47_P3_T30_ppm0.9-2.2",
    "2026-09-22_16-07-30_P2.8_T25_ppm2.2-0",
    "2026-09-22_17-25-05_P3.2_T25_ppm0-2.2",
]


def model(X, a0, a1, a2, b, c0, c1, c2):
    C, T, P = X
    span = a0 + a1 * T + a2 * P
    baseline = c0 + c1 * T + c2 * P
    return span * np.exp(-b * C) + baseline


def fit(df):
    X = (df["co2_ppm"].values, df["temp_c"].values, df["pressure_bar"].values)
    y = df["spl_signal"].values
    p0 = [y.max() - y.min(), 0.0, 0.0, 1.0 / 200.0, y.min(), 0.0, 0.0]
    popt, _ = curve_fit(model, X, y, p0=p0, maxfev=20000)
    return popt


def main():
    df = pd.read_csv("../pooled_data.csv")
    cd = df[df["source"] == "cont_dev"].reset_index(drop=True)

    print(f"Fitting on all {len(cd)} cont_dev rows (full T/P range: "
          f"{cd['temp_c'].min():.1f}-{cd['temp_c'].max():.1f} C, "
          f"{cd['pressure_bar'].min():.2f}-{cd['pressure_bar'].max():.2f} bar)")
    popt_full = fit(cd)
    pred_full = model((cd["co2_ppm"].values, cd["temp_c"].values, cd["pressure_bar"].values), *popt_full)
    rmse_full = rmse(cd["spl_signal"], pred_full)
    print(f"In-sample R^2: {r2_score(cd['spl_signal'], pred_full):.4f}, "
          f"RMSE: {rmse_full:,.0f} counts "
          f"(signal range ~{cd['spl_signal'].min():,.0f}-{cd['spl_signal'].max():,.0f})")
    print()
    print("Fitted params (a0,a1,a2,b,c0,c1,c2):")
    print(popt_full)
    print()

    print("Leave-one-segment-out cross-validation:")
    for held_out in SESSIONS:
        train = cd[cd["session"] != held_out]
        test = cd[cd["session"] == held_out]
        popt = fit(train)
        pred = model((test["co2_ppm"].values, test["temp_c"].values, test["pressure_bar"].values), *popt)
        r2 = r2_score(test["spl_signal"], pred)
        test_rmse = rmse(test["spl_signal"], pred)
        print(f"  held out {held_out}: R^2={r2:.4f}  RMSE={test_rmse:,.0f}")

    np.save("concentration_response_params.npy", popt_full)


if __name__ == "__main__":
    main()
