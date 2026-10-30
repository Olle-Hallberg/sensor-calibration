#!/usr/bin/env python3
"""Robustness check for the Goal-4 held-out design: repeat the same
single-session-holdout evaluation for every thesis dry cylinder session
that could, in principle, have played CORE-2's role (dry/low-RH, not a
cont_dev segment, not chem_N's unique C=0 anchor), to see whether the
held-out score (R^2=0.751 for CORE-2) is a one-off or representative.

Not included, and why:
  - cont_dev segments: each covers a disjoint slice of the concentration
    sweep (e.g. segment 1 is the ONLY data in 0-0.4 ppm); holding one out
    removes unique concentration coverage, not a like-for-like swap.
  - thesis humid sessions: only 2 sessions ever show real RH variation;
    holding either out removes half the RH support, which is a different
    (and already-known-fragile) failure mode, not a check of "could this
    have been chosen instead of CORE-2".
  - chem_N: the only C=0 anchor anywhere in either dataset - included here
    but flagged separately, since losing it is expected to be a materially
    harder case than losing any other single dry session.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from fit_joint_model import Xy, fit, model

CANDIDATES = [
    ("thesis_CORE-1_dry_214017", "interchangeable with CORE-2"),
    ("thesis_CORE-2_dry_001851", "interchangeable with CORE-2 (original choice)"),
    ("thesis_atm_luft_dry_164032", "interchangeable with CORE-2"),
    ("thesis_teknisk_luft_dry_142847", "interchangeable with CORE-2"),
    ("thesis_teknisk_luft_dry_195527", "interchangeable with CORE-2 (small, 478 rows)"),
    ("thesis_chem_N_dry_101014", "UNIQUE C=0 anchor - expect worse, not a like-for-like swap"),
]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    df = pd.read_csv("../pooled_data.csv")
    results = []
    for session, note in CANDIDATES:
        train = df[df["session"] != session].reset_index(drop=True)
        test = df[df["session"] == session].reset_index(drop=True)
        popt = fit(train)
        X_test, y_test = Xy(test)
        pred = model(X_test, *popt)
        r2 = r2_score(y_test, pred)
        test_rmse = rmse(y_test, pred)
        c = test["co2_ppm"].iloc[0]
        results.append((session, c, len(test), r2, test_rmse, note))
        print(f"{session:38s} C={c:6.1f} ppm  n={len(test):5d}  R^2={r2:7.4f}  RMSE={test_rmse:9,.0f}  [{note}]")

    print()
    comparable = [r for r in results if "UNIQUE" not in r[5]]
    r2s = [r[3] for r in comparable]
    print(f"Interchangeable-session R^2: mean={np.mean(r2s):.4f}  std={np.std(r2s):.4f}  "
          f"range=[{min(r2s):.4f}, {max(r2s):.4f}]")


if __name__ == "__main__":
    main()
