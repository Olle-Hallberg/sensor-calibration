#!/usr/bin/env python3
"""Production calibration pipeline for the K96 sensor.

Deployment target: the SAME physical unit used in cont_dev_raw. This
matters for how training data is prepared:

  - cont_dev_raw rows are used as-is (native scale of the deployed unit).
  - thesis_raw rows come from a DIFFERENT physical unit with a
    substantially different, non-uniform per-channel raw signal gain
    (established earlier: LPL/MPL/SPL read ~1.33x/3.34x/3.81x LOWER on the
    thesis rig than on cont_dev, at comparable conditions). Those rows are
    rescaled up to the cont_dev unit's scale before training, so every
    training row looks like what THIS deployed unit would have read.
    Verified this doesn't hurt accuracy vs. the earlier is_thesis-flag
    approach (33.6 vs 32.6 ppm RMSE on the same held-out session) - and
    it's the only one of the two that makes sense at deployment, since a
    real reading from this unit has no "which rig" flag to set.

Two known, deliberately-not-fixed limitations are surfaced in the output
rather than hidden:

  1. CH4 output is flagged unreliable. CH4 and CO2 are ~collinear in the
     training data (no calibration run varied them independently), so the
     CH4 model mostly tracks CO2 via the trained gas mixture's ratio, not
     a validated independent CH4 response. No new data collection is
     planned, so this can't be fixed here - only flagged.
  2. An envelope check flags readings whose (signal, T, P, RH) combination
     falls outside what training ever covered - either a single feature
     out of its trained range, or an in-range combination of features
     that was never seen together (e.g. high concentration signal AND high
     humidity, each individually sampled but not jointly) - the second
     case is exactly why this uses a nearest-neighbor check in addition to
     a plain per-feature bounding box, which a bounding box alone would
     miss.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

FEATURES = ["lpl_signal", "mpl_signal", "spl_signal", "temp_c", "pressure_bar", "humidity_pct"]

# cont_dev / thesis mean-ratio per channel (see module docstring).
CHANNEL_GAIN_TO_CONT_DEV = {
    "lpl_signal": 1.329992,
    "mpl_signal": 3.342829,
    "spl_signal": 3.805869,
}

MODEL_PATH = "k96_calibration_pipeline.joblib"

CH4_WARNING = (
    "CH4 and CO2 are ~collinear in all available training data (every calibration "
    "run used a single gas mixture or a fixed-ratio cylinder - none varied CH4 and "
    "CO2 independently). This prediction tracks CO2 through that trained ratio, not "
    "a validated independent CH4 response. Do not use for any scenario where CH4 "
    "could diverge from CO2 (e.g. a methane source with background CO2)."
)


def rescale_thesis_rows(df):
    """Bring thesis_raw's three signal channels up to the cont_dev unit's
    scale. cont_dev rows pass through unchanged."""
    df = df.copy()
    is_thesis = df["source"] == "thesis"
    for col, factor in CHANNEL_GAIN_TO_CONT_DEV.items():
        df.loc[is_thesis, col] = df.loc[is_thesis, col] * factor
    return df


def build_envelope_check(X_train):
    bounds = {c: (float(X_train[c].min()), float(X_train[c].max())) for c in FEATURES}

    scaler = StandardScaler().fit(X_train[FEATURES])
    Xs = scaler.transform(X_train[FEATURES])
    nn = NearestNeighbors(n_neighbors=6).fit(Xs)  # self (dist 0) + 5 real neighbors
    train_dists, _ = nn.kneighbors(Xs)
    self_5th_nn_dist = train_dists[:, -1]
    # Threshold from the training data's own neighbor-spacing distribution,
    # not an arbitrary constant - tune the percentile if this is too
    # strict/loose in practice.
    threshold = float(np.percentile(self_5th_nn_dist, 99))

    return {"bounds": bounds, "scaler": scaler, "nn": nn, "threshold": threshold}


def check_envelope_batch(X, envelope):
    """X: DataFrame with the FEATURES columns, no NaNs. Vectorized over all
    rows at once - the per-row check_envelope() below is a thin wrapper
    around this for single-reading convenience, not a separate code path.

    Returns (in_envelope, violations, feature_out_of_range). The first two
    combine both checks below into one flag, same as before. On field data
    in_envelope is often False for nearly every row - mostly from check (2),
    which is expected to fire a lot (the lab's clustered experimental design
    and real field conditions are different joint distributions; see the
    module docstring). feature_out_of_range isolates just check (1) - a
    single feature genuinely beyond anything ever trained on - which is the
    much rarer, more serious signal worth telling apart from "merely a
    novel combination of otherwise in-range values." Confirmed on real
    field data: 100% of predicted rows failed check (2), only ~33% also
    failed check (1)."""
    n = len(X)
    violations = [[] for _ in range(n)]

    # (1) Per-feature bounding box - a single feature outside anything
    # ever seen in training. Hard, binary, says nothing about combinations.
    feature_out_of_range = np.zeros(n, dtype=bool)
    for c in FEATURES:
        lo, hi = envelope["bounds"][c]
        bad = (X[c] < lo) | (X[c] > hi)
        feature_out_of_range |= bad.values
        for i in np.flatnonzero(bad.values):
            violations[i].append(f"{c}={X[c].iloc[i]:.4g} outside training range [{lo:.4g}, {hi:.4g}]")

    # (2) Joint nearest-neighbor distance - see docstring above for why
    # this one is tight enough to fire on most field data even when every
    # individual feature is in-range.
    xs = envelope["scaler"].transform(X[FEATURES])
    dist, _ = envelope["nn"].kneighbors(xs, n_neighbors=1)
    dist = dist[:, 0]
    novel = dist > envelope["threshold"]
    for i in np.flatnonzero(novel):
        violations[i].append(
            f"combination of features is novel: nearest-neighbor distance {dist[i]:.2f} "
            f"exceeds training threshold {envelope['threshold']:.2f} (features may be individually "
            f"in-range but this combination was not seen together in training)"
        )

    in_envelope = np.array([len(v) == 0 for v in violations])
    return in_envelope, violations, feature_out_of_range


def check_envelope(reading, envelope):
    X = pd.DataFrame([reading])[FEATURES]
    in_envelope, violations, feature_out_of_range = check_envelope_batch(X, envelope)
    return {
        "in_envelope": bool(in_envelope[0]),
        "violations": violations[0],
        "feature_out_of_range": bool(feature_out_of_range[0]),
    }


def train(data_path="pooled_data.csv", out_path=MODEL_PATH):
    df = pd.read_csv(data_path)
    df = rescale_thesis_rows(df)
    X = df[FEATURES]

    co2_model = HistGradientBoostingRegressor(random_state=0).fit(X, df["co2_ppm"])
    ch4_model = HistGradientBoostingRegressor(random_state=0).fit(X, df["ch4_ppb"])
    envelope = build_envelope_check(X)

    joblib.dump(
        {"co2_model": co2_model, "ch4_model": ch4_model, "envelope": envelope, "features": FEATURES},
        out_path,
    )
    print(f"Trained on {len(df)} rows (cont_dev native + thesis rescaled to cont_dev unit scale)")
    print(f"Saved pipeline to {out_path}")
    return out_path


def load_pipeline(model_path=MODEL_PATH):
    """Loads the artifact saved by train() - does NOT retrain. train() is a
    one-time step (run `python3 production_pipeline.py`); every prediction
    run, on any number of field CSVs, should call this instead."""
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No trained pipeline found at '{model_path}'. Run `python3 production_pipeline.py` "
            f"once to train and save it before predicting."
        ) from None


def predict_batch(X, bundle):
    """X: DataFrame with the FEATURES columns, no NaNs, any number of rows.
    bundle: result of load_pipeline() - load it ONCE outside any per-file or
    per-row loop and reuse it; joblib.load() is not something to call per row."""
    X = X[bundle["features"]]
    co2 = bundle["co2_model"].predict(X)
    ch4 = bundle["ch4_model"].predict(X)
    in_envelope, violations, feature_out_of_range = check_envelope_batch(X, bundle["envelope"])

    return pd.DataFrame({
        "co2_ppm": co2,
        "ch4_ppb": ch4,
        "ch4_reliable": False,
        "ch4_warning": CH4_WARNING,
        "in_envelope": in_envelope,
        "feature_out_of_range": feature_out_of_range,
        "envelope_violations": ["; ".join(v) for v in violations],
    }, index=X.index)


def predict(reading, model_path=MODEL_PATH):
    """Single-reading convenience wrapper - loads the model fresh each call,
    so fine for one-off use, but NOT for scoring a whole file (use
    load_pipeline() + predict_batch() for that - see process_log_csv.py)."""
    bundle = load_pipeline(model_path)
    row = predict_batch(pd.DataFrame([reading]), bundle).iloc[0]
    return {
        "co2_ppm": float(row["co2_ppm"]),
        "ch4_ppb": float(row["ch4_ppb"]),
        "ch4_reliable": False,
        "ch4_warning": CH4_WARNING,
        "in_envelope": bool(row["in_envelope"]),
        "feature_out_of_range": bool(row["feature_out_of_range"]),
        "envelope_violations": row["envelope_violations"].split("; ") if row["envelope_violations"] else [],
    }


if __name__ == "__main__":
    train()

    # Demo readings: start from an ACTUAL training row (post-rescale) rather
    # than hand-picked numbers, so the "normal" case is a real combination
    # the envelope check should recognize.
    demo_df = rescale_thesis_rows(pd.read_csv("pooled_data.csv"))
    sample = demo_df[demo_df["source"] == "cont_dev"].iloc[1000]
    normal_reading = {c: float(sample[c]) for c in FEATURES}

    print()
    print("--- demo: a real training-set reading (should be in-envelope) ---")
    print(predict(normal_reading))

    print()
    print("--- demo: same reading, humidity swapped to a high value never jointly seen at this signal level ---")
    novel_reading = dict(normal_reading)
    novel_reading["humidity_pct"] = 70.0
    print(predict(novel_reading))
