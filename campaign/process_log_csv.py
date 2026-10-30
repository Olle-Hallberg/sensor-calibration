#!/usr/bin/env python3
"""Middle stage of the bigger pipeline: jsonl -> [this raw CSV] -> **predict** -> analysis/plots.

Input: a raw K96 telemetry CSV in the field-log schema (capturedAt,
data.sensorData.K96_*, data.chamberPressureBar, data.humidityRh_k96, ... -
see FCT_log_26_10_13.csv). This is the SAME schema cont_dev_raw's training
files used, produced live in the field by an already-working jsonl->CSV
converter upstream - there is no reference instrument here, unlike the
training data.

Output: the input CSV, byte-for-byte the same columns and row order, with
prediction columns appended:
    pred_co2_ppm, pred_ch4_ppb, ch4_reliable, ch4_warning,
    in_envelope, envelope_violations

Two things this does that a naive "call predict() per row" wrapper would
get wrong:

1. Not every row has sensor data. Field logs mix real telemetry rows with
   housekeeping events (session/connection/command - see event counts in
   FCT_log_26_10_13.csv: ~3% of rows). Those rows are kept in the output
   unchanged, with blank prediction columns - they're not fed to the model
   (there's nothing to feed it) and not counted as envelope violations.

2. The same hardware glitches found in training can recur here (isolated
   chamberPressureBar dropouts to ~-8 bar, an NTC startup sentinel of
   exactly 0.0) - but this script does NOT interpolate over them the way
   training does. During lab collection we could look at the rig and
   confirm those were sensor/software artifacts before reconstructing a
   value; the field setup runs unsupervised from a ground station, so a
   suspicious reading can't be confirmed as an artifact rather than a
   genuine (if unusual) physical event - a 0.0 temp reading might be a real
   freezing point, not the sentinel. So these are flagged and left as
   missing data (no prediction for that row) instead of being filled in.

3. A nonzero signal doesn't by itself mean the chamber holds a stable,
   trustworthy sample. The gate is data.pressureStateName == "MEASUREMENT"
   (singular) - NOT the broader data.mode == "MEASUREMENTS" (plural), which
   was tried first and found to be far too permissive: mode=="MEASUREMENTS"
   actually covers an entire pressure-cycling sequence (PREPRESSURISATION ->
   COMPRESSION -> AIR_EXCHANGE -> MEASUREMENT -> back to STANDBY), and the
   optical sensor keeps reporting a nonzero signal throughout ALL of those
   sub-phases, not just the final one - confirmed on real field data: of
   13,098 rows with mode=="MEASUREMENTS", only 606 (4.6%) were actually in
   the MEASUREMENT sub-phase; the other ~95% were mid-compression or
   mid-air-exchange, i.e. a chamber whose contents are still actively
   changing, not a stable sample. Rows outside the MEASUREMENT sub-phase
   get no prediction, regardless of what their signal looks like.

   This does NOT hold in the lab training data (cont_dev_raw) - there,
   pressureStateName is mostly "STANDBY" even during rows we know are
   good, physically valid signal (confirmed: 2996/3007 in one lab file).
   That's expected, not a contradiction: the lab rig was run under manual/
   direct command, where this field doesn't mean anything, rather than
   through the field firmware's autonomous pressure-cycling state machine.
   Don't be confused by that if you go checking it against training data -
   this rule is deliberately field-only, and training (harmonize_pool.py)
   correctly does not apply it.

Batches all predictions and the envelope's nearest-neighbor lookups across
the whole file at once - the model is loaded once, not per row.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from production_pipeline import FEATURES as FEATURES_FOR_AGG
from production_pipeline import load_pipeline, predict_batch

RAW_TO_FEATURE = {
    "data.sensorData.K96_LPL_Signal": "lpl_signal",
    "data.sensorData.K96_MPL_Signal": "mpl_signal",
    "data.sensorData.K96_SPL_Signal": "spl_signal",
    "data.chamberPressureBar": "pressure_bar",
    "data.humidityRh_k96": "humidity_pct",
}
NTC_COLS = ["data.sensorData.K96_NTC0_Temp", "data.sensorData.K96_NTC1_Temp"]
# NOT data.mode - see module docstring point 3 for why. data.mode=="MEASUREMENTS"
# covers the whole pressure-cycling sequence; only the "MEASUREMENT" (singular)
# pressureStateName sub-phase is an actually-stable, trustworthy chamber sample.
PRESSURE_STATE_COL = "data.pressureStateName"
MEASURING_VALUE = "MEASUREMENT"

# predict_batch()'s result-column name -> this script's output column name.
# feature_out_of_range is the narrower, rarer signal (a single feature truly
# beyond anything trained on); in_envelope also fires on a merely-novel
# COMBINATION of otherwise in-range features, which is common on field data
# and less serious - see production_pipeline.check_envelope_batch.
PRED_COLUMN_MAP = {
    "co2_ppm": "pred_co2_ppm",
    "ch4_ppb": "pred_ch4_ppb",
    "ch4_reliable": "ch4_reliable",
    "ch4_warning": "ch4_warning",
    "in_envelope": "in_envelope",
    "feature_out_of_range": "feature_out_of_range",
    "envelope_violations": "envelope_violations",
}
PRED_COLUMNS = list(PRED_COLUMN_MAP.values())


def extract_features(raw):
    missing = [c for c in list(RAW_TO_FEATURE) + NTC_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"Input CSV is missing expected columns: {missing}")

    feat = pd.DataFrame(index=raw.index)
    for raw_col, feature_col in RAW_TO_FEATURE.items():
        feat[feature_col] = raw[raw_col]
    feat["temp_c"] = raw[NTC_COLS].mean(axis=1)
    return feat


def flag_invalid_pressure_and_temp(feat):
    """Deliberately NOT the same rule as training. During lab collection we
    could look at the rig and confirm a pressure dropout or a temp sentinel
    really was a sensor/software glitch, so interpolating between two valid
    neighbors was a justified reconstruction of a known-real signal (that
    interpolation step stays as-is in harmonize_pool.py for the TRAINING
    pool - this comment is about field data only).

    In the field, the setup runs unsupervised from a ground station - there
    is no way to confirm a suspicious pressure_bar<1.0 or temp_c==0.0
    reading is actually an artifact rather than a genuine (if unusual)
    physical event. Interpolating would silently substitute a guessed value
    we have no way to justify; a 0.0 temp reading in particular might be a
    real freezing point, not just the known startup sentinel. So this just
    nulls the suspicious value and leaves it null - the row then has no
    usable reading and gets no prediction (same treatment as a missing
    column or a zero-signal row), rather than fabricating one.

    temp_c's threshold is the exact sentinel value (0.0), not training's
    "<5" shortcut - Norunda is outdoors, and a winter deployment could see
    genuine sub-5, even sub-zero, temperatures that "<5" would wrongly net."""
    bad_pressure = feat["pressure_bar"] < 1.0
    bad_temp = feat["temp_c"] == 0.0
    feat = feat.copy()
    feat.loc[bad_pressure, "pressure_bar"] = np.nan
    feat.loc[bad_temp, "temp_c"] = np.nan
    return feat, int(bad_pressure.sum()), int(bad_temp.sum())


def compute_validity(raw, feat, verbose=True):
    """The same three gates used for per-row predictions, factored out so
    the per-period aggregation (below) uses exactly the same notion of
    "usable row" rather than a second, possibly-drifting copy of the logic."""
    no_signal = feat["lpl_signal"].notna() & (
        (feat["lpl_signal"] <= 0) | (feat["mpl_signal"] <= 0) | (feat["spl_signal"] <= 0)
    )
    missing = feat.isna().any(axis=1)
    not_measuring = raw[PRESSURE_STATE_COL] != MEASURING_VALUE
    has_data = ~missing & ~no_signal & ~not_measuring

    if verbose:
        if missing.sum():
            print(f"{missing.sum()} of {len(raw)} rows have no sensor reading at all "
                  f"(housekeeping/event rows, or an unfillable pressure/temp gap) - passed through with blank predictions")
        if no_signal.sum():
            print(f"{no_signal.sum()} of {len(raw)} rows have a zero/inactive signal reading "
                  f"(sensor present in the log but not producing real data) - passed through with blank predictions")
        n_not_measuring_but_otherwise_ok = int((not_measuring & ~missing & ~no_signal).sum())
        if n_not_measuring_but_otherwise_ok:
            print(f"{n_not_measuring_but_otherwise_ok} of {len(raw)} rows have a seemingly-valid "
                  f"signal but the chamber was not in the '{MEASURING_VALUE}' pressure state "
                  f"(e.g. still compressing/exchanging air - not a stable sample) - passed through with blank predictions")
    return has_data


def predict_rows(raw, feat, has_data, bundle):
    preds = pd.DataFrame(index=raw.index, columns=PRED_COLUMNS)
    if has_data.any():
        result = predict_batch(feat.loc[has_data], bundle)
        for key, col in PRED_COLUMN_MAP.items():
            preds.loc[has_data, col] = result[key].values

    n_out = int((has_data & (preds["in_envelope"] == False)).sum())  # noqa: E712
    if n_out:
        print(f"{n_out} of {int(has_data.sum())} predicted rows are flagged out_of_envelope")
    n_feat_out = int((has_data & (preds["feature_out_of_range"] == True)).sum())  # noqa: E712
    if n_feat_out:
        print(f"{n_feat_out} of {int(has_data.sum())} predicted rows have at least one feature "
              f"genuinely outside anything seen in training (feature_out_of_range)")
    return preds


def find_measurement_periods(raw):
    """Maximal contiguous runs of data.pressureStateName == 'MEASUREMENT',
    in file/time order (NOT data.mode=="MEASUREMENTS" - see module docstring
    point 3; that field covers the whole pressure-cycling sequence, this one
    is the actual stable-sample sub-phase). Returns a Series aligned to
    raw.index: an integer period id for rows inside a MEASUREMENT run, NaN
    for rows outside one. Boundaries are defined by the raw pressure-state
    value itself, not by our own validity filter - a glitchy row in the
    middle of an otherwise-good period doesn't split it, it's just excluded
    when averaging within it.

    Other event types (command/connection rows - see point 1) get
    time-interleaved into the telemetry stream and carry NO pressure-state
    value at all (NaN), even in the middle of a real, continuous MEASUREMENT
    run - confirmed on real data: a NaN row sitting between two MEASUREMENT
    rows a fraction of a second apart, with every other MEASUREMENT row
    around it part of the same physical cycle. A NaN isn't evidence the
    chamber left MEASUREMENT, it's just a row with no state information -
    so boundaries are decided on the state forward-filled through NaNs
    (an actual different state value still correctly splits the period;
    only a NaN with no information either way does not). The period id
    itself is still only assigned to rows that themselves genuinely report
    MEASUREMENT - the interleaved NaN rows are never part of any period's
    row count, they're just invisible to the grouping rather than a break
    in it."""
    filled_state = raw[PRESSURE_STATE_COL].ffill()
    is_meas_filled = filled_state == MEASURING_VALUE
    block = (is_meas_filled != is_meas_filled.shift()).cumsum()
    is_meas = raw[PRESSURE_STATE_COL] == MEASURING_VALUE
    period_id = block.where(is_meas)
    return period_id


def aggregate_periods(raw, feat, has_data, bundle):
    """One row per MEASUREMENT period: average the FEATURES (not the
    predictions) across that period's valid rows, then predict once on the
    averaged reading - reduces input sensor noise before it reaches the
    model, rather than averaging noisy per-row outputs after the fact."""
    period_id = find_measurement_periods(raw)
    usable = period_id.notna() & has_data

    periods = []
    for pid, idx in feat.loc[usable].groupby(period_id[usable]).groups.items():
        all_idx = period_id[period_id == pid].index
        periods.append({
            "period_start": raw.loc[all_idx, "capturedAt"].iloc[0],
            "period_end": raw.loc[all_idx, "capturedAt"].iloc[-1],
            "n_rows_total": len(all_idx),
            "n_rows_valid": len(idx),
            **feat.loc[idx, FEATURES_FOR_AGG].mean().to_dict(),
        })

    if not periods:
        return pd.DataFrame(columns=["period_start", "period_end", "n_rows_total", "n_rows_valid"]
                             + FEATURES_FOR_AGG + PRED_COLUMNS)

    agg = pd.DataFrame(periods)
    result = predict_batch(agg[FEATURES_FOR_AGG], bundle)
    for key, col in PRED_COLUMN_MAP.items():
        agg[col] = result[key].values
    return agg


def process(input_path, output_path=None, model_path=None, periods_output_path=None):
    if not output_path and not periods_output_path:
        raise ValueError("Provide output_path, periods_output_path, or both - nothing to write otherwise")

    raw = pd.read_csv(input_path, low_memory=False)
    if PRESSURE_STATE_COL not in raw.columns:
        raise ValueError(f"Input CSV is missing expected column: {PRESSURE_STATE_COL}")
    feat = extract_features(raw)
    feat, n_bad_p, n_bad_t = flag_invalid_pressure_and_temp(feat)
    if n_bad_p or n_bad_t:
        print(f"Flagged {n_bad_p} suspicious pressure and {n_bad_t} suspicious temp readings as "
              f"unusable (NOT interpolated - unsupervised field data, can't confirm these are "
              f"artifacts rather than real readings) - those rows get no prediction")

    has_data = compute_validity(raw, feat)
    bundle = load_pipeline(model_path) if model_path else load_pipeline()

    if output_path:
        preds = predict_rows(raw, feat, has_data, bundle)
        out = pd.concat([raw, preds], axis=1)
        out.to_csv(output_path, index=False)
        print(f"Wrote {len(out)} rows to {output_path}")

    if periods_output_path:
        periods = aggregate_periods(raw, feat, has_data, bundle)
        periods.to_csv(periods_output_path, index=False)
        print(f"Wrote {len(periods)} MEASUREMENT periods to {periods_output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("output_csv", nargs="?", default=None,
                         help="Per-row prediction output CSV. Optional if --periods-output is given instead - "
                              "provide either, or both.")
    parser.add_argument("--model", default=None, help="Path to the trained pipeline (default: production_pipeline.MODEL_PATH)")
    parser.add_argument("--periods-output", default=None,
                         help="Write one averaged prediction per contiguous MEASUREMENT period to this CSV. "
                              "Optional if output_csv is given instead - provide either, or both.")
    args = parser.parse_args()
    if not args.output_csv and not args.periods_output:
        parser.error("provide output_csv, --periods-output, or both - nothing to write otherwise")
    process(args.input_csv, args.output_csv, args.model, args.periods_output)


if __name__ == "__main__":
    sys.exit(main())
