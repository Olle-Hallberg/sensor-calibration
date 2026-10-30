#!/usr/bin/env python3
"""Convert an SD-card flight recording (metadata_noNN.log + sensor_data_noNN.csv)
into the live-logged raw CSV schema (the capturedAt / data.* schema used by
e.g. 2026-10-14_08-27-44_total.csv), preserving every column from both SD
files that has a verified counterpart.

The SD card stores the same physical session as the live-logged CSV, but
split across two files and in a more compact form:
  - metadata_noNN.log: HH,MM,SS + controller/pressure-cycle state, as
    numeric codes rather than the live format's strings.
  - sensor_data_noNN.csv: HH,MM,SS + raw sensor channels, with generic
    labels (TP1, PP2, ...) instead of the live format's named fields.
  - HH:MM:SS is ELAPSED time since the SD recorder started (resets to
    00:00:00), NOT wall-clock time-of-day - confirmed because it doesn't
    overlap the live file's known capturedAt range at all.

Four things had to be reverse-engineered against a live-logged file we
already trust (2026-10-14_08-27-44_total.csv) before this conversion was
possible - all confirmed empirically below, not assumed:

1. Row alignment. metadata has 59,371 data rows, sensor_data has 59,375 -
   checked row-by-row: all 59,371 metadata rows match sensor_data's first
   59,371 rows exactly (by HH:MM:SS), so sensor_data just has 4 unmatched
   trailing rows (presumably a cutoff at the very end of the recording).
   A plain positional join on the first 59,371 rows of each is correct here
   - verified, not assumed.

2. The elapsed-to-wall-clock anchor. Found by matching the CADENCE of
   MEASUREMENT-period start times (a repeating pressurize/measure cycle has
   a distinctive gap pattern, independent of absolute values) against the
   already-known periods in 2026-10-14_08-27-44_total.csv. The first ~16
   period-to-period gaps match to within ~1s (the SD log's whole-second
   rounding) at a CONSTANT offset, checked at multiple widely-separated
   points (not just the start) to rule out clock drift:
     total.csv period[0]  (sd_elapsed=6659)  -> anchor 12:04:44.247+02:00
     total.csv period[5]  (sd_elapsed=7344)  -> anchor 12:04:43.879+02:00
     total.csv period[10] (sd_elapsed=8034)  -> anchor 12:04:44.090+02:00
     total.csv period[15] (sd_elapsed=8735)  -> anchor 12:04:43.490+02:00
   All four agree within <1s, which is exactly the whole-second rounding
   noise expected - not a coincidence. ANCHOR below is the average of these.

   Side-finding from this check: total.csv has two abnormally long gaps
   (870s, 644s) exactly where the SD card shows several additional,
   regularly-spaced periods instead - the live ground-station link most
   likely dropped out for a while and missed cycles the onboard SD card
   kept recording. Worth knowing if comparing the two files further.

   If this is run on a DIFFERENT no-NN recording, re-derive the anchor the
   same way (or get a real one from the operator) rather than reusing
   ANCHOR blindly - it is specific to this one recording's power-on time.

3. Decoded enum columns, checked by crosstab against the live file's
   string/bool columns (not guessed from the names):
     metadata.pressure_state -> data.pressureStateName (string) and
       data.statusData.pressure_state / data.pressureState (numeric, both
       confirmed identical to the raw code):
         0=STANDBY, 1=PREPRESSURISATION, 2=AIR_EXCHANGE, 5=COMPRESSION, 6=MEASUREMENT
     metadata.mode -> data.mode (string) and data.statusData.operating_mode
       (numeric, confirmed identical to the raw code):
         2=STANDBY, 3=MEASUREMENTS

4. "Duplicate name" pairs. The live schema often carries the same value
   under two different column names (e.g. data.chamberPressureBar IS
   data.sensorData.Pp2 - confirmed byte-for-byte identical, not just
   correlated; same for data.humidityRh_k96/data.sensorData.K96_RH, and
   data.pressureState/data.statusData.pressure_state). A second family of
   pairs (data.commandReceived/data.statusData.command_received,
   data.connectionLost/.connection_lost, data.pressurePump1Pwm/
   .pressure_pump1_pwm, data.pressureRelayMask/.pressure_relay_mask,
   data.storageFreePct/.storage_free_pct, and by the same pattern
   data.statusOk, data.pressureSystemOn, data.k96On, data.thermalOnline,
   data.thermalState, data.pressureError, data.pressureManualOverride,
   data.pressureValveOpen, data.onboardLogging) all show the EXACT SAME
   match rate against their statusData.* counterpart: 0.9935351215174867.
   That's not independent noise on each field, it's one systemic cause (a
   few rows only got one of the two duplicate-carrying telemetry packets) -
   checked on 4 of these pairs directly, the rest inferred from that
   single consistent signature plus the naming convention. Since metadata
   only has ONE copy of each such value, this script writes it to BOTH
   target names.

NOT reconstructed, written through under their own plain names instead of
guessing a data.* target:
  - thermal_error1..thermal_error8: the live schema has these as a single
    list-valued column (data.thermalErrors, e.g. "[20,255,255,...]"), not
    8 separate columns - reformatting into that exact list-string style
    wasn't verified, so these stay as their own columns (sd_thermal_error1
    etc.) rather than risk a wrong reconstruction.
  - captured_errors_low_hex / captured_errors_high_hex: the live schema
    has ONE combined hex field (data.statusData.captured_errors_bytes_hex);
    the exact concatenation order needed to rebuild it from the two halves
    wasn't verified against an independent source, so these pass through
    as sd_captured_errors_low_hex / sd_captured_errors_high_hex.
  - metadata.controller_state has a numeric target (data.statusData.
    controller_state, used directly) but data.controller is a STRING
    (e.g. "MAIN_MCU_READY") with only one example value seen - not enough
    to build a reliable decode table, so data.controller is left out
    rather than guessed.

K96_LPL_flt/K96_SPL_flt/K96_MPL_flt map to the live schema's "_Signal_filtered"
fields by naming convention (K96_LPL -> K96_LPL_Signal, so K96_LPL_flt ->
K96_LPL_Signal_filtered) - not individually spot-checked against the live
file the way the fields in point 4 were, since "_filtered" is an
unambiguous, consistently-used suffix elsewhere in the live schema.
"""
import argparse
import sys

import pandas as pd

PRESSURE_STATE_MAP = {0: "STANDBY", 1: "PREPRESSURISATION", 2: "AIR_EXCHANGE",
                       5: "COMPRESSION", 6: "MEASUREMENT"}
MODE_MAP = {2: "STANDBY", 3: "MEASUREMENTS"}

# sensor_data_noNN.csv column -> one or more data.* targets.
SENSOR_TO_TARGETS = {
    "TP1": ["data.sensorData.Tp1"],
    "TP2": ["data.sensorData.Tp2"],
    "TP3": ["data.sensorData.Tp3"],
    "TP4": ["data.sensorData.Tp4"],
    "TP5": ["data.sensorData.Tp5"],
    "TP6": ["data.sensorData.Tp6"],
    "PP1": ["data.sensorData.Pp1"],
    "PP2": ["data.sensorData.Pp2", "data.chamberPressureBar"],  # verified identical
    "PP3": ["data.sensorData.Pp3"],
    "PA1": ["data.sensorData.Pa1"],
    "TA1": ["data.sensorData.Ta1"],
    "TA2": ["data.sensorData.Ta2"],
    "TA3": ["data.sensorData.Ta3"],
    "HA1": ["data.sensorData.Ha1"],
    "TT1": ["data.sensorData.Tt1"],
    "TT2": ["data.sensorData.Tt2"],
    "TT3": ["data.sensorData.Tt3"],
    "K96_LPL": ["data.sensorData.K96_LPL_Signal"],
    "K96_LPL_flt": ["data.sensorData.K96_LPL_Signal_filtered"],
    "K96_MPL": ["data.sensorData.K96_MPL_Signal"],
    "K96_MPL_flt": ["data.sensorData.K96_MPL_Signal_filtered"],
    "K96_SPL": ["data.sensorData.K96_SPL_Signal"],
    "K96_SPL_flt": ["data.sensorData.K96_SPL_Signal_filtered"],
    "K96_ADuCdie_Temp": ["data.sensorData.K96_ADuCdie_Temp"],
    "K96_ADuCdie_Temp_filtered": ["data.sensorData.K96_ADuCdie_Temp_filtered"],
    "K96_NTC0_Temp": ["data.sensorData.K96_NTC0_Temp"],
    "K96_NTC0_Temp_filtered": ["data.sensorData.K96_NTC0_Temp_filtered"],
    "K96_NTC1_Temp": ["data.sensorData.K96_NTC1_Temp"],
    "K96_NTC1_Temp_filtered": ["data.sensorData.K96_NTC1_Temp_filtered"],
    "K96_RH": ["data.sensorData.K96_RH", "data.humidityRh_k96"],  # verified identical
    "K96_RH_Temp": ["data.sensorData.K96_RH_Temp"],
    "K96_MPL_uflt_IR_Signal": ["data.sensorData.K96_MPL_uflt_IR_Signal"],
    "K96_MPL_flt_IR_Signal": ["data.sensorData.K96_MPL_flt_IR_Signal"],
    "K96_MPL_uflt_Conc": ["data.sensorData.K96_MPL_uflt_Conc"],
    "K96_MPL_flt_Conc": ["data.sensorData.K96_MPL_flt_Conc"],
    "K96_MPL_uflt_Error": ["data.sensorData.K96_MPL_uflt_Error"],
    "K96_LPL_uflt_IR_Signal": ["data.sensorData.K96_LPL_uflt_IR_Signal"],
    "K96_LPL_flt_IR_Signal": ["data.sensorData.K96_LPL_flt_IR_Signal"],
    "K96_LPL_uflt_Conc": ["data.sensorData.K96_LPL_uflt_Conc"],
    "K96_LPL_uflt_Error": ["data.sensorData.K96_LPL_uflt_Error"],
    "K96_LPL_flt_Error": ["data.sensorData.K96_LPL_flt_Error"],
    "K96_SPL_uflt_IR_Signal": ["data.sensorData.K96_SPL_uflt_IR_Signal"],
    "K96_SPL_flt_IR_Signal": ["data.sensorData.K96_SPL_flt_IR_Signal"],
    "K96_SPL_uflt_Conc": ["data.sensorData.K96_SPL_uflt_Conc"],
    "K96_SPL_uflt_Error": ["data.sensorData.K96_SPL_uflt_Error"],
    "K96_SPL_flt_Error": ["data.sensorData.K96_SPL_flt_Error"],
    "K96_error": ["data.sensorData.K96_error", "data.k96Error"],  # only source for k96Error either way
}

# metadata_noNN.log column -> one or more data.* targets, each optionally
# with a (map_to_bool) flag for targets confirmed/assumed boolean-typed.
META_TO_TARGETS = {
    "command_received": [("data.statusData.command_received", False), ("data.commandReceived", True)],
    "connection_lost": [("data.statusData.connection_lost", False), ("data.connectionLost", True)],
    "status_ok": [("data.statusData.status_ok", False), ("data.statusOk", True)],
    "pressure_system_on": [("data.statusData.pressure_system_on", False), ("data.pressureSystemOn", True)],
    "k96_on": [("data.statusData.k96_on", False), ("data.k96On", True)],
    "thermal_online": [("data.statusData.thermal_online", False), ("data.thermalOnline", True)],
    "thermal_state": [("data.statusData.thermal_state", False), ("data.thermalState", False)],
    "pressure_error": [("data.statusData.pressure_error", False), ("data.pressureError", True)],
    "pressure_relay_mask": [("data.statusData.pressure_relay_mask", False), ("data.pressureRelayMask", False)],
    "pressure_pump1_pwm": [("data.statusData.pressure_pump1_pwm", False), ("data.pressurePump1Pwm", False)],
    "pressure_pump2_pwm": [("data.statusData.pressure_pump2_pwm", False), ("data.pressurePump2Pwm", False)],
    "pressure_compressor_pwm": [("data.statusData.pressure_compressor_pwm", False), ("data.pressureCompressorPwm", False)],
    "pressure_manual_override": [("data.statusData.pressure_manual_override", False), ("data.pressureManualOverride", True)],
    "pressure_valve_open": [("data.statusData.pressure_valve_open", False), ("data.pressureValveOpen", True)],
    "onboard_logging": [("data.statusData.onboard_logging", False), ("data.onboardLogging", True)],
    "storage_free_pct": [("data.statusData.storage_free_pct", False), ("data.storageFreePct", False)],
    "controller_state": [("data.statusData.controller_state", False)],
    "heater1_duty": [("data.heater1ActuationPct", False)],
    "heater2_duty": [("data.heater2ActuationPct", False)],
    "heater3_duty": [("data.heater3ActuationPct", False)],
    "heater4_duty": [("data.heater4ActuationPct", False)],
    "heater5_duty": [("data.heater5ActuationPct", False)],
    "heater6_duty": [("data.heater6ActuationPct", False)],
    "heater7_duty": [("data.heater7ActuationPct", False)],
    "heater8_duty": [("data.heater8ActuationPct", False)],
}
# Not reconstructed into the live schema's exact field - see docstring.
META_PASSTHROUGH = [
    "thermal_error1", "thermal_error2", "thermal_error3", "thermal_error4",
    "thermal_error5", "thermal_error6", "thermal_error7", "thermal_error8",
    "captured_errors_low_hex", "captured_errors_high_hex",
]

# This recording's power-on time, derived empirically - see module
# docstring point 2. Specific to metadata_no43.log / sensor_data_no43.csv.
ANCHOR = pd.Timestamp("2026-10-14T12:04:43.93+02:00")


def convert(metadata_path, sensor_path, output_path, anchor=ANCHOR):
    meta = pd.read_csv(metadata_path, low_memory=False)
    sens = pd.read_csv(sensor_path, low_memory=False)
    sens.columns = [c.strip() for c in sens.columns]

    n = min(len(meta), len(sens))
    if len(meta) != len(sens):
        print(f"Row count mismatch: metadata has {len(meta)}, sensor_data has {len(sens)} - "
              f"using the first {n} rows of each (checked: they align exactly by HH:MM:SS "
              f"over that range).")
    meta = meta.iloc[:n].reset_index(drop=True)
    sens = sens.iloc[:n].reset_index(drop=True)

    mismatched = (meta[["HH", "MM", "SS"]].values != sens[["HH", "MM", "SS"]].values).any()
    if mismatched:
        raise ValueError(
            "metadata and sensor_data HH:MM:SS sequences don't align over the common row "
            "range - the positional join this script relies on is not valid here, don't "
            "trust the output without checking alignment some other way first."
        )

    elapsed_s = meta["HH"] * 3600 + meta["MM"] * 60 + meta["SS"]
    captured_at = anchor + pd.to_timedelta(elapsed_s, unit="s")

    columns = {"capturedAt": captured_at.map(lambda t: t.isoformat())}

    for sensor_col, targets in SENSOR_TO_TARGETS.items():
        for target in targets:
            columns[target] = sens[sensor_col]

    for meta_col, targets in META_TO_TARGETS.items():
        for target, as_bool in targets:
            columns[target] = meta[meta_col].astype(bool) if as_bool else meta[meta_col]

    for meta_col in META_PASSTHROUGH:
        columns[f"sd_{meta_col}"] = meta[meta_col]

    columns["data.pressureState"] = meta["pressure_state"]
    columns["data.statusData.pressure_state"] = meta["pressure_state"]
    columns["data.pressureStateName"] = meta["pressure_state"].map(PRESSURE_STATE_MAP)
    columns["data.statusData.operating_mode"] = meta["mode"]
    columns["data.mode"] = meta["mode"].map(MODE_MAP)

    out = pd.DataFrame(columns, index=meta.index)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} rows, {len(out.columns)} columns to {output_path}")
    print(f"capturedAt range: {out['capturedAt'].iloc[0]} -> {out['capturedAt'].iloc[-1]}")
    unmapped_pressure = meta["pressure_state"][~meta["pressure_state"].isin(PRESSURE_STATE_MAP)]
    if len(unmapped_pressure):
        print(f"WARNING: {len(unmapped_pressure)} rows had a pressure_state code not in "
              f"{PRESSURE_STATE_MAP} - check whether this recording uses the same encoding.")
    print(f"Columns not reconstructed into their live-schema form (see docstring): "
          f"{[f'sd_{c}' for c in META_PASSTHROUGH]}, data.controller")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_log")
    parser.add_argument("sensor_data_csv")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    convert(args.metadata_log, args.sensor_data_csv, args.output_csv)


if __name__ == "__main__":
    sys.exit(main())
