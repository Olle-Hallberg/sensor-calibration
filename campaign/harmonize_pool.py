#!/usr/bin/env python3
"""Harmonize and pool cont_dev_raw (dataset 1) and thesis_raw (dataset 2)
into one schema for K96 calibration modeling.

Output: pooled_data.csv with columns
    source, session, gas_source,
    lpl_signal, mpl_signal, spl_signal,
    temp_c, ntc0_temp_c, ntc1_temp_c,
    pressure_bar, humidity_pct,
    ch4_ppb, co2_ppm,
    lgr_ch4_flag, pressure_interpolated

Data-quality handling:
  - Invalid pressure_bar (<1.0 bar) and startup temp_c (<5 C) are set to NaN,
    then interpolated per-session with limit_area='inside' - i.e. only when
    there are valid readings on BOTH sides within that session (confirmed by
    inspection: in cont_dev_raw/2026-09-22_17-25-05_P3.2_T25_ppm0-2.2.csv the
    ~-8 bar rows are isolated single-row dropouts recurring every ~5 rows,
    with every other channel (signals, NTCs) continuous through them - so the
    chamber pressure itself did not actually move, only that one channel
    glitched). Rows that can't be interpolated this way (e.g. the very first
    row of a file, before the NTCs/pressure sensor have initialized) are
    dropped - there is no valid neighbor to interpolate from there.
  - cont_dev_raw: `data.rawPressures.k96Hpa` is dropped - it is aliased to
    NTC0_Temp in the raw export (a logging bug), not a real pressure
    reading. The external `chamberPressureBar` is used instead.
  - thesis_raw: `k96_pressure_hpa` is dropped for the same reason (it is
    pinned near-constant, not a live reading). The external
    `mpx5500_pressure_hpa` (+1 atm, since it's a relative-pressure sensor)
    is used instead.
  - thesis_raw/{CORE-1,CORE-2,atm_luft,teknisk_luft,chem_N}: these sessions
    used a certified calibration gas with no reference instrument at all -
    ch4_ppb/co2_ppm are the cylinder's labeled molar fraction, constant for
    the whole session (0/0 for chem_N, a pure-N2 zero-gas cylinder).
  - thesis_raw/test: the only thesis_raw session with a live reference
    (DECOUPLING_LGRdata_2026-05-11.csv). That LGR logger runs on winter time
    (UTC+1) year-round; the K96 sensor_log timestamps are real local time,
    which in May is Swedish summer time (UTC+2). So the LGR timestamp needs
    +1h added before matching, exactly like the cont_dev_raw fix - this was
    missing in the original preprocessing.py, which produced a mislabeled
    test_data.csv.
"""
import glob
import os

import pandas as pd

ATM_HPA = 1013.25

CONT_DEV_DIR = "cont_dev_raw"
THESIS_DIR = "thesis_raw"

# Certified cylinder concentrations (ppm), confirmed from the thesis repo's
# preprocessed/<cylinder>/data_<cylinder>.csv files, except chem_N, which is
# a pure-N2 zero-gas cylinder (confirmed: no reference instrument is used for
# any of these sessions - the value is just the cylinder's label).
THESIS_CYLINDER_PPM = {
    "CORE-1": (1.79608, 379.62),
    "CORE-2": (2.19489, 465.52),
    "atm_luft": (2.2981, 450.1),
    "teknisk_luft": (1.9672, 435.8),
    "chem_N": (0.0, 0.0),
}

THESIS_RAW_COLUMNS = [
    "elapsed_time", "mpx5500_pressure_hpa", "rh_temp_c", "humidity_pct",
    "k96_pressure_hpa", "k96_pressure_filtered_hpa",
    "ntc0_temp_c", "ntc0_temp_filtered_c", "ntc1_temp_c", "ntc1_temp_filtered_c",
    "lpl_signal", "mpl_signal", "spl_signal",
    "lpl_signal_filtered", "mpl_signal_filtered", "spl_signal_filtered",
    "mpl_conc", "lpl_conc", "spl_conc",
]

POOLED_COLUMNS = [
    "source", "session", "gas_source",
    "lpl_signal", "mpl_signal", "spl_signal",
    "temp_c", "ntc0_temp_c", "ntc1_temp_c",
    "pressure_bar", "humidity_pct",
    "ch4_ppb", "co2_ppm", "lgr_ch4_flag", "pressure_interpolated",
]


def clean_pressure_and_temp(df):
    """Null out implausible pressure/temp, interpolate interior gaps, flag."""
    bad_pressure = df["pressure_bar"] < 1.0
    bad_temp = df["temp_c"] < 5.0
    df.loc[bad_pressure, "pressure_bar"] = pd.NA
    df.loc[bad_temp, "temp_c"] = pd.NA

    df["pressure_bar"] = df["pressure_bar"].interpolate(method="linear", limit_area="inside")
    df["temp_c"] = df["temp_c"].interpolate(method="linear", limit_area="inside")

    df["pressure_interpolated"] = bad_pressure & df["pressure_bar"].notna()
    still_bad = df["pressure_bar"].isna() | df["temp_c"].isna()
    return df, int(still_bad.sum())


def load_cont_dev():
    """K96 sweep files + LGR reference, merged on second-resolution timestamp."""
    lgr_path = os.path.join(CONT_DEV_DIR, "DECOUPLING_LGRdata_2026-09-22.csv")
    lgr = pd.read_csv(lgr_path, skiprows=[0, 2, 3], encoding="latin-1")
    lgr["TIMESTAMP"] = pd.to_datetime(lgr["TIMESTAMP"]) + pd.Timedelta(hours=1)
    lgr = lgr[["TIMESTAMP", "LGR_CH4", "LGR_CO2"]].rename(
        columns={"LGR_CH4": "ch4_ppb", "LGR_CO2": "co2_ppm"}
    )

    k96_cols = [
        "capturedAt",
        "data.sensorData.K96_LPL_Signal",
        "data.sensorData.K96_MPL_Signal",
        "data.sensorData.K96_SPL_Signal",
        "data.chamberPressureBar",
        "data.humidityRh_k96",
        "data.sensorData.K96_NTC0_Temp",
        "data.sensorData.K96_NTC1_Temp",
    ]
    rename = {
        "data.sensorData.K96_LPL_Signal": "lpl_signal",
        "data.sensorData.K96_MPL_Signal": "mpl_signal",
        "data.sensorData.K96_SPL_Signal": "spl_signal",
        "data.chamberPressureBar": "pressure_bar",
        "data.humidityRh_k96": "humidity_pct",
        "data.sensorData.K96_NTC0_Temp": "ntc0_temp_c",
        "data.sensorData.K96_NTC1_Temp": "ntc1_temp_c",
    }

    frames = []
    dropped_total = 0
    for path in sorted(glob.glob(os.path.join(CONT_DEV_DIR, "2026-09-22_*_P*ppm*.csv"))):
        session = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, usecols=k96_cols)
        df = df.dropna(subset=["data.sensorData.K96_LPL_Signal"]).reset_index(drop=True)
        df = df.rename(columns=rename)

        ts = pd.to_datetime(df["capturedAt"])
        df["TIMESTAMP"] = ts.dt.tz_localize(None).dt.floor("s")
        df = df.merge(lgr, on="TIMESTAMP", how="left")

        df["temp_c"] = df[["ntc0_temp_c", "ntc1_temp_c"]].mean(axis=1)
        df, n_bad = clean_pressure_and_temp(df)
        dropped_total += n_bad

        df["source"] = "cont_dev"
        df["session"] = session
        df["gas_source"] = "LGR_live"
        df["lgr_ch4_flag"] = df["ch4_ppb"].eq(-10.0)
        frames.append(df[POOLED_COLUMNS])

    print(f"cont_dev: dropped {dropped_total} unfillable rows (edge of file, no neighbor to interpolate from)")
    return pd.concat(frames, ignore_index=True)


def load_thesis_cylinders():
    frames = []
    dropped_total = 0
    for cylinder, (ch4_ppm, co2_ppm) in THESIS_CYLINDER_PPM.items():
        folder = os.path.join(THESIS_DIR, cylinder)
        for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
            fname = os.path.basename(path)
            kind = "humid" if fname.startswith("humid_") else "dry"
            stamp = os.path.splitext(fname)[0].rsplit("_", 1)[-1]
            session = f"thesis_{cylinder}_{kind}_{stamp}"

            if cylinder == "chem_N":
                df = pd.read_csv(path, header=None, names=THESIS_RAW_COLUMNS)
            else:
                df = pd.read_csv(path)

            df = df.dropna(subset=["lpl_signal"]).reset_index(drop=True)
            df["pressure_bar"] = df["mpx5500_pressure_hpa"] / 1000.0 + ATM_HPA / 1000.0
            df["temp_c"] = df[["ntc0_temp_c", "ntc1_temp_c"]].mean(axis=1)
            df, n_bad = clean_pressure_and_temp(df)
            dropped_total += n_bad

            df["ch4_ppb"] = ch4_ppm * 1000.0
            df["co2_ppm"] = co2_ppm
            df["source"] = "thesis"
            df["session"] = session
            df["gas_source"] = "cylinder_constant"
            df["lgr_ch4_flag"] = False
            frames.append(df[POOLED_COLUMNS])

    print(f"thesis cylinders: dropped {dropped_total} unfillable rows (edge of file, no neighbor to interpolate from)")
    return pd.concat(frames, ignore_index=True)


def load_thesis_test():
    """thesis_raw/test: the one thesis_raw session with a live LGR reference.
    Needs the same +1h winter->summer correction as cont_dev_raw."""
    lgr_path = os.path.join(THESIS_DIR, "test", "DECOUPLING_LGRdata_2026-05-11.csv")
    lgr = pd.read_csv(lgr_path, skiprows=[1])
    lgr["ch4_ppb"] = pd.to_numeric(lgr["LGR_CH4"], errors="coerce")
    lgr["co2_ppm"] = pd.to_numeric(lgr["LGR_CO2"], errors="coerce")
    lgr["TIMESTAMP"] = pd.to_datetime(lgr["TIMESTAMP"]) + pd.Timedelta(hours=1)
    lgr = lgr[["TIMESTAMP", "ch4_ppb", "co2_ppm"]]

    frames = []
    dropped_total = 0
    n_unmatched_total = 0
    for path in sorted(glob.glob(os.path.join(THESIS_DIR, "test", "sensor_log_*.csv"))):
        fname = os.path.basename(path)
        # filename pattern: sensor_log_YYYYMMDD_HHMMSS.csv
        date_part, time_part = fname[len("sensor_log_"):-4].split("_")
        start_time = pd.Timestamp(
            f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} "
            f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
        )

        df = pd.read_csv(path)
        df = df.dropna(subset=["lpl_signal"]).reset_index(drop=True)
        df["elapsed_time"] = pd.to_timedelta(df["elapsed_time"])
        df["TIMESTAMP"] = (start_time + df["elapsed_time"]).dt.floor("s")

        df = df.merge(lgr, on="TIMESTAMP", how="left")
        n_unmatched_total += int(df["ch4_ppb"].isna().sum())

        df["pressure_bar"] = df["mpx5500_pressure_hpa"] / 1000.0 + ATM_HPA / 1000.0
        df["temp_c"] = df[["ntc0_temp_c", "ntc1_temp_c"]].mean(axis=1)
        df, n_bad = clean_pressure_and_temp(df)
        dropped_total += n_bad

        df["source"] = "thesis"
        df["session"] = f"thesis_test_{date_part}_{time_part}"
        df["gas_source"] = "LGR_live"
        df["lgr_ch4_flag"] = df["ch4_ppb"].eq(-10.0)
        frames.append(df[POOLED_COLUMNS])

    print(f"thesis test: dropped {dropped_total} unfillable rows; {n_unmatched_total} rows had no LGR timestamp match")
    return pd.concat(frames, ignore_index=True)


def main():
    cont_dev = load_cont_dev()
    thesis_cyl = load_thesis_cylinders()
    thesis_test = load_thesis_test()

    pooled = pd.concat([cont_dev, thesis_cyl, thesis_test], ignore_index=True)
    pooled = pooled.dropna(subset=["pressure_bar", "temp_c"]).reset_index(drop=True)

    # Zero-count optical signal is not physically possible for an active
    # reading - these are invalid/dropout rows (mostly at session starts).
    zero_signal = (pooled["lpl_signal"] <= 0) | (pooled["mpl_signal"] <= 0) | (pooled["spl_signal"] <= 0)
    print(f"Dropping {int(zero_signal.sum())} rows with zero/negative signal (invalid reading):")
    print(pooled[zero_signal].groupby(["source", "session"]).size())
    pooled = pooled[~zero_signal].reset_index(drop=True)

    # Exact-duplicate consecutive samples (same session, identical signal
    # triple) - concentrated in a couple of thesis "humid" sessions (up to
    # ~11% of rows there), which would otherwise double-weight those points.
    dup = pooled.duplicated(subset=["source", "session", "lpl_signal", "mpl_signal", "spl_signal"])
    print(f"Dropping {int(dup.sum())} exact-duplicate rows:")
    print(pooled[dup].groupby(["source", "session"]).size())
    pooled = pooled[~dup].reset_index(drop=True)
    print()

    pooled.to_csv("pooled_data.csv", index=False)

    print()
    print(f"Pooled rows: {len(pooled)}")
    print()
    print("Rows per session:")
    print(pooled.groupby(["source", "session"]).size())
    print()
    print("Rows with missing ch4_ppb/co2_ppm (no LGR timestamp match):")
    print(pooled[pooled["ch4_ppb"].isna()].groupby("session").size())
    print()
    print("Rows flagged lgr_ch4_flag (LGR -10 floor reading):")
    print(int(pooled["lgr_ch4_flag"].sum()))
    print()
    print("Rows flagged pressure_interpolated (isolated mid-session pressure dropout, fixed by interpolation):")
    print(int(pooled["pressure_interpolated"].sum()))
    print(pooled[pooled["pressure_interpolated"]].groupby("session").size())


if __name__ == "__main__":
    main()
