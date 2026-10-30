#!/usr/bin/env python3
"""Shift, synchronize, and write filtered logger/sensor CSV data."""

### Run with ###
# python3 sync_csv_by_timestamp.py \
#   data/csv/k96_file.csv \
#   data/norunda/norunda_file.csv \
#   data/synced/synced_data.csv

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, TextIO, Tuple


SYNCED_FIELDS = [
    "TIMESTAMP",
    "LGR_CH4_ppb",
    "LGR_H2O_mmolmol-1",
    "LGR_CO2_ppm",
    "capturedAt",
    "data.sensorData.K96_LPL_Signal",
    "data.sensorData.K96_MPL_Signal",
    "data.sensorData.K96_SPL_Signal",
    "data.chamberPressureBar",
    "data.humidityRh_k96",
    "data.chamberTempC_K96",
    "data.sensorData.K96_NTC0_Temp",
    "data.sensorData.K96_NTC1_Temp",
]


def sync_csv_files(
    file1_path: Path,
    file2_path: Path,
    output_path: Path,
    file1_timestamp_column: str = "TIMESTAMP",
    file2_timestamp_column: str = "capturedAt",
) -> int:
    """Write rows whose timestamps match after normalization and return the count."""
    logger_path, sensor_path = _identify_inputs(file1_path, file2_path)
    logger_rows = _read_logger_rows(logger_path, file1_timestamp_column)
    sensor_rows = _read_sensor_rows(sensor_path, file2_timestamp_column)

    logger_by_timestamp = {
        row["_timestamp"]: row for row in logger_rows
    }
    matched_rows: List[Dict[str, str]] = []
    for sensor_row in sensor_rows:
        match_timestamp = sensor_row["_timestamp"]
        logger_row = logger_by_timestamp.get(match_timestamp)
        if logger_row is not None:
            matched_rows.append({**logger_row, **sensor_row})

    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=SYNCED_FIELDS,
            delimiter=";",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(matched_rows)

    return len(matched_rows)


def _identify_inputs(file1_path: Path, file2_path: Path) -> Tuple[Path, Path]:
    """Return logger and sensor paths, accepting either command-line order."""
    paths = (file1_path, file2_path)
    headers = [_read_header(path) for path in paths]
    for index, header in enumerate(headers):
        if "TIMESTAMP" in header:
            logger_path = paths[index]
            sensor_path = paths[1 - index]
            if "capturedAt" not in headers[1 - index]:
                raise ValueError("The sensor CSV must contain a 'capturedAt' column")
            return logger_path, sensor_path
    raise ValueError("Could not identify a logger CSV containing a 'TIMESTAMP' column")


def _read_header(path: Path) -> List[str]:
    with path.open("r", newline="", encoding="utf-8", errors="replace") as csv_file:
        first_line = csv_file.readline()
        if first_line.startswith('"TOA5"'):
            first_line = csv_file.readline()
        if first_line.strip():
            delimiter = "," if first_line.count(",") >= first_line.count(";") else ";"
            return next(csv.reader([first_line], delimiter=delimiter))
    return []


def _read_sensor_rows(path: Path, timestamp_column: str) -> List[Dict[str, str]]:
    delimiter = _delimiter(path)
    with path.open("r", newline="", encoding="utf-8", errors="replace") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        fields = reader.fieldnames or []
        if timestamp_column not in fields:
            raise ValueError(f"Column {timestamp_column!r} was not found in {path}")
        rows = []
        for row in reader:
            timestamp = datetime.fromisoformat(row[timestamp_column])
            row["_timestamp"] = timestamp.replace(tzinfo=None, microsecond=0)
            rows.append(row)
        return rows


def _read_logger_rows(path: Path, timestamp_column: str) -> List[Dict[str, str]]:
    delimiter = _delimiter(path)
    with path.open("r", newline="", encoding="utf-8", errors="replace") as csv_file:
        first_line = csv_file.readline()
        csv_file.seek(0)
        if first_line.startswith('"TOA5"'):
            return _read_toa5_rows(csv_file, delimiter, timestamp_column)

        reader = csv.DictReader(csv_file, delimiter=delimiter)
        fields = reader.fieldnames or []
        if timestamp_column not in fields:
            raise ValueError(f"Column {timestamp_column!r} was not found in {path}")
        return _normalize_logger_rows(reader, timestamp_column)


def _read_toa5_rows(
    csv_file: TextIO, delimiter: str, timestamp_column: str
) -> List[Dict[str, str]]:
    reader = csv.reader(csv_file, delimiter=delimiter)
    next(reader)  # TOA5 metadata
    fields = next(reader)
    if timestamp_column not in fields:
        raise ValueError(f"Column {timestamp_column!r} was not found in the TOA5 header")
    next(reader, None)  # units
    next(reader, None)  # sample labels
    next(reader, None)  # sample/source labels
    return _normalize_logger_rows(
        (dict(zip(fields, values)) for values in reader), timestamp_column
    )


def _normalize_logger_rows(
    rows: Iterable[Dict[str, str]], timestamp_column: str
) -> List[Dict[str, str]]:
    column_map = {
        "LGR_CH4": "LGR_CH4_ppb",
        "LGR_H2O": "LGR_H2O_mmolmol-1",
        "LGR_CO2": "LGR_CO2_ppm",
    }
    normalized_rows = []
    for row in rows:
        timestamp = datetime.strptime(row[timestamp_column], "%Y-%m-%d %H:%M:%S")
        timestamp += timedelta(hours=1)
        normalized = {
            target: row.get(source, row.get(target, ""))
            for source, target in column_map.items()
        }
        normalized["TIMESTAMP"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        normalized["_timestamp"] = timestamp
        normalized_rows.append(normalized)
    return normalized_rows


def _delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as csv_file:
        sample = csv_file.readline()
    return "," if sample.count(",") >= sample.count(";") else ";"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Shift file 1 timestamps by one hour, match them to file 2 timestamps "
            "at second precision, and write the matching rows."
        )
    )
    parser.add_argument("file1", type=Path, help="CSV containing TIMESTAMP")
    parser.add_argument("file2", type=Path, help="CSV containing capturedAt")
    parser.add_argument("output", type=Path, help="Path for the new synchronized CSV")
    parser.add_argument(
        "--file1-timestamp-column",
        default="TIMESTAMP",
        help="Timestamp column in file 1 (default: TIMESTAMP)",
    )
    parser.add_argument(
        "--file2-timestamp-column",
        default="capturedAt",
        help="Timestamp column in file 2 (default: capturedAt)",
    )
    args = parser.parse_args()

    for input_path in (args.file1, args.file2):
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    row_count = sync_csv_files(
        args.file1,
        args.file2,
        args.output,
        args.file1_timestamp_column,
        args.file2_timestamp_column,
    )
    print(f"Wrote {row_count} synchronized rows to {args.output}")


if __name__ == "__main__":
    main()