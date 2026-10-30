#!/usr/bin/env python3
"""Interpolate short, bounded zero-runs and numeric outliers in CSV columns."""

### Run with ###
# python3 data/interpolate_csv.py data/synced_relevant_rows/2026-09-22_all_relevant.csv \
#   --max-zero-run 1 \
#   --output-dir data/synced_relevant_rows/interpolated

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        header = csv_file.readline()
    return "," if header.count(",") > header.count(";") else ";"


def _interpolate_column(
    rows: List[dict[str, str]],
    column: str,
    max_gap: int,
    outlier_window: int,
    outlier_sigma: float,
) -> Tuple[int, int]:
    numeric_values: List[Optional[float]] = []
    for row in rows:
        raw_value = (row.get(column) or "").strip().replace("−", "-")
        if not raw_value:
            numeric_values.append(None)
            continue
        try:
            numeric_values.append(float(raw_value))
        except ValueError:
            return 0, 0

    corrected = 0
    skipped = 0
    outliers = set()
    for index, value in enumerate(numeric_values):
        if value is None or not math.isfinite(value):
            continue

        left_indices = range(max(0, index - outlier_window), index)
        right_indices = range(index + 1, min(len(numeric_values), index + outlier_window + 1))
        left_values = [
            numeric_values[i]
            for i in left_indices
            if numeric_values[i] is not None and math.isfinite(numeric_values[i])
        ]
        right_values = [
            numeric_values[i]
            for i in right_indices
            if numeric_values[i] is not None and math.isfinite(numeric_values[i])
        ]
        if len(left_values) < 1 or len(right_values) < 1:
            continue

        neighbors = left_values + right_values
        if len(neighbors) < 3:
            continue
        neighbor_median = statistics.median(neighbors)
        median_deviation = statistics.median(
            abs(neighbor - neighbor_median) for neighbor in neighbors
        )
        threshold = max(
            outlier_sigma * 1.4826 * median_deviation,
            5e-4 * max(abs(neighbor_median), 1.0),
        )
        if abs(value - neighbor_median) > threshold:
            outliers.add(index)

    index = 0
    while index < len(numeric_values):
        if numeric_values[index] != 0 and index not in outliers:
            index += 1
            continue

        run_start = index
        while index < len(numeric_values) and (
            numeric_values[index] == 0 or index in outliers
        ):
            index += 1
        run_end = index - 1
        run_length = run_end - run_start + 1

        if run_length > max_gap or run_start == 0 or index == len(numeric_values):
            skipped += 1
            continue

        left_value = numeric_values[run_start - 1]
        right_value = numeric_values[index]
        if (
            left_value is None
            or right_value is None
            or not math.isfinite(left_value)
            or not math.isfinite(right_value)
        ):
            skipped += 1
            continue

        span = run_length + 1
        for offset in range(run_length):
            value = left_value + (right_value - left_value) * (offset + 1) / span
            rows[run_start + offset][column] = format(value, ".15g")
            corrected += 1

    return corrected, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate short zero-runs and local outliers in numeric columns "
            "when each run is bounded by valid values."
        )
    )
    parser.add_argument("input", type=Path, help="Input CSV file")
    output_options = parser.add_mutually_exclusive_group()
    output_options.add_argument(
        "--output",
        type=Path,
        help="Output CSV file path",
    )
    output_options.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for output (default: same directory as input)",
    )
    parser.add_argument(
        "--max-gap",
        "--max-zero-run",
        dest="max_gap",
        type=int,
        default=5,
        help="Maximum consecutive zeros or outliers to interpolate (default: 5 rows)",
    )
    parser.add_argument(
        "--outlier-window",
        type=int,
        default=2,
        help="Neighbor rows on each side used to detect outliers (default: 2)",
    )
    parser.add_argument(
        "--outlier-sigma",
        type=float,
        default=4.0,
        help="Outlier sensitivity in robust standard deviations (default: 4)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if args.max_gap < 1:
        parser.error("--max-gap must be at least 1")
    if args.outlier_window < 2:
        parser.error("--outlier-window must be at least 2")
    if args.outlier_sigma <= 0:
        parser.error("--outlier-sigma must be greater than 0")

    delimiter = _delimiter(args.input)
    with args.input.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError(f"No CSV header found in {args.input}")
        rows = list(reader)

    corrected_by_column: Dict[str, int] = {}
    skipped_by_column: Dict[str, int] = {}
    for column in fieldnames:
        corrected, skipped = _interpolate_column(
            rows,
            column,
            args.max_gap,
            args.outlier_window,
            args.outlier_sigma,
        )
        if corrected:
            corrected_by_column[column] = corrected
        if skipped:
            skipped_by_column[column] = skipped

    output_name = f"{args.input.stem}_interpolated{args.input.suffix}"
    output_path = args.output or (args.output_dir or args.input.parent) / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)

    for column, count in corrected_by_column.items():
        print(f"Interpolated {count} value(s) in {column}")
    for column, count in skipped_by_column.items():
        print(f"Left {count} zero/outlier run(s) unchanged in {column}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()