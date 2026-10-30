#!/usr/bin/env python3

### Run with ###
# python3 jsonl_to_csv.py input.jsonl output/custom.csv

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

def flatten_record(value: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(value, dict):
        flattened: Dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_record(child, child_prefix))
        return flattened

    if isinstance(value, list):
        return {prefix: json.dumps(value, ensure_ascii=False)}

    return {prefix: value}


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            normalized.update(flatten_record(value, key))
        else:
            normalized[key] = value
    return normalized


def convert_jsonl_to_csv(input_path: Path, output_path: Path) -> int:
    rows = []
    fieldnames = set()

    with input_path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            row = normalize_record(record)
            rows.append(row)
            fieldnames.update(row.keys())

    ordered_fieldnames = sorted(fieldnames)

    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=ordered_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in ordered_fieldnames})

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a JSONL file into a CSV file")
    parser.add_argument("input", type=Path, help="Path to the input .jsonl file")
    parser.add_argument("output", type=Path, nargs="?", help="Path to the output .csv file")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output is None:
        output_path = input_path.with_suffix(".csv")
    else:
        output_path = args.output.resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = convert_jsonl_to_csv(input_path, output_path)
    print(f"Converted {row_count} rows from {input_path} to {output_path}")


if __name__ == "__main__":
    main()
