#!/usr/bin/env python3
"""Create plots with independent y-axes from synchronized sensor data."""

### Run with ###
# python3 plot_synced_data.py data/synced/synced_sensor_logger_data.csv \
#   --x-column data.sensorData.K96_LPL_Signal \
#   --left-column LGR_CH4_ppb \
#   --right-column data.chamberPressureBar \
#   --output-dir plots \
#   --output-name methane-pressure.png
# or
# python3 plot_synced_data.py data/synced_relevant_rows/interpolated/2026-09-22_all_relevant_interpolated.csv \
#   --x-column index \
#   --left-column LGR_CH4_ppb data.sensorData.K96_LPL_Signal \
#   --right-column data.chamberPressureBar data.chamberTempC_K96 \
#   --output-dir plots/all_relevant_interpolated_vs_index \
#   --output-name ch4_lpl_pressure_temp.png


import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


PlotDefinition = Tuple[str, str, str, str, str]

PLOTS: List[PlotDefinition] = [
    (
        "LGR_CH4_ppb",
        "data.sensorData.K96_LPL_Signal",
        "LGR_CH4_ppb",
        "data.sensorData.K96_LPL_Signal",
        "ch4_vs_lpl_signal.png",
    ),
    (
        "LGR_H2O_mmolmol-1",
        "data.sensorData.K96_MPL_Signal",
        "LGR_H2O_mmolmol-1",
        "data.sensorData.K96_MPL_Signal",
        "h2o_vs_mpl_signal.png",
    ),
    (
        "LGR_CO2_ppm",
        "data.sensorData.K96_SPL_Signal",
        "LGR_CO2_ppm",
        "data.sensorData.K96_SPL_Signal",
        "co2_vs_spl_signal.png",
    ),
]


def _x_parser(value: str) -> Callable[[str], Union[datetime, float, str]]:
    """Choose a parser for datetime, numeric, or categorical x-axis values."""
    try:
        datetime.fromisoformat(value)
        return datetime.fromisoformat
    except ValueError:
        pass

    try:
        float(value.replace("−", "-"))
        return lambda item: float(item.replace("−", "-"))
    except ValueError:
        return lambda item: item


def read_csv(
    path: Path, x_column: str, y_columns: List[str]
) -> Tuple[List[Union[datetime, float, str]], dict[str, List[Optional[float]]]]:
    """Read selected x and numeric y columns from a comma- or semicolon-delimited CSV."""
    use_data_point_index = x_column.casefold() == "index"
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        header = csv_file.readline()
        delimiter = "," if header.count(",") > header.count(";") else ";"
        csv_file.seek(0)
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        required_columns = set(y_columns)
        if not use_data_point_index:
            required_columns.add(x_column)
        missing_columns = required_columns - set(fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required column(s): {missing}")

        x_values: List[Union[datetime, float, str]] = []
        values = {column: [] for column in y_columns}
        parse_x: Optional[Callable[[str], Union[datetime, float, str]]] = None
        for data_point, row in enumerate(reader, start=1):
            if use_data_point_index:
                x_values.append(float(data_point))
            else:
                raw_x = row[x_column].strip()
                if not raw_x:
                    continue
                if parse_x is None:
                    parse_x = _x_parser(raw_x)
                x_values.append(parse_x(raw_x))
            for column in values:
                value = row[column].strip().replace("−", "-")
                try:
                    values[column].append(float(value) if value else None)
                except ValueError as error:
                    raise ValueError(
                        f"Non-numeric value {value!r} in column {column!r}"
                    ) from error

    return x_values, values


def plot_pair(
    x_values: List[Union[datetime, float, str]],
    values: dict[str, List[Optional[float]]],
    left_columns: List[str],
    right_columns: Optional[List[str]],
    x_label: str,
    output_path: Path,
) -> None:
    """Create and save a plot with an independently scaled axis per series."""
    all_columns = left_columns + (right_columns or [])
    legend_columns = min(3, len(all_columns))
    legend_rows = (len(all_columns) + legend_columns - 1) // legend_columns
    figure_width = max(14, 11 + 1.15 * (len(all_columns) - 2))
    figure_height = max(7, 5.5 + 0.35 * legend_rows)
    figure, left_axis = plt.subplots(figsize=(figure_width, figure_height))
    colors = plt.get_cmap("tab20")
    lines = []
    axis_sides = []

    def add_series(column: str, axis: plt.Axes, side: str, color_index: int) -> None:
        color = colors(color_index / max(1, len(all_columns) - 1))
        lines.extend(
            axis.plot(
                x_values,
                values[column],
                color=color,
                linewidth=1.2,
                label=column,
            )
        )
        axis.set_ylabel(column, color=color, labelpad=8)
        axis.tick_params(axis="y", colors=color)
        axis.yaxis.set_offset_position(side)
        axis.spines[side].set_color(color)
        axis_sides.append((axis, side))

    left_axis.spines["right"].set_visible(False)
    for index, column in enumerate(left_columns):
        axis = left_axis
        if index:
            axis = left_axis.twinx()
            axis.patch.set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_position(("outward", 55 * index))
            axis.yaxis.set_label_position("left")
            axis.yaxis.tick_left()
        axis.spines["top"].set_visible(False)
        axis.spines["left"].set_visible(True)
        add_series(column, axis, "left", index)

    for index, column in enumerate(right_columns or []):
        axis = left_axis.twinx()
        axis.patch.set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["right"].set_position(("outward", 55 * index))
        axis.yaxis.set_label_position("right")
        axis.yaxis.tick_right()
        axis.spines["top"].set_visible(False)
        add_series(column, axis, "right", len(left_columns) + index)

    left_axis.set_xlabel(x_label)
    left_axis.grid(True, alpha=0.3)
    if x_values and isinstance(x_values[0], datetime):
        left_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
        figure.autofmt_xdate()

    legend_bottom = min(0.5, 0.04 + 0.06 * legend_rows)
    figure.legend(
        lines,
        [line.get_label() for line in lines],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=legend_columns,
        frameon=False,
    )
    title = (
        " and ".join(all_columns)
        if len(all_columns) <= 3
        else f"{len(all_columns)} parameters versus {x_label}"
    )
    figure.suptitle(title)
    left_margin = min(0.32, 0.12 + 0.04 * (len(left_columns) - 1))
    right_margin = max(0.68, 0.88 - 0.04 * (len(right_columns or []) - 1))
    figure.tight_layout(rect=(left_margin, legend_bottom, right_margin, 0.95))
    figure.canvas.draw()
    for axis, side in axis_sides:
        spine_bounds = axis.spines[side].get_window_extent()
        spine_x = spine_bounds.x0 if side == "left" else spine_bounds.x1
        offset_x = axis.transAxes.inverted().transform((spine_x, axis.bbox.y0))[0]
        axis.yaxis.get_offset_text().set_x(offset_x)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot selected columns from a semicolon-delimited CSV file."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("synced_sensor_logger_data.csv"),
        help="Synchronized CSV file (default: synced_sensor_logger_data.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory for PNG plots (default: plots)",
    )
    parser.add_argument(
        "--output-names",
        nargs=len(PLOTS),
        metavar=("CH4_FILE", "H2O_FILE", "CO2_FILE"),
        default=[plot[-1] for plot in PLOTS],
        help="Output filenames for the CH4, H2O, and CO2 plots",
    )
    parser.add_argument(
        "--x-column",
        default="TIMESTAMP",
        help="CSV column for the x-axis, or 'index' to use 1-based data points (default: TIMESTAMP)",
    )
    parser.add_argument(
        "--left-column",
        nargs="+",
        metavar="COLUMN",
        help="One or more CSV columns for the left y-axis; selecting them creates one custom plot",
    )
    parser.add_argument(
        "--right-column",
        nargs="+",
        metavar="COLUMN",
        help="One or more CSV columns for the right y-axis",
    )
    parser.add_argument(
        "--output-name",
        help="Output filename for a custom plot (default: derived from selected columns)",
    )
    args = parser.parse_args()

    if args.right_column and not args.left_column:
        parser.error("--right-column requires --left-column")

    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if args.left_column:
        y_columns = args.left_column + (args.right_column or [])
        x_values, values = read_csv(args.input, args.x_column, y_columns)
        if not x_values:
            raise ValueError(f"No data rows found in {args.input}")
        x_label = (
            "Data point" if args.x_column.casefold() == "index" else args.x_column
        )
        filename = args.output_name or f"{args.left_column[0]}_vs_{x_label}.png"
        output_path = args.output_dir / filename
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_pair(
            x_values,
            values,
            args.left_column,
            args.right_column,
            x_label,
            output_path,
        )
        print(f"Wrote {output_path}")
        return

    y_columns = list(dict.fromkeys(column for plot in PLOTS for column in plot[:2]))
    x_values, values = read_csv(args.input, args.x_column, y_columns)
    if not x_values:
        raise ValueError(f"No data rows found in {args.input}")

    x_label = (
        "Data point" if args.x_column.casefold() == "index" else args.x_column
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for plot, filename in zip(PLOTS, args.output_names):
        left_column, right_column, _, _, _ = plot
        output_path = args.output_dir / filename
        plot_pair(
            x_values,
            values,
            [left_column],
            [right_column],
            x_label,
            output_path,
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()