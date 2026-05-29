"""
serial_logger.py — reads CSV data from Arduino over serial and saves to file.

Usage:
    python serial_logger.py                                 # uses defaults below
    python serial_logger.py --port COM3                     # Windows
    python serial_logger.py --port /dev/ttyUSB0             # Linux
    python serial_logger.py --port /dev/cu.usbmodem14101    # macOS
    python serial_logger.py --port COM3 --out my_run.csv

Requires:  pip install pyserial
"""

import argparse
import csv
import os
import serial
import sys
from datetime import datetime

# ---------------------------------------------------------------
# Defaults — change these if you always use the same port

DEFAULT_PORT     = "/dev/cu.usbserial-AH06N8K9" # obtained from running /dev/cu.usbserial* in terminal
DEFAULT_BAUDRATE = 9600
DEFAULT_OUT_DIR  = "data_logs"

# Every CSV line from the Arduino starts with this prefix
CSV_PREFIX = "DATA,"

# ---------------------------------------------------------------

def find_serial_port():
    """Try to auto-detect the Arduino port (Linux/macOS only)."""
    import glob
    candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/cu.usbmodem*")
    return candidates[0] if candidates else DEFAULT_PORT


def make_output_path(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"sensor_log_{timestamp}.csv")


def run(port: str, baudrate: int, out_path: str):
    print(f"Opening serial port {port} at {baudrate} baud...")
    print(f"Logging to: {out_path}")
    print("Press Ctrl+C to stop.\n")

    try:
        ser = serial.Serial(port, baudrate, timeout=2)
    except serial.SerialException as e:
        print(f"ERROR: Could not open port — {e}")
        sys.exit(1)

    header_written = False
    row_count = 0

    with open(out_path, "w", newline="") as csvfile:
        writer = None  # initialised when we see the header row

        try:
            while True:
                raw = ser.readline()
                if not raw:
                    continue

                try:
                    line = raw.decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue  # skip garbled bytes at startup

                # Echo everything to terminal so you can see what's happening
                print(line)

                # Only process lines that start with our DATA prefix
                if not line.startswith(CSV_PREFIX):
                    continue

                # Strip the prefix and parse as CSV
                data = line[len(CSV_PREFIX):]
                row = data.split(",")

                if not header_written:
                    # First DATA line is the header
                    writer = csv.writer(csvfile)
                    writer.writerow(row)
                    csvfile.flush()
                    header_written = True
                    print(f"  → Header captured: {row}")
                else:
                    writer.writerow(row)
                    csvfile.flush()  # flush every row so data isn't lost on Ctrl+C
                    row_count += 1
                    print(f"  → Row {row_count} saved.")

        except KeyboardInterrupt:
            print(f"\nStopped. {row_count} data rows saved to {out_path}")
        finally:
            ser.close()


# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log Arduino sensor data to CSV.")
    parser.add_argument("--port",     default=None,            help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud",     default=DEFAULT_BAUDRATE, type=int, help="Baud rate (default 9600)")
    parser.add_argument("--out",      default=None,            help="Output CSV file path")
    parser.add_argument("--out-dir",  default=DEFAULT_OUT_DIR, help="Output directory (default: logs/)")
    args = parser.parse_args()

    port     = args.port or find_serial_port()
    out_path = args.out  or make_output_path(args.out_dir)

    run(port, args.baud, out_path)
