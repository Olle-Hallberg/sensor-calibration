import pandas as pd

# Read files
file1 = pd.read_csv("data/raw/test/sensor_log_20260511_211806.csv")
file2 = pd.read_csv("DECOUPLING_LGRdata_2026-05-11.csv")

# Start time for file 1
start_time = pd.Timestamp("2026-05-11 21:18:06")

# Make elapsed_time to real time
file1["elapsed_time"] = pd.to_timedelta(file1["elapsed_time"])

# Create absolute timestamps
file1["TIMESTAMP"] = start_time + file1["elapsed_time"]

# Convert timestamps in file 2
file2["TIMESTAMP"] = pd.to_datetime(file2["TIMESTAMP"])

# Add LGR data to file 1 based on TIMESTAMP
merged = pd.merge(
    file1,
    file2[["TIMESTAMP", "LGR_CH4", "LGR_CO2"]],
    on="TIMESTAMP",
    how="left"
)

# Save the merged data
merged.to_csv("some_place_on_your_computer.csv", index=False)