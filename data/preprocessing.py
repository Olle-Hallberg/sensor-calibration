import pandas as pd

# Läs in filerna
df = pd.read_csv("data/preprocessed/test_data.csv")

df["lgr_ch4_ppm"] = (df["lgr_ch4_ppb"]/1000).round(7)

df.to_csv("data/preprocessed/test_data.csv", index=False)

# # Starttid för fil 1
# start_time = pd.Timestamp("2026-05-11 21:18:06")

# # Gör elapsed_time till riktig tid
# file1["elapsed_time"] = pd.to_timedelta(file1["elapsed_time"])

# # Skapa absoluta timestamps
# file1["TIMESTAMP"] = start_time + file1["elapsed_time"]

# # Konvertera timestamps i fil 2
# file2["TIMESTAMP"] = pd.to_datetime(file2["TIMESTAMP"])

# # Lägg till LGR-kolumner genom timestamp-matchning
# merged = pd.merge(
#     file1,
#     file2[["TIMESTAMP", "LGR_CH4", "LGR_CO2"]],
#     on="TIMESTAMP",
#     how="left"
# )

# # Spara ny fil
# merged.to_csv("data/preprocessed/test/night_merged.csv", index=False)