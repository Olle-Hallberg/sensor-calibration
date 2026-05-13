import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### dataframe ###
df = pd.read_csv('teknisk luft/Preprocessed/data_teknisk_luft.csv')
# print(f'data.shape: {df.shape}')
# df.info()
# print(df.describe())
# print(df.head())

### Concentration calculations ###
C_air = ((df['mpx5500_pressure_hpa'] + 1013) * 100) / (8.314 * (df['rh_temp_c'] + 273.15))
C_CH4 = 1967.2/1000 * C_air
C_CO2 = 435.8 * C_air

### COncentration columns ###
df['ch4_conc_mol_m3'] = C_CH4
df['co2_conc_mol_m3'] = C_CO2

### Save ###
df.to_csv('teknisk luft/Preprocessed/data_teknisk_luft.csv', index=False)