import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### dataframe ###
df = pd.read_csv('data/raw/atm_luft/humid_sensor_log_20260511_191212.csv')
# print(f'data.shape: {df.shape}')
# df.info()
# print(df.describe())
# print(df.head())

### Concentration calculations ###
C_air = ((df['mpx5500_pressure_hpa']*100) + 1000) / (8.314 * (df['rh_temp_c'] + 273.15))
C_CH4 = 2298.1/1e9 * C_air
C_CO2 = 450.1/1e6 * C_air

### plot ###
fig, (ax11, ax21) = plt.subplots(2, 1, figsize=(14, 7))

## Plot 1 ##
# Pressure
ax11.plot(df['elapsed_time'], df['mpx5500_pressure_hpa'], color='purple', linewidth=2)
ax11.set_ylabel("Pressure [hPa]", color='purple')
ax11.tick_params(axis='y', labelcolor='purple')

# Temperature
ax12 = ax11.twinx()
ax12.plot(df['elapsed_time'], df['rh_temp_c'], color='red', linewidth=2)
ax12.set_ylabel("Temperature [°C]", color='red')
ax12.tick_params(axis='y', labelcolor='red')

# Humidity
ax13 = ax11.twinx()
ax13.spines['right'].set_position(('outward', 50))
ax13.plot(df['elapsed_time'], df['humidity_pct'], color='blue', linewidth=2)
ax13.set_ylabel("Humidity [%]", color='blue')
ax13.tick_params(axis='y', labelcolor='blue')

# Labels and more
ax11.set_xticks(np.arange(0, df.shape[0] + 1, 1500))
ax11.set_xlabel("Time [hh:mm:ss]")
ax11.grid(True)

## Plot 2 ##
# LPL
ax21.plot(df['elapsed_time'], df['lpl_signal'], color='green', linewidth=2, label='LPL ($CO_2$, $CH_4$)')
ax21.plot(df['elapsed_time'], df['lpl_signal_filtered'], color='lightgreen', linewidth=2, label='LPL filtered ($CO_2$, $CH_4$)')
ax21.set_ylabel("LPL", color='green')
ax21.tick_params(axis='y', labelcolor='green')

# MPL
ax22 = ax21.twinx()
ax22.plot(df['elapsed_time'], df['mpl_signal'], color='gold', linewidth=2, label='MPL ($H_20$)')
ax22.plot(df['elapsed_time'], df['mpl_signal_filtered'], color='yellow', linewidth=2, label='MPL filtered ($H_20$)')
ax22.set_ylabel("MPL", color='gold')
ax22.tick_params(axis='y', labelcolor='gold')

# SPL
ax23 = ax21.twinx()
ax23.spines['right'].set_position(('outward', 50))
ax23.plot(df['elapsed_time'], df['spl_signal'], color='red', linewidth=2, label='SPL ($CO_2$)')
ax23.plot(df['elapsed_time'], df['spl_signal_filtered'], color='lightcoral', linewidth=2, label='SPL filtered ($CO_2$)')
ax23.set_ylabel("SPL", color='red')
ax23.tick_params(axis='y', labelcolor='red')
ax23.yaxis.get_offset_text().set_x(1.059)

# Calculated concentrations
# ax21.plot(df['elapsed_time'], C_CH4, color='magenta', linewidth=2, label='$CH_4$ Concentration [mol/m³] (calculated)')
# ax21.plot(df['elapsed_time'], C_CO2, color='orange', linewidth=2, label='$CO_2$ Concentration [mol/m³] (calculated)')

# Concentration signals
# ax22.plot(df['elapsed_time'], df['lpl_conc'], '--', color='purple', linewidth=2, label='LPL conc ($CO_2$, $CH_4$)')
# ax22.plot(df['elapsed_time'], df['mpl_conc'], '--', color='yellow', linewidth=2, label='MPL conc ($H_20$)')
# ax22.plot(df['elapsed_time'], df['spl_conc'], '--', color='pink', linewidth=2, label=' SPL conc ($CO_2$)')

# Reference concentrations CH4
# ax22.plot(df['elapsed_time'], np.linspace(1967.2/1000, 1967.2/1000, len(df)), color='black', linewidth=2, label='CH4 [ppm] (Tek. luft)')
# ax22.plot(df['elapsed_time'], np.linspace(2298.1/1000, 2298.1/1000, len(df)), color='black', linewidth=2, label='$CH_4$ [ppm] (Atm. luft)')
# ax22.plot(df['elapsed_time'], np.linspace(1796.08/1000, 1796.08/1000, len(df)), color='black', linewidth=2, label='CH4 [ppm] (CORE-1)')
# ax22.plot(df['elapsed_time'], np.linspace(2194.89/1000, 2194.89/1000, len(df)), color='black', linewidth=2, label='CH4 [ppm] (CORE-2)')

# Reference concentrations CO2
# ax22.plot(df['elapsed_time'], np.linspace(435.8, 435.8, len(df)), color='grey', linewidth=2, label='CO2 [ppm] (Tek. luft)')
# ax22.plot(df['elapsed_time'], np.linspace(450.1, 450.1, len(df)), color='grey', linewidth=2, label='$CO_2$ [ppm] (Atm. luft)')
# ax22.plot(df['elapsed_time'], np.linspace(379.63, 379.63, len(df)), color='black', linewidth=2, label='CO2 [ppm] (CORE-1)')
# ax22.plot(df['elapsed_time'], np.linspace(465.52, 465.52, len(df)), color='black', linewidth=2, label='CO2 [ppm] (CORE-2)')

# Labels and more
ax21.set_xticks(np.arange(0, df.shape[0] + 1, 1500))
ax21.set_xlabel("Time [hh:mm:ss]")
ax21.grid(True)

handles = []
labels = []
for ax in [ax21, ax22, ax23]:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
ax23.legend(handles, labels, loc='upper right', framealpha=0.25)

# Adjust layout and show plot
plt.tight_layout()
plt.show()