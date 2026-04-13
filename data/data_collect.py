import numpy as np
import matplotlib.pyplot as plt

### Define parameters ###
# Time [s]
t_min = 0
t_max = 11760
t_step = 0.1

# Temperature [°C]
T_min = 15
T_max = 45
T_num_levels = 4    # number of T levels per CH4 level

# Relative Humidity [%]
H_min = 0
H_max = 100
H_num_levels = 4    # number of H levels per T level

# Pressure [Bar]
P_min = 2.5
P_max = 3.2

##############################################################################################

### Make calibration profile ###
# time [s]
t = np.arange(t_min, t_max, t_step) # t_max/t_step data points

# CH4 [ppb]
CH4_1 = 1796.08 # CORE-1
CH4_2 = 1967.2  # Tek. luft
CH4_3 = 2194.89 # CORE-2
CH4_4 = 2298.1  # Atm. luft
CH4_levels = [CH4_1, CH4_2, CH4_3, CH4_4]
CH4_segment = t_max / len(CH4_levels)               # duration of each CH4 level in seconds
CH4_idx = (t // CH4_segment).astype(int)            # index for CH4 level based on time
CH4_idx = np.clip(CH4_idx, 0, len(CH4_levels) - 1)  # ensure index is within bounds
CH4 = np.array(CH4_levels)[CH4_idx]                 # CH4 concentration profile over time

# CO2 [ppm]
CO2_2 = 435.8  # Tek. luft
CO2_4 = 450.1  # Atm. luft
CO2_levels = [0, CO2_2, 0, CO2_4]
CO2_idx = (t // CH4_segment).astype(int)            # index of each CO2 segment based on time
CO2_idx = np.clip(CO2_idx, 0, len(CO2_levels) - 1)  # ensure index is within bounds
CO2 = np.array(CO2_levels)[CO2_idx]                 # CO2 concentration profile over time

# Temperature [°C]
T_levels = np.linspace(T_min, T_max, T_num_levels)
T_cycle = np.concatenate([T_levels, T_levels[-2::-1]])  # create a cycle that goes up and then down
T_segment = CH4_segment / len(T_cycle)                  # duration of each T level in seconds
T_local_t = t % CH4_segment                             # local time within the current CH4 segment
T_idx = (T_local_t // T_segment).astype(int)            # index for T level based on local time
T_idx = np.clip(T_idx, 0, len(T_cycle) - 1)             # ensure index is within bounds
T = np.array(T_cycle)[T_idx]                            # Temperature profile over time, cycling through T levels within each CH4 segment

# Humidity [%]
H_levels = np.linspace(H_min, H_max, H_num_levels)
H_cycle = np.concatenate([H_levels, H_levels[-2::-1]])  # create a cycle that goes up and then down
H_segment = T_segment / len(H_cycle)                    # duration of each H level in seconds
H_local_t = T_local_t % T_segment                       # local time within the current T segment
H_idx = (H_local_t // H_segment).astype(int)            # index for H level based on local time
H_idx = np.clip(H_idx, 0, len(H_cycle) - 1)             # ensure index is within bounds
H = np.array(H_cycle)[H_idx]                            # Humidity profile over time, cycling through H levels within each T segment

### Plot ###
fig, (ax11, ax21) = plt.subplots(2, 1, figsize=(14, 7))

## Temperature and Humidity over Time ##
# Temperature (left axis)
ax11.plot(t, T, color='red', linewidth=2)
ax11.set_ylabel("Temperature [°C]", color='red')
ax11.tick_params(axis='y', labelcolor='red')

# Humidity (right axis)
ax12 = ax11.twinx()
ax12.plot(t, H, color='blue', linewidth=2)
ax12.set_ylabel("Humidity [%]", color='blue')
ax12.tick_params(axis='y', labelcolor='blue')

# Labels and more
ax11.set_xlabel("Time [s]")
ax11.set_title("Temperature and Humidity over Time")
ax11.grid(True)

## CH4 and CO2 over Time ##
# CH4 (left axis)
ax21.plot(t, CH4, color='green', linewidth=2)
ax21.set_ylabel("CH4 [ppb]", color='green')
ax21.tick_params(axis='y', labelcolor='green')

# CO2 (right axis)
ax22 = ax21.twinx()
ax22.plot(t, CO2, color='orange', linewidth=2)
ax22.set_ylabel("CO2 [ppm]", color='orange')
ax22.tick_params(axis='y', labelcolor='orange')

# Labels and more
ax21.set_zorder(2)
ax22.set_zorder(1)
ax21.patch.set_visible(False)
ax21.set_xlabel("Time [s]")
ax21.set_title("CH4 and CO2 over Time")
ax21.grid(True)

# Adjust layout and show plot
plt.tight_layout()
plt.show()