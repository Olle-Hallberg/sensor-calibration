#include "DataLogger.h"

static const char CSV_PREFIX[] = "DATA,";

// ---------------------------------------------------------------

void DataLogger::printHeader() {
  Serial.println(F(
    "DATA,elapsed_time,"
    // Environmental
    "mpx5500_pressure_hpa,"
    "rh_temp_c,humidity_pct,k96_pressure_hpa,k96_pressure_filtered_hpa,"
    // NTC temperatures
    "ntc0_temp_c,ntc0_temp_filtered_c,ntc1_temp_c,ntc1_temp_filtered_c,"
    // Raw detector signals
    "lpl_signal,mpl_signal,spl_signal,"
    // Filtered detector signals
    "lpl_signal_filtered,mpl_signal_filtered,spl_signal_filtered,"
    // SenseAir concentration estimates
    "mpl_conc,lpl_conc,spl_conc"
  ));
}

// ---------------------------------------------------------------

void DataLogger::printRow(const K96Data &k96, const MPX5500Data &mpx5500) {
  char timestamp[9];
  formatElapsed(millis(), timestamp);

  Serial.print(CSV_PREFIX);
  Serial.print(timestamp);                     Serial.print(',');

  // --- MPX5500 ---
  Serial.print(mpx5500.pressure_hPa, 2);       Serial.print(',');

  // --- K96 ---
  // Environmental
  Serial.print(k96.rh_temp, 2);                Serial.print(',');
  Serial.print(k96.humidity, 2);               Serial.print(',');
  Serial.print(k96.pressure, 2);               Serial.print(',');
  Serial.print(k96.pressure_filtered, 2);      Serial.print(',');

  // NTC temperatures
  Serial.print(k96.ntc0_temp, 4);              Serial.print(',');
  Serial.print(k96.ntc0_temp_filtered, 4);     Serial.print(',');
  Serial.print(k96.ntc1_temp, 4);              Serial.print(',');
  Serial.print(k96.ntc1_temp_filtered, 4);     Serial.print(',');

  // Raw detector signals (integer, full precision)
  Serial.print(k96.lpl_signal);                Serial.print(',');
  Serial.print(k96.mpl_signal);                Serial.print(',');
  Serial.print(k96.spl_signal);                Serial.print(',');

  // Filtered detector signals (float, high precision)
  Serial.print(k96.lpl_signal_filtered, 4);    Serial.print(',');
  Serial.print(k96.mpl_signal_filtered, 4);    Serial.print(',');
  Serial.print(k96.spl_signal_filtered, 4);    Serial.print(',');

  // SenseAir concentration estimates
  Serial.print(k96.mpl_conc);                  Serial.print(',');
  Serial.print(k96.lpl_conc);                  Serial.print(',');
  Serial.println(k96.spl_conc);
  }

// ---------------------------------------------------------------

void DataLogger::formatElapsed(unsigned long ms, char *dest) {
  unsigned long totalSeconds = ms / 1000;
  unsigned int  hours   = totalSeconds / 3600;
  unsigned int  minutes = (totalSeconds % 3600) / 60;
  unsigned int  seconds = totalSeconds % 60;

  dest[0] = '0' + hours   / 10;  dest[1] = '0' + hours   % 10;
  dest[2] = ':';
  dest[3] = '0' + minutes / 10;  dest[4] = '0' + minutes % 10;
  dest[5] = ':';
  dest[6] = '0' + seconds / 10;  dest[7] = '0' + seconds % 10;
  dest[8] = '\0';
}
