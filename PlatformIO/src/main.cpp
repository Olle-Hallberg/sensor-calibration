#include <Arduino.h>
#include "K96/K96Sensor.h"
#include "MPX5500/MPX5500Sensor.h"
#include "Data/DataLogger.h"

// ---------------------------------------------------------------
// Configuration

static const unsigned long READ_INTERVAL_MS = 1000;  // Sensor read rate (ms)
static const unsigned long LOG_INTERVAL_MS  = 1000;  // CSV logging rate (ms)
// ---------------------------------------------------------------

K96Sensor k96;
K96Data   k96Data;
MPX5500Sensor mpx5500(A3);
MPX5500Data   mpx5500Data;

// ---------------------------------------------------------------

void printHumanReadable() {
  Serial.println();

  Serial.println(F("--- MPX5500 ---"));

  Serial.print(F("Pressure    = ")); Serial.print(mpx5500Data.pressure_hPa, 2);   Serial.println(F(" hPa"));
  
  Serial.println();

  Serial.println(F("--- K96 ---"));

  Serial.print(F("Temp (BME)  = ")); Serial.print(k96Data.rh_temp, 2);   Serial.println(F(" *C"));
  Serial.print(F("Humidity    = ")); Serial.print(k96Data.humidity, 2);   Serial.println(F(" %RH"));
  Serial.print(F("Pressure    = ")); Serial.print(k96Data.pressure, 2);   Serial.println(F(" hPa"));
  Serial.print(F("Pressure (filtered) = ")); Serial.print(k96Data.pressure_filtered, 2);   Serial.println(F(" hPa"));

  Serial.print(F("NTC0 (SPL)  = ")); Serial.print(k96Data.ntc0_temp, 4); Serial.println(F(" *C"));
  Serial.print(F("NTC1 (LPL/MPL) = ")); Serial.print(k96Data.ntc1_temp, 4); Serial.println(F(" *C"));

  Serial.print(F("LPL raw     = ")); Serial.println(k96Data.lpl_signal);
  Serial.print(F("MPL raw     = ")); Serial.println(k96Data.mpl_signal);
  Serial.print(F("SPL raw     = ")); Serial.println(k96Data.spl_signal);

  Serial.print(F("MPL conc    = ")); Serial.println(k96Data.mpl_conc);
  Serial.print(F("LPL conc    = ")); Serial.println(k96Data.lpl_conc);
  Serial.print(F("SPL conc    = ")); Serial.println(k96Data.spl_conc);

  Serial.println();
}

// ---------------------------------------------------------------

void setup() {
  Serial.begin(9600);
  Serial.println();
  Serial.println(F("Initialising sensors..."));
  Serial.println();

  k96.begin();
  mpx5500.begin();

  delay(1000);

  Serial.println(F("Data header:"));
  DataLogger::printHeader();

  Serial.println(F("Ready."));
  Serial.println();
}

// ---------------------------------------------------------------

void loop() {
  static unsigned long lastReadTime = 0;
  static unsigned long lastLogTime  = 0;
  unsigned long now = millis();

  if (now - lastReadTime >= READ_INTERVAL_MS) {
    lastReadTime = now;

    k96.readAll(k96Data);
    mpx5500.read(mpx5500Data);

    printHumanReadable();

    // OPTIONAL: overwrite K96 pressure so logger uses it
    // k96Data.pressure = mpx5500Data.pressure_hPa;
  }

  if (now - lastLogTime >= LOG_INTERVAL_MS) {
    lastLogTime = now;
    Serial.println(F("--- Data ---"));
    DataLogger::printRow(k96Data, mpx5500Data);
    Serial.println();
  }
}
