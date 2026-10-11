#pragma once
#include <Arduino.h>
#include "K96/K96Sensor.h"
# include "MPX5500/MPX5500Sensor.h"

class DataLogger {
public:
  // Print the CSV header line — call once in setup()
  static void printHeader();

  // Format and print one CSV data row to Serial
  static void printRow(const K96Data &k96, const MPX5500Data &mpx5500);

private:
  static void formatElapsed(unsigned long ms, char *dest);
};
