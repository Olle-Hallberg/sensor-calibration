#pragma once
#include <Arduino.h>
#include <AltSoftSerial.h>

// ---------------------------------------------------------------
// K96Data — full parameter set recommended by the datasheet (section 5.2)
// for users building their own calibration/concentration model.
//
// Format notes (from datasheet):
//   S16     — signed 16-bit integer, scaled per field
//   S16.8   — 3-byte signed fixed-point: int16 + uint8 fraction, divide by 256
//   S32     — signed 32-bit integer (raw ADC accumulation, no scaling)
//   S32.16  — 6-byte signed fixed-point: int32 + uint16 fraction, divide by 65536
// ---------------------------------------------------------------

struct K96Data {

  // --- Environmental (from internal BME280, S16 format) ---
  float rh_temp;      // °C   — RH_Temp         (0x01F8, scale 0.01)
  float humidity;     // %RH  — RH               (0x01F0, scale 0.01)
  float pressure;     // hPa  — P_Sensor0_10Pa   (0x01D0, scale 0.1)

  // --- Environmental (from internal BME280, S16.8 format) ---
  float pressure_filtered; // hPa — P_Sensor0_10Pa_flt (0x01DE, scale 0.1)

  // --- NTC temperatures (S16.8 format, scale 0.01 °C) ---
  // NTC0 is physically near the SPL detector
  // NTC1 is physically between LPL and MPL detectors
  float ntc0_temp;            // °C — NTC0_Temp          (0x01B8)
  float ntc0_temp_filtered;   // °C — NTC0_Temp_filtered (0x01BC)
  float ntc1_temp;            // °C — NTC1_Temp          (0x01C0)
  float ntc1_temp_filtered;   // °C — NTC1_Temp_filtered (0x01C4)

  // --- Raw detector signals (S32 format, dimensionless ADC counts) ---
  int32_t lpl_signal;   // LPL_Signal  (0x0180)
  int32_t mpl_signal;   // MPL_Signal  (0x0360)
  int32_t spl_signal;   // SPL_Signal  (0x0190)

  // --- Filtered detector signals (S32.16 format) ---
  float lpl_signal_filtered;  // LPL_Signal_filtered (0x0184)
  float mpl_signal_filtered;  // MPL_Signal_filtered (0x0364)
  float spl_signal_filtered;  // SPL_Signal_filtered (0x0194)

  // --- SenseAir calculated concentrations (unfiltered, S16) ---
  // The sensor's own estimates — useful as reference baseline
  int16_t mpl_conc;     // MPL_uflt_Conc   (0x038A)
  int16_t lpl_conc;     // LPL_uflt_Conc   (0x042A)
  int16_t spl_conc;     // SPL_uflt_Conc   (0x04AA)
};

// ---------------------------------------------------------------

class K96Sensor {
public:
  K96Sensor(uint8_t deviceAddress = 0xFE);

  // Call once in setup()
  void begin(uint32_t baudRate = 115200);

  // Read the full calibration parameter set into data.
  // Returns true only if all reads succeed.
  bool readAll(K96Data &data);

private:
  AltSoftSerial _serial;  // Fixed pins on Arduino Nano: TX=9, RX=8
  uint8_t       _addr;
  byte          _buf[10]; // Largest payload is 6 bytes (S32.16); 10 gives headroom

  // Low-level Modbus RAM read — fills _buf with bytesToRead bytes
  bool     readRam(uint16_t address, uint8_t bytesToRead);
  uint16_t calculateCRC(byte *data, byte length);

  // Format decoders — all read from _buf after a successful readRam()
  int16_t  decodeS16();     // 2 bytes → signed int16
  int32_t  decodeS32();     // 4 bytes → signed int32
  float    decodeS16_8();   // 3 bytes → fixed-point, divide by 256, then scale 0.01
  float    decodeS32_16();  // 6 bytes → fixed-point, divide by 65536
};
