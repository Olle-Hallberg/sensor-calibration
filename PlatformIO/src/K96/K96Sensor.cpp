#include "K96Sensor.h"

// ---------------------------------------------------------------
// Register map (from K96 Integration User Guide, Table 5 / section 4.5)
// ---------------------------------------------------------------

// Environmental — S16
static const uint16_t REG_RH_TEMP   = 0x01F8;  // Temperature from BME280, 0.01°C
static const uint16_t REG_HUMIDITY  = 0x01F0;  // RH from BME280, 0.01%
static const uint16_t REG_PRESSURE  = 0x01D0;  // Pressure from BME280, 0.1 hPa

// Environmental — S16.8 (3 bytes)
static const uint16_t REG_PRESSURE_FILTERED = 0x01DE; // Pressure from BME280, filtered, 0.1 hPa

// NTC temperatures — S16.8 (3 bytes each)
static const uint16_t REG_NTC0_TEMP          = 0x01B8;  // Near SPL detector
static const uint16_t REG_NTC0_TEMP_FILTERED = 0x01BC;
static const uint16_t REG_NTC1_TEMP          = 0x01C0;  // Between LPL and MPL
static const uint16_t REG_NTC1_TEMP_FILTERED = 0x01C4;

// Raw detector signals — S32 (4 bytes each)
static const uint16_t REG_LPL_SIGNAL = 0x0180;
static const uint16_t REG_SPL_SIGNAL = 0x0190;
static const uint16_t REG_MPL_SIGNAL = 0x0360;

// Filtered detector signals — S32.16 (6 bytes each)
static const uint16_t REG_LPL_SIGNAL_FILTERED = 0x0184;
static const uint16_t REG_SPL_SIGNAL_FILTERED = 0x0194;
static const uint16_t REG_MPL_SIGNAL_FILTERED = 0x0364;

// SenseAir calculated concentrations (unfiltered) — S16
static const uint16_t REG_MPL_CONC = 0x038A;
static const uint16_t REG_LPL_CONC = 0x042A;
static const uint16_t REG_SPL_CONC = 0x04AA;

// ---------------------------------------------------------------
// Protocol constants
// ---------------------------------------------------------------

static const uint8_t  MODBUS_READ_CMD     = 0x44;
static const uint16_t RESPONSE_TIMEOUT    = 500;  // ms
static const uint8_t  INTER_REQUEST_DELAY = 20;   // ms — recovery gap between requests

// ---------------------------------------------------------------

K96Sensor::K96Sensor(uint8_t deviceAddress) : _addr(deviceAddress) {}

void K96Sensor::begin(uint32_t baudRate) {
  _serial.begin(baudRate);
}

// ---------------------------------------------------------------
// readAll — reads every parameter from Figure 5 of the datasheet
// ---------------------------------------------------------------

bool K96Sensor::readAll(K96Data &data) {
  bool ok = true;

  // --- Environmental (S16, 2 bytes) ---

  if (readRam(REG_RH_TEMP, 2))
    data.rh_temp = decodeS16() * 0.01f;
  else ok = false;

  if (readRam(REG_HUMIDITY, 2))
    data.humidity = decodeS16() * 0.01f;
  else ok = false;

  if (readRam(REG_PRESSURE, 2))
    data.pressure = decodeS16() * 0.1f;
  else ok = false;

  // --- Environmental (S16.8, 3 bytes) ---
  
  if (readRam(REG_PRESSURE_FILTERED, 3))
    data.pressure_filtered = decodeS16_8();
  else ok = false;

  // --- NTC temperatures (S16.8, 3 bytes) ---
  // S16.8: treat as 24-bit signed integer, divide by 256 to get value,
  // then multiply by 0.01 for °C (as per datasheet scale).

  if (readRam(REG_NTC0_TEMP, 3))
    data.ntc0_temp = decodeS16_8();
  else ok = false;

  if (readRam(REG_NTC0_TEMP_FILTERED, 3))
    data.ntc0_temp_filtered = decodeS16_8();
  else ok = false;

  if (readRam(REG_NTC1_TEMP, 3))
    data.ntc1_temp = decodeS16_8();
  else ok = false;

  if (readRam(REG_NTC1_TEMP_FILTERED, 3))
    data.ntc1_temp_filtered = decodeS16_8();
  else ok = false;

  // --- Raw detector signals (S32, 4 bytes) ---

  if (readRam(REG_LPL_SIGNAL, 4))
    data.lpl_signal = decodeS32();
  else ok = false;

  if (readRam(REG_MPL_SIGNAL, 4))
    data.mpl_signal = decodeS32();
  else ok = false;

  if (readRam(REG_SPL_SIGNAL, 4))
    data.spl_signal = decodeS32();
  else ok = false;

  // --- Filtered detector signals (S32.16, 6 bytes) ---
  // S32.16: 4-byte signed integer part + 2-byte unsigned fractional part,
  // combined value divided by 65536.

  if (readRam(REG_LPL_SIGNAL_FILTERED, 6))
    data.lpl_signal_filtered = decodeS32_16();
  else ok = false;

  if (readRam(REG_MPL_SIGNAL_FILTERED, 6))
    data.mpl_signal_filtered = decodeS32_16();
  else ok = false;

  if (readRam(REG_SPL_SIGNAL_FILTERED, 6))
    data.spl_signal_filtered = decodeS32_16();
  else ok = false;

  // --- SenseAir concentration estimates (S16, 2 bytes) ---

  if (readRam(REG_MPL_CONC, 2))
    data.mpl_conc = decodeS16();
  else ok = false;

  if (readRam(REG_LPL_CONC, 2))
    data.lpl_conc = decodeS16();
  else ok = false;

  if (readRam(REG_SPL_CONC, 2))
    data.spl_conc = decodeS16();
  else ok = false;

  return ok;
}

// ---------------------------------------------------------------
// Format decoders — each reads from _buf after a successful readRam()
// All multi-byte values are big-endian (MSB first) per Modbus convention.
// ---------------------------------------------------------------

int16_t K96Sensor::decodeS16() {
  return (int16_t)(((uint16_t)_buf[0] << 8) | _buf[1]);
}

int32_t K96Sensor::decodeS32() {
  return (int32_t)(((uint32_t)_buf[0] << 24) |
                   ((uint32_t)_buf[1] << 16) |
                   ((uint32_t)_buf[2] <<  8) |
                    (uint32_t)_buf[3]);
}

float K96Sensor::decodeS16_8() {
  // 3 bytes: [int_hi][int_lo][frac]
  // Reconstruct as 24-bit signed integer, then divide by 256 to get
  // the fixed-point value, then apply 0.01 scale for °C.
  int32_t raw = ((int32_t)(int8_t)_buf[0] << 16) |
                ((int32_t)        _buf[1] <<  8) |
                 (int32_t)        _buf[2];
  return (raw / 256.0f) * 0.01f;
}

float K96Sensor::decodeS32_16() {
  // 6 bytes: [int_b3][int_b2][int_b1][int_b0][frac_hi][frac_lo]
  // Reconstruct as 48-bit signed fixed-point: integer part (4 bytes) +
  // fractional part (2 bytes). Divide by 65536 to get float value.
  int32_t  intPart  = (int32_t)(((uint32_t)_buf[0] << 24) |
                                ((uint32_t)_buf[1] << 16) |
                                ((uint32_t)_buf[2] <<  8) |
                                 (uint32_t)_buf[3]);
  uint16_t fracPart = (uint16_t)(((uint16_t)_buf[4] << 8) | _buf[5]);
  return (float)intPart + (float)fracPart / 65536.0f;
}

// ---------------------------------------------------------------
// readRam — low-level Modbus RAM read
// ---------------------------------------------------------------

bool K96Sensor::readRam(uint16_t address, uint8_t bytesToRead) {
  // Flush stale bytes
  while (_serial.available()) _serial.read();

  // Build request frame: [addr][0x44][reg_hi][reg_lo][N]
  byte request[] = {
    _addr,
    MODBUS_READ_CMD,
    (byte)(address >> 8),
    (byte)(address & 0xFF),
    bytesToRead
  };

  uint16_t crc = calculateCRC(request, 5);
  _serial.write(request, 5);
  _serial.write((byte)(crc & 0xFF));        // CRC low byte first (Modbus standard)
  _serial.write((byte)((crc >> 8) & 0xFF));

  // Response: [addr][0x44][N][data x N][crc_lo][crc_hi]
  int  expectedBytes = 3 + bytesToRead + 2;
  byte response[15];  // Max payload is 6 bytes (S32.16) → max frame 11 bytes
  int  received = 0;

  unsigned long startTime = millis();
  while (millis() - startTime < RESPONSE_TIMEOUT) {
    if (_serial.available()) {
      response[received++] = _serial.read();
      if (received >= expectedBytes) break;
    }
  }

  // Error checks
  if (received == 0) {
    Serial.print(F("K96 TIMEOUT at 0x")); Serial.println(address, HEX);
    return false;
  }
  if (received != expectedBytes) {
    Serial.print(F("K96 WRONG LENGTH at 0x")); Serial.print(address, HEX);
    Serial.print(F(": got ")); Serial.print(received);
    Serial.print(F(", expected ")); Serial.println(expectedBytes);
    return false;
  }

  uint16_t recvCRC = ((uint16_t)response[expectedBytes - 1] << 8)
                   |  response[expectedBytes - 2];
  if (calculateCRC(response, expectedBytes - 2) != recvCRC) {
    Serial.print(F("K96 CRC ERROR at 0x")); Serial.println(address, HEX);
    return false;
  }

  // Copy payload into _buf
  for (int i = 0; i < bytesToRead; i++) {
    _buf[i] = response[3 + i];
  }

  delay(INTER_REQUEST_DELAY);
  return true;
}

// ---------------------------------------------------------------
// Standard Modbus CRC-16
// ---------------------------------------------------------------

uint16_t K96Sensor::calculateCRC(byte *data, byte length) {
  uint16_t crc = 0xFFFF;
  for (int pos = 0; pos < length; pos++) {
    crc ^= (uint16_t)data[pos];
    for (int i = 8; i != 0; i--) {
      if (crc & 0x0001) { crc >>= 1; crc ^= 0xA001; }
      else                 crc >>= 1;
    }
  }
  return crc;
}
