# Calibration of an NDIR Gas Sensor for Stratospheric Pressure and Temperature Conditions
>This project is part of the Methane Infrared Absorption Gas Experiment (MIRAGE). MIRAGE is Uppsala University’s submission to the Balloon Experiment for University Students (BEXUS), a programme for student experiments conducted on a balloon.

---
## Project Introduction
#### MIRAGE Mission Statement
Methane measurements are of great importance for climate modelling. However, systems with which methane measurements are taken, are oftentimes expensive and complex in setup. Therefore, MIRAGE aims to demonstrate a low-cost, low-complexity solution for measuring methane concentrations in the upper stratosphere. By using a novel non-dispersive infrared (NDIR) sensor, off-the-shelf technology will be shown to provide an alternative to laser-based approaches.

#### Problem Description
The NDIR sensor is sensitive to temperature and humidity variations, leading to measurement uncertainties. The goal of this project is therefore to improve the accuracy through calibration, with the aim of enabling high-quality greenhouse gas measurements. This will be achieved using machine learning (ML) models and interpolation-based calibration methods.

---
## Repository Structure
```
data/
├── preprocessed/             Preprocessed datasets for calibration and evaluation
│                             Includes training, testing, and aggregated datasets
│
├── raw/                      Raw measurement data
│                             Includes calibration gas and nitrogen datasets
│
├── analysis.py               Data visualization script
└── preprocessing.py          Data preprocessing scripts

interpolation/                Interpolation-based calibration methods

ML/                           Machine learning models for sensor calibration

PlatformIO/                   Embedded firmware and sensor data acquisition
├── data_logs/                Logged sensor measurements
│   └── data_analysis.py      Data visualization script
├── src/
│   ├── Data/                 Real-time sensor data logging and export
│   │   └── serial_logger.py  Serial data acquisition and CSV logging script
│   ├── K96/                  NDIR methane sensor communication and data decoding
│   └── MPX5500/              Pressure sensor interface and conversion
└── platformio.ini            PlatformIO project configuration
```
