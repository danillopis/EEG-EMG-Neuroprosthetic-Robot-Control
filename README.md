# EEG-EMG-Neuroprosthetic-Robot-Control
Hybrid EEG-EMG neuroprosthetic control system for real-time robotic arm operation using motor imagery classification, CSP, LDA/SVM, Arduino, and PhantomX Reactor.

This repository contains the implementation of a hybrid EEG-EMG neuroprosthetic control system developed for Deliverable 3 of the Neuroprosthetics course in the MSc in Science in Neurotechnology at Universidad Politécnica de Madrid.

The project integrates EEG-based motor imagery classification, EMG-based confirmation, machine learning, serial communication, and robotic actuation using a PhantomX Reactor 5-DOF robotic arm.

## Project Overview

The objective of this project is to control a robotic manipulator using biological signals.

The system combines:

- EEG acquisition using a BitBrain device and LSL streaming.
- EEG preprocessing and filtering.
- Common Spatial Pattern feature extraction.
- Machine learning classification of motor imagery tasks.
- Serial communication between Python and Arduino.
- EMG-based grasp and release confirmation.
- Real-time execution on the PhantomX Reactor robotic platform.

## Final Online Architecture

The final deployed EEG pipeline uses a hierarchical two-stage classifier:

1. Rest vs Motor Imagery classification using Linear Discriminant Analysis.
2. Left vs Right motor imagery classification using a linear Support Vector Machine.

The online processing pipeline includes:

```text
EEG acquisition through LSL
        ↓
Common Average Referencing
        ↓
High-pass filtering at 1 Hz
        ↓
Notch filtering at 50 Hz
        ↓
Band-pass filtering between 8 and 30 Hz
        ↓
Sliding windows of 2.5 s updated every 0.5 s
        ↓
CSP feature extraction
        ↓
Log-variance feature computation
        ↓
StandardScaler normalization
        ↓
LDA Rest vs Motor Imagery
        ↓
SVM Left vs Right
        ↓
Temporal voting and thresholding
        ↓
Serial command to Arduino
        ↓
Robot movement + EMG confirmation
```

## Repository Structure

```text
.
├── README.md
├── D3_NPTS.pdf
├── requirements.txt
├── python/
│   └── eeg_emg_robot_python.py
├── arduino/
│   └── GR01-D3/
│       └── GR01-D3.ino
├── models/
│   └── GR01-D3_eeg_csp_model.pkl
├── eeg_recordings/
│   └── EEG CSV recordings
└── eeg_plots/
    └── Generated EEG plots and confusion matrices
```

## Hardware

* BitBrain EEG acquisition system
* EEG cap with electrodes placed over sensorimotor areas
* EMG sensor connected to the forearm
* Arduino-compatible controller
* PhantomX Reactor 5-DOF robotic arm
* Laptop or PC for Python-based processing

## Software

* Python 3
* Arduino IDE
* Lab Streaming Layer
* NumPy
* SciPy
* Pandas
* Matplotlib
* Scikit-learn
* MNE
* PySerial
* pylsl

## How to Run

1. Install Python dependencies

```text
pip install -r requirements.txt
```

2. Upload Arduino firmware

Open the Arduino sketch:

```text
arduino/GR01-D3/GR01-D3.ino
```

Upload it to the Arduino-compatible controller connected to the PhantomX Reactor platform.

3. Start EEG acquisition

Start the BitBrain acquisition software and make sure the EEG stream is available through LSL.

4. Run the Python control script

```text
python python/eeg_emg_robot_python.py
```

The script can either load a previously trained subject-specific model or record new EEG trials and retrain the classifiers.

Generated Outputs

The Python pipeline generates:

* Raw EEG plots
* Band-power visualizations
* Confusion matrices
* Recorded EEG CSV files
* Trained model files

## Report

The full written report is available in:

```text
GR01-D3.pdf
```
## Authors

Daniel Llopis Conejo
Matías Nevado García

MSc in Science in Neurotechnology
Universidad Politécnica de Madrid



