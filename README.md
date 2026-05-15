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
