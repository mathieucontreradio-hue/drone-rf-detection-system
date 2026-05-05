# drone-rf-detection-system
a drone rf detection system based on GNU RADIO and Hack RF ONE 
# Drone RF Detection System

## Project Overview

Development of a passive RF drone detection, classification,
tracking and localization system using SDR (Software Defined Radio),
GNU Radio, signal processing and machine learning.

The objective is to detect suspicious drone RF activity,
classify the signal type, track targets in real time,
and later integrate localization and hardware optimization.

---

## Main Objectives

- RF drone signal detection
- Signal classification (Drone / WiFi / FPV / Noise)
- Real-time tracking
- Localization using antenna systems
- RF front-end hardware optimization
- Full prototype development

---

## Project Roadmap

### Phase 1 — Detection MVP

Basic SDR front-end with:

- signal simulation
- FFT analysis
- threshold detection
- RF event detection

### Phase 2 — Machine Learning Classification

- dataset generation
- IQ recordings
- feature extraction
- Random Forest classifier
- CNN (later stage)

### Phase 3 — RF Hardware Receiver

- antennas
- low-noise amplifier (LNA)
- RF filters
- multi-band architecture

### Phase 4 — Real-Time Tracking

- Kalman filter
- target association
- continuous tracking

### Phase 5 — Localization System

- RSSI triangulation
- Direction Finding (DF)
- antenna arrays
- Angle of Arrival (AoA)

### Phase 6 — Full Integrated Prototype

Complete operational prototype.

---

## Target Frequencies

Primary monitoring bands:

- 433 MHz
- 868 MHz
- 2.4 GHz
- 5.8 GHz

---

## Technologies

- GNU Radio
- Python
- SDR (HackRF / RTL-SDR / USRP)
- Scikit-learn
- TensorFlow (future)
- RF design tools

---

## Current Progress

### RF Signal Simulation

- GMSK, FSK, AM and noise signals simulated using GNU Radio

### Feature Extraction

- Amplitude, phase and frequency features extracted
- Initial analysis of discriminative features

### Machine Learning

- K-means clustering implemented for unsupervised classification
- Preliminary grouping of modulation types achieved

### Next Step

- Supervised classification using Random Forest

## Experimental Validation

Each experiment is documented inside:

```text
/experiments/logs/logbook.md
