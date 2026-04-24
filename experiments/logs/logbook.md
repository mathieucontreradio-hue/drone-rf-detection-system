## Experiment #001 — Drone WiFi Signal Simulator V1

### Date

2026-04-25

---

### Objective

Create a first simulated drone WiFi-like RF signal
using GNU Radio for FFT detection and future ML dataset generation.

---

### Flowgraph Architecture

Random Source
→ Chunks to Symbols (QPSK)
→ RRC Filter
→ Multiply Const
→ Throttle
→ Channel Model
→ QT GUI Frequency Sink

---

### Parameters

Sample Rate: 2 MS/s

Modulation:
QPSK

RRC Filter:

* Samples per Symbol: 4
* Roll-off Factor: 0.35

Channel Model:

* Noise Voltage: 0.05
* Frequency Offset: 0.001
* Timing Offset: 1.0

Amplitude Gain:
0.5

---

### Expected Result

Visible occupied bandwidth on FFT
with clear spectral signature
different from pure noise.

---

### Actual Result

(To complete after test)

Example:
FFT shows clear occupied spectrum.
Signal visible and stable.
Noise level acceptable.

---

### Problems Encountered

(To complete)

Example:
Throttle instability
Incorrect constellation mapping
Too much noise masking FFT

---

### Conclusion

(To complete)

Example:
Simulation successful.
Next step is burst transmission modeling.

---

### Next Step

Add burst gate for intermittent transmission
to better simulate real drone behavior.

