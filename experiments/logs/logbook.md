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

## Experiment #002 — K-means RF Signal Clustering (3 Features Limitation)

### Date

2026-05-04

---

### Objective

Evaluate the capability of K-means clustering to separate different RF signal types
(GMSK, FSK, AM, Noise) using a minimal feature set.

---

### Setup

* Signals:

  * GMSK
  * FSK
  * AM
  * Noise
* Data source: GNU Radio simulations
* Processing: Python (feature extraction + K-means)

---

### Parameters

* Number of clusters (K): 4
* Feature space dimension: 3
* Dataset: balanced between signal types
* Normalization: applied (standard scaling)

---

### Features Extracted

* Mean amplitude
* Amplitude variance
* Phase variance

---

### Observations

* Noise is relatively well separated due to high randomness
* FSK shows partial separation due to frequency-related behavior
* AM and GMSK overlap significantly in feature space
* Clusters are not clearly defined and show strong mixing

---

### Results

* K-means failed to produce clearly separable clusters
* Significant confusion between AM and GMSK classes
* Cluster boundaries are not aligned with actual modulation types
* Visual inspection (3D feature space) confirms overlap

---

### Problems

* Feature space too low-dimensional (3 features only)
* Selected features not sufficiently discriminative
* K-means assumes spherical clusters → not adapted to RF feature distributions
* AM and GMSK share similar statistical behavior in selected features
* Lack of frequency-domain features

---

### Conclusion

* K-means with limited features is not sufficient for reliable RF signal classification
* Feature engineering is critical for meaningful clustering
* AM and GMSK cannot be separated using only amplitude and phase statistics
* Unsupervised learning alone is insufficient in this configuration

---

### Key Insight

Adding more discriminative features (especially frequency-based or cyclostationary features)
is necessary to improve separability between modulation types.

---

### Next Step

* Increase feature set (frequency variance, higher-order statistics)
* Test higher-dimensional feature spaces
* Transition to supervised learning (Random Forest)
* Evaluate feature importance to guide selection

## Experiment #003 — Extended Feature Space for Robust RF Classification

### Date

2026-05-05

---

### Objective

Improve RF signal classification by expanding the feature space
and evaluating whether a richer set of features enables better separation
between modulation types (GMSK, FSK, AM, Noise).

---

### Setup

* Signals:

  * GMSK
  * FSK
  * AM
  * Noise
* Data source: GNU Radio simulations
* Processing: Python (feature extraction + preprocessing + clustering / preparation for supervised learning)

---

### Parameters

* Feature space dimension: extended (>3 features)
* Dataset: balanced across all classes
* Normalization: StandardScaler applied
* Windowing: fixed-size segments for feature computation

---

### Features Extracted

#### Amplitude Domain

* Mean amplitude
* Amplitude variance
* Amplitude standard deviation

#### Phase Domain

* Mean phase
* Phase variance
* Phase derivative variance

#### Frequency Domain

* Instantaneous frequency
* Frequency variance

#### Higher-Order Statistics

* Skewness
* Kurtosis

---

### Observations

* 
* 
* 
* 
* 

---

### Results

* 
* 
* 
* 

---

### Problems

* 
* 
* 
* 

---

### Conclusion

* 
* 
* 
* 

---

### Key Insight

Combining multiple domains (amplitude, phase, frequency)
is necessary to capture the full signature of RF modulations.

---

### Next Step

* Train Random Forest classifier on extended feature set
* Evaluate feature importance
* Reduce feature space using selection techniques
* Compare performance with previous K-means approach

