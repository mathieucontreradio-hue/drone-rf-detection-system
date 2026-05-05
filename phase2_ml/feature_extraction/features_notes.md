# RF Feature Extraction Notes

## Amplitude-Based Features

### Mean Amplitude

Average signal magnitude

### Amplitude Variance

* High for AM
* Low for constant-envelope signals (FSK, GMSK)

---

## Phase-Based Features

### Mean Phase

Not very discriminative alone

### Phase Variance

* High for noisy signals
* Useful for distinguishing modulation types

---

## Frequency-Based Features

### Instantaneous Frequency

Derived from phase derivative

### Frequency Variance

* High for FSK
* Lower for stable carriers

---

## Key Insights

* AM can be identified by amplitude fluctuations
* FSK/GMSK are constant-envelope signals
* Noise shows high randomness across all domains
