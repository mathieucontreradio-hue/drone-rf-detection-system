import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from scipy.stats import kurtosis
from scipy.signal import welch

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# PARAMETERS
# ============================================================
FS = 48000
N = 1024
SAMPLES_PER_CLASS = 3000

SNRS = [-10, -5, 0, 5, 10, 15, 20, 25]

class_names = {
    0: "Noise",
    1: "AM",
    2: "GMSK",
    3: "FSK"
}

# ============================================================
# IMPERFECTIONS SDR REALISTES
# ============================================================

def add_awgn(iq, snr_db):

    sig_power = np.mean(np.abs(iq)**2)
    snr = 10**(snr_db / 10)
    noise_power = sig_power / snr

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(iq)) + 1j*np.random.randn(len(iq))
    )

    return iq + noise


def add_cfo(iq, max_offset=0.01):

    N = len(iq)
    t = np.arange(N)

    f_off = np.random.uniform(-max_offset, max_offset)

    return iq * np.exp(1j * 2*np.pi * f_off * t)


def add_phase_noise(iq, sigma=0.02):

    noise = np.cumsum(np.random.randn(len(iq)) * sigma)
    return iq * np.exp(1j * noise)


def add_gain_variation(iq):

    gain = np.random.uniform(0.5, 1.5)
    return iq * gain


def add_dc_offset(iq):

    dc = (np.random.randn() + 1j*np.random.randn()) * 0.05
    return iq + dc


def add_fading(iq):

    h = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
    return iq * h


def add_timing_shift(iq):

    shift = np.random.randint(0, 50)
    return np.roll(iq, shift)

# ============================================================
# SIGNALS
# ============================================================

def gen_noise():
    return np.random.randn(N) + 1j*np.random.randn(N)

def gen_am():

    t = np.arange(N) / FS
    carrier = np.exp(1j*2*np.pi*5000*t)
    mod = 1 + 0.7*np.sin(2*np.pi*1000*t)

    return mod * carrier

def gen_fsk():

    bits = np.random.randint(0, 2, 32)
    freq = np.repeat(bits, N//32)
    freq = 2000 + freq*4000

    phase = 2*np.pi*np.cumsum(freq)/FS

    return np.exp(1j*phase)

def gen_gmsk():

    data = np.random.choice([-1, 1], N)
    phase = np.cumsum(data * 0.03)

    return np.exp(1j*phase)

# ============================================================
# FEATURES
# ============================================================

def extract_features(iq):

    amp = np.abs(iq)
    phase = np.unwrap(np.angle(iq))
    dphi = np.diff(phase)
    inst_freq = dphi / (2*np.pi)

    f, psd = welch(iq, fs=FS)
    psd = psd / (np.sum(psd) + 1e-12)

    entropy = -np.sum(psd * np.log2(psd + 1e-12))

    spectral_flatness = np.exp(np.mean(np.log(psd + 1e-12))) / (np.mean(psd) + 1e-12)

    cyclo = np.abs(np.mean(iq[:-1] * np.conj(iq[1:])))

    return {
        "var_amp": np.var(amp),
        "mean_dphi": np.mean(dphi),
        "var_dphi": np.var(dphi),
        "var_ifreq": np.var(inst_freq),
        "kurt_ifreq": kurtosis(inst_freq),
        "entropy": entropy,
        "spectral_flatness": spectral_flatness,
        "kurtosis": kurtosis(np.real(iq)),
        "cyclo": cyclo
    }

# ============================================================
# DATASET
# ============================================================

rows = []

classes = [
    (gen_noise, 0),
    (gen_am, 1),
    (gen_gmsk, 2),
    (gen_fsk, 3)
]

for gen, label in classes:

    print("Class", label)

    for _ in range(SAMPLES_PER_CLASS):

        iq = gen()

        # ====================================================
        # SDR REALISM PIPELINE
        # ====================================================
        iq = add_gain_variation(iq)
        iq = add_dc_offset(iq)
        iq = add_fading(iq)
        iq = add_cfo(iq)
        iq = add_phase_noise(iq)
        iq = add_timing_shift(iq)
        iq = add_awgn(iq, np.random.choice(SNRS))

        feat = extract_features(iq)
        feat["label"] = label

        rows.append(feat)

df = pd.DataFrame(rows)

df.to_csv("dataset_rf_realistic.csv", index=False)

print("\nDataset saved ✔")

# ============================================================
# SPLIT
# ============================================================

X = df.drop(columns=["label"])
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# PCA
# ============================================================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()

for c in sorted(y.unique()):
    idx = (y == c)
    plt.scatter(X_pca[idx,0], X_pca[idx,1], s=5, label=class_names[c])

plt.title("PCA - Realistic SDR Dataset")
plt.legend()
plt.show()

# ============================================================
# t-SNE
# ============================================================

tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure()

for c in sorted(y.unique()):
    idx = (y == c)
    plt.scatter(X_tsne[idx,0], X_tsne[idx,1], s=5, label=class_names[c])

plt.title("t-SNE - Realistic SDR Dataset")
plt.legend()
plt.show()

# ============================================================
# TRAIN RF
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=400,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure()

sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names.values(),
            yticklabels=class_names.values())

plt.title("Confusion Matrix - Realistic SDR")
plt.show()

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nMODEL SAVED ✔")
