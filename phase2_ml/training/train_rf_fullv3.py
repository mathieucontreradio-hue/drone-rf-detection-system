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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================
# PARAMETERS
# ============================================================
FS = 48000
N = 1024
SAMPLES_PER_CLASS = 3000

SNRS = [-15, -10, -5, 0, 5, 10, 15, 20]

class_names = {0:"Noise",1:"AM",2:"GMSK",3:"FSK"}

# ============================================================
# IMPERFECTIONS SDR (V2 REALISTIC)
# ============================================================

def add_awgn(iq, snr_db):
    sig_power = np.mean(np.abs(iq)**2)
    snr = 10**(snr_db/10)
    noise_power = sig_power / snr

    noise = np.sqrt(noise_power/2) * (
        np.random.randn(len(iq)) + 1j*np.random.randn(len(iq))
    )

    return iq + noise


def add_cfo_drift(iq):
    t = np.arange(len(iq))
    base = np.random.uniform(-0.01, 0.01)
    drift = np.cumsum(np.random.randn(len(iq))*1e-6)
    f = base + drift
    return iq * np.exp(1j*2*np.pi*f*t)


def multipath(iq):
    h0 = (np.random.randn()+1j*np.random.randn())
    h1 = 0.5*(np.random.randn()+1j*np.random.randn())
    delayed = np.roll(iq, np.random.randint(1,20))
    return h0*iq + h1*delayed


def phase_noise(iq):
    noise = np.cumsum(np.random.randn(len(iq))*0.01)
    return iq * np.exp(1j*noise)


def agc(iq):
    return iq * np.random.uniform(0.2,2.0)


def burst(iq):
    out = np.zeros_like(iq)
    s = np.random.randint(0,N//2)
    e = np.random.randint(s+100,N)
    out[s:e] = iq[s:e]
    return out

# ============================================================
# SIGNALS
# ============================================================

def gen_noise():
    return np.random.randn(N)+1j*np.random.randn(N)

def gen_am():
    t = np.arange(N)/FS
    return (1+0.7*np.sin(2*np.pi*1000*t)) * np.exp(1j*2*np.pi*5000*t)

def gen_fsk():
    bits = np.random.randint(0,2,32)
    freq = np.repeat(bits,N//32)
    freq = 2000 + freq*4000
    phase = 2*np.pi*np.cumsum(freq)/FS
    return np.exp(1j*phase)

def gen_gmsk():
    data = np.random.choice([-1,1],N)
    return np.exp(1j*np.cumsum(data*0.03))

# ============================================================
# FEATURES
# ============================================================

def extract_features(iq):

    amp = np.abs(iq)
    phase = np.unwrap(np.angle(iq))
    dphi = np.diff(phase)
    inst_freq = dphi/(2*np.pi)

    f, psd = welch(iq, fs=FS)
    psd = psd/(np.sum(psd)+1e-12)

    entropy = -np.sum(psd*np.log2(psd+1e-12))
    flatness = np.exp(np.mean(np.log(psd+1e-12)))/(np.mean(psd)+1e-12)
    cyclo = np.abs(np.mean(iq[:-1]*np.conj(iq[1:])))

    return {
        "var_amp": np.var(amp),
        "mean_dphi": np.mean(dphi),
        "var_dphi": np.var(dphi),
        "var_ifreq": np.var(inst_freq),
        "kurtosis": kurtosis(np.real(iq)),
        "entropy": entropy,
        "flatness": flatness,
        "cyclo": cyclo
    }

# ============================================================
# DATASET
# ============================================================

rows=[]

classes=[
    (gen_noise,0),
    (gen_am,1),
    (gen_gmsk,2),
    (gen_fsk,3)
]

for gen,label in classes:
    print("Class",label)
    for _ in range(SAMPLES_PER_CLASS):

        iq=gen()

        # SDR REALISM
        iq=agc(iq)
        iq=multipath(iq)
        iq=phase_noise(iq)
        iq=add_cfo_drift(iq)
        iq=burst(iq)
        iq=add_awgn(iq,np.random.choice(SNRS))

        feat=extract_features(iq)
        feat["label"]=label
        rows.append(feat)

df=pd.DataFrame(rows)
df.to_csv("dataset_v2.csv",index=False)

print("Dataset saved ✔")

# ============================================================
# PREPROCESS
# ============================================================

X=df.drop(columns=["label"])
y=df["label"]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

# ============================================================
# PCA
# ============================================================

pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)

plt.figure()

for c in sorted(y.unique()):
    idx=(y==c)
    plt.scatter(X_pca[idx,0],X_pca[idx,1],s=5,label=class_names[c])

plt.title("PCA - SDR Realistic V2")
plt.legend()
plt.grid()
plt.show()

# ============================================================
# t-SNE
# ============================================================

tsne=TSNE(n_components=2,perplexity=30,init="pca",random_state=42)
X_tsne=tsne.fit_transform(X_scaled)

plt.figure()

for c in sorted(y.unique()):
    idx=(y==c)
    plt.scatter(X_tsne[idx,0],X_tsne[idx,1],s=5,label=class_names[c])

plt.title("t-SNE - SDR Realistic V2")
plt.legend()
plt.grid()
plt.show()

# ============================================================
# TRAIN
# ============================================================

X_train,X_test,y_train,y_test=train_test_split(
    X_scaled,y,test_size=0.2,stratify=y,random_state=42
)

model=RandomForestClassifier(
    n_estimators=500,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

# ============================================================
# METRICS
# ============================================================

print("\nACCURACY:",accuracy_score(y_test,y_pred))
print("\nREPORT:\n",classification_report(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred)

plt.figure()
sns.heatmap(cm,annot=True,fmt="d",
            xticklabels=class_names.values(),
            yticklabels=class_names.values())
plt.title("Confusion Matrix")
plt.show()

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(model,"rf_model.pkl")
joblib.dump(scaler,"scaler.pkl")

print("MODEL SAVED ✔")
