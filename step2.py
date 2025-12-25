import os
import numpy as np
import mne

# === SETTINGS ===
EDF_PATH = "data_raw/chb01/chb01_03.edf"
OUT_NPZ = "dataset_chb01_03_windows.npz"

WINDOW_SEC = 10 # each sample = 10 seconds
STEP_SEC = 5 # overlap (move 5 sec each time)
FMIN, FMAX = 0.5, 40.0 # EEG band
BANDS = {
"delta": (0.5, 4),
"theta": (4, 8),
"alpha": (8, 13),
"beta": (13, 30),
}

def bandpower(psd, freqs, fmin, fmax):
    """Integrate PSD in [fmin, fmax] for each channel."""
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    bp = np.trapz(psd[:, idx], freqs[idx], axis=1) # (n_channels,)
    return bp

print("Loading EDF...")
raw = mne.io.read_raw_edf(EDF_PATH, preload=True, verbose=False)

# basic preprocessing
raw = raw.copy().filter(FMIN, FMAX, verbose=False)
raw = raw.copy().resample(256, verbose=False) # keep consistent sampling rate

sfreq = raw.info["sfreq"]
win_samp = int(WINDOW_SEC * sfreq)
step_samp = int(STEP_SEC * sfreq)

data = raw.get_data() # shape (n_channels, n_times)
n_ch, n_times = data.shape

print(f"Channels: {n_ch} | Samples: {n_times} | sfreq: {sfreq}")
print("Windowing...")

X = []
times = []

for start in range(0, n_times - win_samp + 1, step_samp):
    end = start + win_samp
    seg = data[:, start:end] # (n_channels, win_samp)
    # PSD per channel for this segment
    psd, freqs = mne.time_frequency.psd_array_welch(
        seg,
        sfreq=sfreq,
        fmin=FMIN,
        fmax=FMAX,
        n_fft=min(1024, seg.shape[1]),
        verbose=False
        ) # psd: (n_channels, n_freqs)
    # Feature vector: bandpower for each band and channel
    feats = []
    for _, (bmin, bmax) in BANDS.items():
        bp = bandpower(psd, freqs, bmin, bmax)
        feats.append(bp)
        
    feats = np.concatenate(feats, axis=0) # (n_channels * n_bands,)
    # log-transform for stability
    feats = np.log10(feats + 1e-12)
    X.append(feats)
    times.append(start / sfreq)

X = np.stack(X, axis=0) # (n_windows, n_features)
times = np.array(times)

# For now labels are unknown (we’ll add seizure labels in Step 4)
y = np.zeros((X.shape[0],), dtype=int)

np.savez_compressed(
    OUT_NPZ,
    X=X,
    y=y,
    times=times,
    ch_names=np.array(raw.ch_names),
    bands=np.array(list(BANDS.keys())),
    sfreq=sfreq,
    window_sec=WINDOW_SEC,
    step_sec=STEP_SEC
)

print("✅ Saved:", OUT_NPZ)
print("X shape:", X.shape, "| y shape:", y.shape)
print("Example feature vector length:", X.shape[1])
