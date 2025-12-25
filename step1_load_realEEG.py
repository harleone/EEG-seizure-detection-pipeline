import mne
import matplotlib.pyplot as plt

edf_path = "data_raw/chb01/chb01_03.edf"

print("Loading EEG file...")
raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

print("EEG loaded successfully!")
print("Number of channels:", len(raw.ch_names))
print("Sampling frequency:", raw.info["sfreq"])
print("Recording duration (seconds):", raw.n_times / raw.info["sfreq"])

raw.plot(
    duration = 10,
    n_channels = 10,
    
    title = "raw EEG Signal (10 seconds)"
)

plt.show()
