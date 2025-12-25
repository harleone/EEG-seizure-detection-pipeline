import numpy as np

# Input from Step 2 (features + window start times)
IN_NPZ = "dataset_chb01_03_windows.npz"

# Output (same data + real y labels)
OUT_NPZ = "dataset_chb01_03_windows_LABELED.npz"

# ====== IMPORTANT: put seizure times here (seconds from start of recording) ======
# Example format: [(start_sec, end_sec), (start_sec, end_sec), ...]
SEIZURES = [
    (300, 360),
# (start, end),
# e.g. (2996, 3036),
]

# How to label a window:
# Window is [t, t + WINDOW_SEC]. If it overlaps seizure interval => label 1
def overlaps(a_start, a_end, b_start, b_end):
    return (a_start < b_end) and (b_start < a_end)

def main():
    d = np.load(IN_NPZ, allow_pickle=True)
    X = d["X"]
    times = d["times"] # window start times in seconds (from Step 2)
    window_sec = float(d["window_sec"])
    y = np.zeros(len(times), dtype=int)
    
    
    
    for i, t0 in enumerate(times):
        t1 = t0 + window_sec
        for (s0, s1) in SEIZURES:
            if overlaps(t0, t1, s0, s1):
                y[i] = 1
                

# Save everything back + new labels
    np.savez_compressed(
        OUT_NPZ,
        X=X,
        y=y,
        times=times,
        
        window_sec=window_sec,
        step_sec=float(d["step_sec"])
)

    print("✅ Saved:", OUT_NPZ)
    
    print("Positive seizure windows:", y.sum(), "/", len(y))


if __name__ == "__main__":
   main()
