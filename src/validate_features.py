import numpy as np
import scipy.stats as stats
from load_data import load_single_file
from preprocess import preprocess_signal


# -----------------------------
# Energy extraction helper
# -----------------------------
def energy_band(freqs, fft_vals, center, band=0.5):
    mask = (freqs >= center - band) & (freqs <= center + band)
    band_energy = np.sum(np.abs(fft_vals[mask])**2)
    total_energy = np.sum(np.abs(fft_vals)**2) + 1e-10
    return band_energy / total_energy


# -----------------------------
# Feature extraction (ENVELOPE-based)
# -----------------------------
def extract_features(signal, fs, rpm, fault_mult):
    processed = preprocess_signal(signal, fs)

    filtered = processed["filtered_signal"]
    envelope = processed["envelope"]

    # FFT of envelope (IMPORTANT)
    fft_vals = np.fft.rfft(envelope)
    freqs = np.fft.rfftfreq(len(envelope), d=1/fs)

    fr = rpm / 60.0

    features = []

    for key in ["BPFI", "BPFO", "BSF"]:
        fc = fr * fault_mult[key]

        # include harmonics
        energy = (
            energy_band(freqs, fft_vals, fc) +
            energy_band(freqs, fft_vals, 2*fc) +
            energy_band(freqs, fft_vals, 3*fc)
        )

        features.append(energy)

    # -----------------------------
    # NEW: statistical features
    # -----------------------------
    kurt = stats.kurtosis(filtered)
    env_kurt = stats.kurtosis(envelope)

    stat_features = [kurt, env_kurt]

    return np.array(features + stat_features)


# -----------------------------
# MAIN VALIDATION
# -----------------------------
def main():
    FILE_PATH = "SCA bearing dataset/3/train.mat"

    raw_data, rpm, labels, fs, fault_mult = load_single_file(FILE_PATH)

    features = []
    labels_list = []

    for i in range(len(raw_data)):
        if rpm[i] <= 0:
            continue  # skip invalid

        f = extract_features(raw_data[i], fs, rpm[i], fault_mult)

        features.append(f)
        labels_list.append(labels[i])

    features = np.array(features)
    labels_list = np.array(labels_list)

    # -----------------------------
    # Split groups
    # -----------------------------
    healthy = features[labels_list == 0]
    faulty = features[labels_list == -1]

    print("\n=== VALIDATION RESULTS ===")

    print("\nHealthy mean:", healthy.mean(axis=0))
    print("Faulty mean :", faulty.mean(axis=0))

    healthy_mean = healthy.mean(axis=0)
    faulty_mean = faulty.mean(axis=0)

    print("\nFeature importance intuition:")
    print("Kurtosis diff:", faulty_mean[3] - healthy_mean[3])
    print("Env Kurt diff:", faulty_mean[4] - healthy_mean[4])

    print("\nHealthy std :", healthy.std(axis=0))
    print("Faulty std  :", faulty.std(axis=0))


if __name__ == "__main__":
    main()