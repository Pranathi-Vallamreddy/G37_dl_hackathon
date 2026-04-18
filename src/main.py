from pathlib import Path

import numpy as np

from load_data import load_single_file
from pipeline import build_dataset_features, compute_physics_features
from preprocess import preprocess_signal
from visualize import plot_fft, plot_signal

DATASET_ROOT = Path("SCA bearing dataset")
train_files = sorted(DATASET_ROOT.glob("*/train.mat"))

if not train_files:
    raise FileNotFoundError(f"No train.mat files found under {DATASET_ROOT}")

FILE_PATH = None
raw_data = rpm = labels = fs = fault_mult = None

for candidate in train_files:
    try:
        candidate_raw_data, candidate_rpm, candidate_labels, candidate_fs, candidate_fault_mult = load_single_file(
            str(candidate)
        )
    except KeyError:
        continue

    if any(candidate_labels == -1):
        FILE_PATH = str(candidate)
        raw_data = candidate_raw_data
        rpm = candidate_rpm
        labels = candidate_labels
        fs = candidate_fs
        fault_mult = candidate_fault_mult
        break

if FILE_PATH is None:
    raise ValueError("No train.mat file with a faulty sample was found after filtering zero-RPM rows")

print("Using file:", FILE_PATH)

print("\n--- DATA SUMMARY ---")
print("Shape:", raw_data.shape)
print("Sampling rate:", fs)
print("RPM range:", rpm.min(), "-", rpm.max())
print("Labels:", set(labels))

print("\nFault Multipliers:")
for k, v in fault_mult.items():
    print(k, ":", v)

print("\nFirst sample:")
print("Length:", len(raw_data[0]))
print("RPM:", rpm[0])
print("Label:", labels[0])

fault_idx = None
for i in range(len(labels)):
    if labels[i] == -1:
        fault_idx = i
        break

print("First faulty index:", fault_idx)

fault_signal = raw_data[fault_idx]
fault_rpm = rpm[fault_idx]
fault_label = labels[fault_idx]
fault_preprocessed = preprocess_signal(fault_signal, fs)
fault_filtered = fault_preprocessed["filtered_signal"]
fault_envelope = fault_preprocessed["envelope"]
fault_mel = fault_preprocessed["mel_spectrogram"]

print("Mel-spectrogram shape (Branch A input):", fault_mel.shape)

print("\nPlotting faulty sample...")

plot_signal(
    fault_filtered,
    fs,
    title=f"Faulty Sample (Filtered) | RPM={fault_rpm} | Label={fault_label}",
    filename="fault_sample.png",
)

print("\nPlotting faulty sample FFT...")

plot_fft(
    fault_filtered,
    fs,
    fault_rpm,
    fault_mult,
    title=f"Faulty Sample FFT | RPM={fault_rpm} | Label={fault_label}",
    filename="fft_fault_overlay.png",
)

fault_feature_result = compute_physics_features(
    fault_filtered,
    fault_envelope,
    fault_rpm,
    fault_mult,
    fs,
)

print("\nPhysics Features:")
print("BPFI raw sum:", fault_feature_result.summary["BPFI_raw_sum"])
print("BPFI env sum:", fault_feature_result.summary["BPFI_env_sum"])
print("BPFO raw sum:", fault_feature_result.summary["BPFO_raw_sum"])
print("BPFO env sum:", fault_feature_result.summary["BPFO_env_sum"])
print("BSF raw sum :", fault_feature_result.summary["BSF_raw_sum"])
print("BSF env sum :", fault_feature_result.summary["BSF_env_sum"])
print("RMS:", fault_feature_result.summary["rms"])
print("Kurtosis:", fault_feature_result.summary["kurtosis"])
print("Crest factor:", fault_feature_result.summary["crest"])
print("Envelope kurtosis:", fault_feature_result.summary["env_kurtosis"])

print("\nComparing features (first 10 samples):")

feature_names = [
    "BPFI_env",
    "BPFO_env",
    "BSF_env",
    "RMS",
    "Kurtosis",
    "Crest",
    "EnvKurtosis",
]

for i in range(10):
    sample_signal = raw_data[i]
    sample_rpm = rpm[i]
    sample_preprocessed = preprocess_signal(sample_signal, fs)
    sample_filtered = sample_preprocessed["filtered_signal"]
    sample_envelope = sample_preprocessed["envelope"]
    sample_result = compute_physics_features(sample_filtered, sample_envelope, sample_rpm, fault_mult, fs)

    print(f"\nSample {i} | Label={labels[i]}")
    print(
        "BPFI_env={:.4f}, BPFO_env={:.4f}, BSF_env={:.4f}, RMS={:.4f}, Kurtosis={:.4f}, Crest={:.4f}, EnvKurt={:.4f}".format(
            sample_result.summary["BPFI_env_sum"],
            sample_result.summary["BPFO_env_sum"],
            sample_result.summary["BSF_env_sum"],
            sample_result.summary["rms"],
            sample_result.summary["kurtosis"],
            sample_result.summary["crest"],
            sample_result.summary["env_kurtosis"],
        )
    )

print("\nRunning feature-batch comparison demo...")
bundle = build_dataset_features(raw_data, rpm, labels, fs, fault_mult)
labels_list = bundle["labels"]
summaries = bundle["summaries"]

comparison_features = np.array(
    [
        [
            summary["BPFI_env_sum"],
            summary["BPFO_env_sum"],
            summary["BSF_env_sum"],
            summary["rms"],
            summary["kurtosis"],
            summary["crest"],
            summary["env_kurtosis"],
        ]
        for summary in summaries
    ],
    dtype=np.float32,
)

healthy = comparison_features[labels_list == 0]
faulty = comparison_features[labels_list == -1]

print("\n=== Feature Means ===")
print("Healthy mean:", healthy.mean(axis=0))
print("Faulty mean :", faulty.mean(axis=0))
print("Feature order: [BPFI_env, BPFO_env, BSF_env, RMS, Kurtosis, Crest, EnvKurtosis]")

print("\nRunning FFT sanity test...")

fs = 640
t = np.arange(0, 2, 1 / fs)
signal = np.sin(2 * np.pi * 50 * t)
rpm = 60
fault_mult = {
    "BPFI": 1,
    "BPFO": 1,
    "BSF": 1,
    "FTF": 1,
}

plot_fft(signal, fs, rpm, fault_mult, title="FFT Test - 50 Hz Signal", filename="fft_test_50hz.png")