from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats
from preprocess import preprocess_signal


@dataclass(frozen=True)
class PhysicsFeatureResult:
    features: np.ndarray
    summary: dict[str, float]


def get_energy_around(freqs, fft_vals, target_freq, band=2):
    idx = (freqs >= target_freq - band) & (freqs <= target_freq + band)

    band_energy = np.sum(np.abs(fft_vals[idx]) ** 2)
    total_energy = np.sum(np.abs(fft_vals) ** 2)

    if total_energy == 0:
        return 0.0

    return band_energy / total_energy


def compute_physics_features(
    signal: np.ndarray,
    envelope: np.ndarray,
    rpm: float,
    fault_freqs: dict[str, float],
    fs: float,
) -> PhysicsFeatureResult:
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    fft_mag = np.abs(np.fft.rfft(signal)) / n
    env_fft = np.abs(np.fft.rfft(envelope)) / n

    features: list[float] = []
    summary: dict[str, float] = {}
    fr = rpm / 60.0
    total_energy = np.sum(fft_mag ** 2) + 1e-10
    env_total_energy = np.sum(env_fft ** 2) + 1e-10

    for fault_name, multiplier in fault_freqs.items():
        fc = multiplier * fr
        raw_sum = 0.0
        env_sum = 0.0

        for harmonic in (1, 2, 3):
            hf = harmonic * fc
            bw = max(0.5, hf * 0.05)
            mask = (freqs >= hf - bw) & (freqs <= hf + bw)

            band_energy = np.sum(fft_mag[mask] ** 2) / total_energy
            env_energy = np.sum(env_fft[mask] ** 2) / env_total_energy

            features.extend([band_energy, env_energy])
            raw_sum += band_energy
            env_sum += env_energy

        summary[f"{fault_name}_raw_sum"] = float(raw_sum)
        summary[f"{fault_name}_env_sum"] = float(env_sum)

    rms = np.sqrt(np.mean(signal ** 2))
    kurtosis = stats.kurtosis(signal)
    crest = np.max(np.abs(signal)) / (rms + 1e-10)
    skewness = stats.skew(signal)
    env_kurtosis = stats.kurtosis(envelope)

    rms = float(np.nan_to_num(rms))
    kurtosis = float(np.nan_to_num(kurtosis))
    crest = float(np.nan_to_num(crest))
    skewness = float(np.nan_to_num(skewness))
    env_kurtosis = float(np.nan_to_num(env_kurtosis))

    features.extend([rms, kurtosis, crest, skewness, env_kurtosis])
    summary.update(
        {
            "rms": rms,
            "kurtosis": kurtosis,
            "crest": crest,
            "skewness": skewness,
            "env_kurtosis": env_kurtosis,
            "fault_energy_sum_env": float(
                summary.get("BPFI_env_sum", 0.0)
                + summary.get("BPFO_env_sum", 0.0)
                + summary.get("BSF_env_sum", 0.0)
            ),
        }
    )

    return PhysicsFeatureResult(np.array(features, dtype=np.float32), summary)


def normalize_metadata(rpm: float, bearing_type_id: int, fs: float):
    rpm_bucket = int(np.digitize(rpm, bins=[0, 500, 1000, 1500, 2000, 3000, 5000]))
    rpm_normalized = rpm / 3000.0

    return {
        "rpm_raw": float(rpm_normalized),
        "rpm_bucket": rpm_bucket,
        "bearing_type": int(bearing_type_id),
        "fs_normalized": float(fs / 1000.0),
    }


def assign_fault_label(label, envelope_fft, freqs, rpm, fault_freqs):
    if label == 0:
        return 0

    fr = rpm / 60.0
    energies = {}
    for fault_name, multiplier in fault_freqs.items():
        if fault_name == "FTF":
            continue
        fc = multiplier * fr
        bw = max(0.5, fc * 0.05)
        mask = (freqs >= fc - bw) & (freqs <= fc + bw)
        energies[fault_name] = np.sum(envelope_fft[mask] ** 2)

    dominant = max(energies, key=energies.get)
    label_map = {"BPFI": 1, "BPFO": 3, "BSF": 2}
    return label_map[dominant]


def build_dataset_features(raw_data, rpm, labels, fs, fault_mult, bearing_type_id=0):
    feature_rows = []
    metadata_rows = []
    pseudo_labels = []
    summaries = []

    for i in range(len(raw_data)):
        preprocessed = preprocess_signal(raw_data[i], fs)
        filtered = preprocessed["filtered_signal"]
        envelope = preprocessed["envelope"]
        feature_result = compute_physics_features(filtered, envelope, float(rpm[i]), fault_mult, fs)
        feature_rows.append(feature_result.features)
        summaries.append(feature_result.summary)
        metadata_rows.append(normalize_metadata(float(rpm[i]), bearing_type_id, fs))
        pseudo_labels.append(labels[i])

    return {
        "features": np.asarray(feature_rows, dtype=np.float32),
        "metadata": metadata_rows,
        "labels": np.asarray(pseudo_labels),
        "summaries": summaries,
    }
