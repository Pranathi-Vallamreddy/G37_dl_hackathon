from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from load_data import load_single_file
from preprocess import preprocess_signal
from pipeline import assign_fault_label, compute_physics_features


@dataclass(frozen=True)
class SampleItem:
    mel_spec: np.ndarray
    physics_features: np.ndarray
    rpm_raw: float
    rpm_bucket: int
    bearing_type: int
    label: int
    fault_energy_sum_env: float
    summary: dict[str, float]


class BearingFaultDataset(Dataset):
    def __init__(self, dataset_root: str | Path = "SCA bearing dataset", label_mode: str = "binary"):
        self.dataset_root = Path(dataset_root)
        self.label_mode = label_mode
        self.items: list[SampleItem] = []
        self.bearing_type_map: dict[str, int] = {}
        self.max_mel_width = 0
        self._build()
        self.max_mel_width = max(item.mel_spec.shape[1] for item in self.items)

    def _build(self) -> None:
        train_files = sorted(self.dataset_root.glob("*/train.mat"))
        if not train_files:
            raise FileNotFoundError(f"No train.mat files found under {self.dataset_root}")

        bearing_names = sorted({path.parent.name for path in train_files})
        self.bearing_type_map = {name: index for index, name in enumerate(bearing_names)}

        for path in train_files:
            try:
                raw_data, rpm, labels, fs, fault_mult = load_single_file(str(path))
            except KeyError:
                continue

            bearing_type = self.bearing_type_map[path.parent.name]
            for index in range(len(raw_data)):
                raw_signal = raw_data[index]
                sample_rpm = float(rpm[index])
                sample_label = int(labels[index])

                preprocessed = preprocess_signal(raw_signal, fs)
                filtered = preprocessed["filtered_signal"]
                envelope = preprocessed["envelope"]
                mel_db = preprocessed["mel_spectrogram"]
                physics_result = compute_physics_features(filtered, envelope, sample_rpm, fault_mult, fs)
                freqs = np.fft.rfftfreq(len(filtered), d=1 / fs)
                envelope_fft = preprocessed["envelope_fft"]

                if self.label_mode == "pseudo4":
                    assigned_label = assign_fault_label(sample_label, envelope_fft, freqs, sample_rpm, fault_mult)
                else:
                    assigned_label = 0 if sample_label == 0 else 1

                rpm_bucket = int(np.digitize(sample_rpm, bins=[0, 500, 1000, 1500, 2000, 3000, 5000]))
                rpm_raw = float(sample_rpm / 3000.0)

                self.items.append(
                    SampleItem(
                        mel_spec=mel_db.astype(np.float32),
                        physics_features=physics_result.features.astype(np.float32),
                        rpm_raw=rpm_raw,
                        rpm_bucket=rpm_bucket,
                        bearing_type=bearing_type,
                        label=assigned_label,
                        fault_energy_sum_env=float(physics_result.summary["fault_energy_sum_env"]),
                        summary=physics_result.summary,
                    )
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        mel_spec = item.mel_spec
        if mel_spec.shape[1] < self.max_mel_width:
            pad_width = self.max_mel_width - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode="constant")
        return {
            "mel_spec": torch.from_numpy(mel_spec).unsqueeze(0),
            "physics_features": torch.from_numpy(item.physics_features),
            "rpm_raw": torch.tensor(item.rpm_raw, dtype=torch.float32),
            "rpm_bucket": torch.tensor(item.rpm_bucket, dtype=torch.long),
            "bearing_type": torch.tensor(item.bearing_type, dtype=torch.long),
            "label": torch.tensor(item.label, dtype=torch.long),
            "fault_energy_sum_env": torch.tensor(item.fault_energy_sum_env, dtype=torch.float32),
        }

    @property
    def physics_input_dim(self) -> int:
        if not self.items:
            raise ValueError("Dataset is empty")
        return int(self.items[0].physics_features.shape[0])

    @property
    def n_bearing_types(self) -> int:
        return len(self.bearing_type_map)
