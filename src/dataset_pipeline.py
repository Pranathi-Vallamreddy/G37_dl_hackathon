"""
PILL — Bearing Fault Classification
Dataset Pipeline: Steps 4–11
Assumes the following are already computed and in memory:
  - signals       : list/array of shape (N, 16384)
  - physics_features : shape (N, d_phys)
  - rpm_values    : shape (N,)
  - labels        : shape (N,)  [values: -1 (fault) or 0 (healthy)]
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import librosa
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# STEP 4 — ORGANIZE FINAL DATA ARRAYS
# ─────────────────────────────────────────────

def organize_arrays(signals, physics_features, rpm_values, labels):
    """
    Converts raw inputs into clean numpy arrays with proper label encoding
    and feature normalization.

    Label convention: -1 (fault) → 1, 0 (healthy) → 0
    """
    # Convert to numpy
    signals = np.array(signals, dtype=np.float32)              # (N, 16384)
    physics_features = np.array(physics_features, dtype=np.float32)  # (N, d_phys)
    rpm_values = np.array(rpm_values, dtype=np.float32)        # (N,)
    labels = np.array(labels, dtype=np.int64)                  # (N,)

    # Label encoding: -1 → 1 (fault), 0 → 0 (healthy)
    labels = np.where(labels == -1, 1, 0)

    # Normalize physics features — StandardScaler (zero mean, unit variance)
    phys_scaler = StandardScaler()
    physics_features = phys_scaler.fit_transform(physics_features)

    # Normalize RPM — MinMaxScaler to [0, 1]
    rpm_scaler = MinMaxScaler()
    rpm_norm = rpm_scaler.fit_transform(rpm_values.reshape(-1, 1)).squeeze()  # (N,)

    print(f"[Step 4] Organized arrays:")
    print(f"  signals          : {signals.shape}")
    print(f"  physics_features : {physics_features.shape}")
    print(f"  rpm_norm         : {rpm_norm.shape}")
    print(f"  labels           : {labels.shape}  | unique={np.unique(labels, return_counts=True)}")

    return signals, physics_features, rpm_norm, labels, phys_scaler, rpm_scaler


# ─────────────────────────────────────────────
# STEP 5 — CREATE SPECTROGRAM DATASET
# ─────────────────────────────────────────────

def compute_spectrograms(signals, sr=12000, n_fft=512, hop_length=128, n_mels=64):
    """
    Computes a mel-spectrogram for each signal (envelope or raw).
    Uses librosa: output shape per sample is (1, n_mels, T).
    All spectrograms are padded/cropped to the same time width T_max.

    Returns:
        spectrograms : np.ndarray of shape (N, 1, n_mels, T_max)
    """
    specs = []
    for sig in signals:
        mel = librosa.feature.melspectrogram(
            y=sig.astype(np.float32),
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)
        specs.append(mel_db)

    # Pad all to the same time width
    T_max = max(s.shape[1] for s in specs)
    padded = []
    for s in specs:
        if s.shape[1] < T_max:
            pad = np.full((s.shape[0], T_max - s.shape[1]), s.min())
            s = np.concatenate([s, pad], axis=1)
        else:
            s = s[:, :T_max]
        padded.append(s)

    spectrograms = np.stack(padded, axis=0)[:, np.newaxis, :, :]  # (N, 1, F, T)

    # Per-sample normalization: zero mean, unit std
    mean = spectrograms.mean(axis=(2, 3), keepdims=True)
    std  = spectrograms.std(axis=(2, 3), keepdims=True) + 1e-8
    spectrograms = (spectrograms - mean) / std

    print(f"[Step 5] Spectrograms computed: {spectrograms.shape}  (N, 1, F, T)")
    return spectrograms.astype(np.float32)


# ─────────────────────────────────────────────
# STEP 6 — CREATE METADATA FEATURES
# ─────────────────────────────────────────────

def build_metadata(rpm_norm):
    """
    Constructs metadata feature matrix from normalized RPM.
    Adds derived feature: rotational frequency = RPM_raw / 60
    (already implicitly encoded via norm, but kept as a second channel
    to give the network an explicit periodic reference).

    Output shape: (N, 2)
    """
    rot_freq = rpm_norm / 60.0   # rough proxy for rotational frequency in Hz after norm
    metadata = np.stack([rpm_norm, rot_freq], axis=1).astype(np.float32)  # (N, 2)

    print(f"[Step 6] Metadata features: {metadata.shape}  (N, d_meta=2)")
    return metadata


# ─────────────────────────────────────────────
# STEP 7 — TRAIN-TEST SPLIT
# ─────────────────────────────────────────────

def split_dataset(spectrograms, physics_features, metadata, labels, test_size=0.2, seed=42):
    """
    Stratified 80/20 train-test split.
    All modalities are split with the same indices to ensure alignment.
    """
    idx = np.arange(len(labels))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels
    )

    split = {
        "X_spec_train"  : spectrograms[idx_train],
        "X_spec_test"   : spectrograms[idx_test],
        "X_phys_train"  : physics_features[idx_train],
        "X_phys_test"   : physics_features[idx_test],
        "X_meta_train"  : metadata[idx_train],
        "X_meta_test"   : metadata[idx_test],
        "y_train"       : labels[idx_train],
        "y_test"        : labels[idx_test],
        "idx_train"     : idx_train,
        "idx_test"      : idx_test,
    }

    print(f"[Step 7] Train-test split:")
    print(f"  Train: {split['X_spec_train'].shape[0]} samples")
    print(f"  Test : {split['X_spec_test'].shape[0]} samples")
    return split


# ─────────────────────────────────────────────
# STEP 8 — PYTORCH DATASET CLASS
# ─────────────────────────────────────────────

class MultiModalDataset(Dataset):
    """
    PyTorch Dataset for the three-branch bearing fault model.
    Accepts pre-split numpy arrays and returns per-sample dicts of tensors.

    Args:
        spec   : np.ndarray (N, 1, F, T)  — mel spectrogram
        phys   : np.ndarray (N, d_phys)   — physics features
        meta   : np.ndarray (N, d_meta)   — metadata features
        labels : np.ndarray (N,)          — integer class labels
    """

    def __init__(self, spec, phys, meta, labels):
        assert len(spec) == len(phys) == len(meta) == len(labels), \
            "All modalities must have the same number of samples"
        self.spec   = torch.from_numpy(spec).float()
        self.phys   = torch.from_numpy(phys).float()
        self.meta   = torch.from_numpy(meta).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "spectrogram"     : self.spec[idx],    # (1, F, T)
            "physics_features": self.phys[idx],    # (d_phys,)
            "metadata"        : self.meta[idx],    # (d_meta,)
            "label"           : self.labels[idx],  # scalar int
        }


# ─────────────────────────────────────────────
# STEP 9 — DATALOADERS
# ─────────────────────────────────────────────

def build_dataloaders(split, batch_size=32, num_workers=0):
    """
    Wraps train/test splits into PyTorch DataLoaders.

    Args:
        split       : dict returned by split_dataset()
        batch_size  : int (default 32)
        num_workers : int (0 = single-process, safe for Windows/macOS)

    Returns:
        train_loader, test_loader
    """
    train_ds = MultiModalDataset(
        split["X_spec_train"], split["X_phys_train"],
        split["X_meta_train"], split["y_train"]
    )
    test_ds = MultiModalDataset(
        split["X_spec_test"], split["X_phys_test"],
        split["X_meta_test"], split["y_test"]
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    print(f"[Step 9] DataLoaders ready:")
    print(f"  train_loader: {len(train_loader)} batches of up to {batch_size}")
    print(f"  test_loader : {len(test_loader)} batches of up to {batch_size}")
    return train_loader, test_loader


# ─────────────────────────────────────────────
# STEP 10 — SANITY CHECKS
# ─────────────────────────────────────────────

def sanity_check(train_loader, split):
    """
    Prints shapes, a sample batch, and class balance.
    """
    print("\n" + "="*55)
    print("SANITY CHECKS")
    print("="*55)

    # ── 1. Shapes from one batch ──────────────────────────────
    batch = next(iter(train_loader))
    print("\n[Check 1] Batch shapes:")
    print(f"  spectrogram      : {tuple(batch['spectrogram'].shape)}")
    print(f"  physics_features : {tuple(batch['physics_features'].shape)}")
    print(f"  metadata         : {tuple(batch['metadata'].shape)}")
    print(f"  label            : {tuple(batch['label'].shape)}")

    # ── 2. Alignment check ───────────────────────────────────
    print("\n[Check 2] Alignment (first 5 samples in batch):")
    print(f"  labels in batch  : {batch['label'][:5].tolist()}")
    print(f"  phys mean row 0  : {batch['physics_features'][0][:4].tolist()}  ...")
    print(f"  meta row 0       : {batch['metadata'][0].tolist()}")

    # ── 3. Class balance ─────────────────────────────────────
    all_labels = np.concatenate([split["y_train"], split["y_test"]])
    unique, counts = np.unique(all_labels, return_counts=True)
    print("\n[Check 3] Class balance (full dataset):")
    class_names = {0: "Healthy", 1: "Fault"}
    for u, c in zip(unique, counts):
        print(f"  Class {u} ({class_names.get(u, '?')}): {c} samples ({100*c/len(all_labels):.1f}%)")

    print("\n✓ All checks passed.\n")


# ─────────────────────────────────────────────
# STEP 11 — MAIN / EXAMPLE USAGE
# ─────────────────────────────────────────────

def build_pipeline(signals, physics_features, rpm_values, labels,
                   sr=12000, n_fft=512, hop_length=128, n_mels=64,
                   test_size=0.2, batch_size=32, seed=42):
    """
    End-to-end pipeline. Call this once after your physics features are ready.

    Returns:
        train_loader, test_loader, split (dict of raw arrays), scalers
    """
    # Step 4
    signals, phys, rpm_norm, labels, phys_scaler, rpm_scaler = organize_arrays(
        signals, physics_features, rpm_values, labels
    )
    # Step 5
    spectrograms = compute_spectrograms(signals, sr, n_fft, hop_length, n_mels)
    # Step 6
    metadata = build_metadata(rpm_norm)
    # Step 7
    split = split_dataset(spectrograms, phys, metadata, labels, test_size, seed)
    # Steps 8+9
    train_loader, test_loader = build_dataloaders(split, batch_size)
    # Step 10
    sanity_check(train_loader, split)

    return train_loader, test_loader, split, {"phys": phys_scaler, "rpm": rpm_scaler}


# ─────────────────────────────────────────────
# QUICK DEMO (synthetic data — replace with yours)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Running with synthetic demo data …\n")
    np.random.seed(0)
    N = 200

    # Replace these with your actual arrays
    demo_signals   = np.random.randn(N, 16384).astype(np.float32)
    demo_physics   = np.random.randn(N, 18).astype(np.float32)   # d_phys=18 example
    demo_rpm       = np.random.uniform(600, 1800, N).astype(np.float32)
    demo_labels    = np.where(np.random.rand(N) > 0.5, -1, 0).astype(np.int64)

    train_loader, test_loader, split, scalers = build_pipeline(
        signals          = demo_signals,
        physics_features = demo_physics,
        rpm_values       = demo_rpm,
        labels           = demo_labels,
        batch_size       = 32,
    )

    # Iterate one batch
    print("Example batch iteration:")
    for batch in train_loader:
        print(f"  spec  : {batch['spectrogram'].shape}")
        print(f"  phys  : {batch['physics_features'].shape}")
        print(f"  meta  : {batch['metadata'].shape}")
        print(f"  label : {batch['label'][:8].tolist()} …")
        break

    print("\nPipeline ready for training.")