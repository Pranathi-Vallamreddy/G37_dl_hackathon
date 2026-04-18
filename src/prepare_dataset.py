"""
PILL — prepare_dataset.py  (updated)
Walks ALL train.mat files, assigns 4-class pseudo-labels using physics,
builds the full dataset split and saves to outputs/dataset_split.npz.

Label convention:
    0 = Healthy
    1 = Inner race fault  (BPFI dominant)
    2 = Ball fault        (BSF dominant)
    3 = Outer race fault  (BPFO dominant)

Run from PILL/ root:
    python src/prepare_dataset.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent))

from load_data  import load_single_file
from pipeline   import compute_physics_features, assign_fault_label
from preprocess import preprocess_signal
from dataset_pipeline import compute_spectrograms

# ── CONFIG ───────────────────────────────────────────────────────────────
DATASET_ROOT = Path("SCA bearing dataset")
TEST_SIZE    = 0.2
SEED         = 42

# ── COLLECT ───────────────────────────────────────────────────────────────
train_files = sorted(DATASET_ROOT.glob("*/train.mat"))
if not train_files:
    raise FileNotFoundError(f"No train.mat files found under {DATASET_ROOT}")

print(f"Found {len(train_files)} train.mat files\n")

all_signals      = []
all_physics      = []
all_rpm_raw      = []
all_rpm_bucket   = []
all_bearing_type = []
all_labels       = []

bearing_type_map = {}   # folder name → int id
skipped = 0
actual_fs = 512         # updated from first successful file

for mat_file in train_files:
    folder = mat_file.parent.name
    if folder not in bearing_type_map:
        bearing_type_map[folder] = len(bearing_type_map)
    b_type = bearing_type_map[folder]

    try:
        raw_data, rpm_arr, labels_arr, fs, fault_mult = load_single_file(str(mat_file))
    except Exception as e:
        print(f"  [SKIP] {mat_file}: {e}")
        skipped += 1
        continue

    actual_fs = int(fs)
    print(f"  {mat_file}  —  {len(raw_data)} samples  fs={fs}")

    for i in range(len(raw_data)):
        signal    = raw_data[i]
        rpm_val   = float(rpm_arr[i])
        bin_label = int(labels_arr[i])   # 0 or -1

        # preprocess
        prep     = preprocess_signal(signal, fs)
        filtered = prep["filtered_signal"]
        envelope = prep["envelope"]

        # physics features → (29,) vector
        result = compute_physics_features(
            filtered, envelope, rpm_val, fault_mult, fs
        )

        # pseudo-label: use assign_fault_label from pipeline.py
        n       = len(envelope)
        freqs   = np.fft.rfftfreq(n, d=1.0 / fs)
        env_fft = np.abs(np.fft.rfft(envelope)) / n
        pseudo_label = assign_fault_label(
            bin_label, env_fft, freqs, rpm_val, fault_mult
        )
        # 0=healthy, 1=inner race, 2=ball, 3=outer race

        # rpm metadata
        rpm_norm   = float(np.clip(rpm_val / 3000.0, 0.0, 1.0))
        rpm_bucket = int(np.clip(
            np.digitize(rpm_val, bins=[0, 500, 1000, 1500, 2000, 3000, 5000]),
            0, 7
        ))

        all_signals.append(envelope.astype(np.float32))
        all_physics.append(result.features)
        all_rpm_raw.append(np.float32(rpm_norm))
        all_rpm_bucket.append(np.int64(rpm_bucket))
        all_bearing_type.append(np.int64(b_type))
        all_labels.append(np.int64(pseudo_label))

print(f"\nTotal: {len(all_signals)} samples  |  skipped files: {skipped}")
print(f"Bearing types: {bearing_type_map}")

# ── PAD SIGNALS TO UNIFORM LENGTH ────────────────────────────────────────
max_len     = max(s.shape[0] for s in all_signals)
signals_arr = np.zeros((len(all_signals), max_len), dtype=np.float32)
for i, s in enumerate(all_signals):
    signals_arr[i, :len(s)] = s

physics_arr      = np.stack(all_physics).astype(np.float32)
rpm_raw_arr      = np.array(all_rpm_raw,      dtype=np.float32)
rpm_bucket_arr   = np.array(all_rpm_bucket,   dtype=np.int64)
bearing_type_arr = np.array(all_bearing_type, dtype=np.int64)
labels_arr       = np.array(all_labels,       dtype=np.int64)

# label distribution
label_names = {0:"Healthy", 1:"Inner race", 2:"Ball", 3:"Outer race"}
print("\nLabel distribution:")
for u, c in zip(*np.unique(labels_arr, return_counts=True)):
    print(f"  Class {u} ({label_names[u]}): {c}")

# ── SPECTROGRAMS ──────────────────────────────────────────────────────────
print("\nComputing spectrograms …")
spectrograms = compute_spectrograms(
    signals_arr, sr=actual_fs, n_fft=256, hop_length=64, n_mels=64
)   # (N, 1, 64, T)

# ── NORMALISE PHYSICS ─────────────────────────────────────────────────────
print("Normalising physics features …")
phys_scaler  = StandardScaler()
physics_norm = phys_scaler.fit_transform(physics_arr).astype(np.float32)

# ── STRATIFIED SPLIT ──────────────────────────────────────────────────────
idx = np.arange(len(labels_arr))
idx_train, idx_test = train_test_split(
    idx, test_size=TEST_SIZE, random_state=SEED, stratify=labels_arr
)

split = {
    # spectrograms
    "X_spec_train"       : spectrograms[idx_train],
    "X_spec_test"        : spectrograms[idx_test],
    # physics
    "X_phys_train"       : physics_norm[idx_train],
    "X_phys_test"        : physics_norm[idx_test],
    # metadata — all three components stored separately
    "rpm_raw_train"      : rpm_raw_arr[idx_train],
    "rpm_raw_test"       : rpm_raw_arr[idx_test],
    "rpm_bucket_train"   : rpm_bucket_arr[idx_train],
    "rpm_bucket_test"    : rpm_bucket_arr[idx_test],
    "bearing_type_train" : bearing_type_arr[idx_train],
    "bearing_type_test"  : bearing_type_arr[idx_test],
    # labels
    "y_train"            : labels_arr[idx_train],
    "y_test"             : labels_arr[idx_test],
    # model config
    "n_bearing_types"    : np.array([len(bearing_type_map)]),
    "physics_input_dim"  : np.array([physics_arr.shape[1]]),
}

OUT = Path("outputs/dataset_split.npz")
OUT.parent.mkdir(exist_ok=True)
np.savez_compressed(OUT, **split)

print(f"\n✓ Saved → {OUT}")
print(f"  spec   : {split['X_spec_train'].shape}")
print(f"  phys   : {split['X_phys_train'].shape}  (input_dim={physics_arr.shape[1]})")
print(f"  n_bearing_types : {len(bearing_type_map)}")
print(f"  y_train: {np.unique(split['y_train'], return_counts=True)}")
print(f"  y_test : {np.unique(split['y_test'],  return_counts=True)}")
