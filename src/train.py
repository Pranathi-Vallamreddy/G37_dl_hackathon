"""
PILL — train.py  (final)
Trains BearingFaultNet with:
  - 4-class pseudo labels from prepare_dataset.py
  - PhysicsInformedLoss (CE + physics penalty)
  - OOD detector fitted on training embeddings post-training
  - Temperature scaling calibration on val set

Run from PILL/ root:
    python src/train.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

sys.path.append(str(Path(__file__).parent))

from model        import BearingFaultNet
from physics_loss import PhysicsInformedLoss
from ood_detector import OODDetector, extract_embeddings

# ── CONFIG ────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS      = 20
BATCH_SIZE  = 32
LR          = 1e-3
LAMBDA_PHYS = 0.3
N_CLASSES   = 4
VAL_FRAC    = 0.15
SPLIT_PATH  = Path("outputs/dataset_split.npz")
MODEL_OUT   = Path("outputs/bearing_fault_net_4class.pt")
OOD_OUT     = Path("outputs/ood_detector.pkl")

CLASS_NAMES = {0:"Healthy", 1:"Inner race", 2:"Ball fault", 3:"Outer race"}


# ── DATASET ───────────────────────────────────────────────────────────────
class BearingDataset(Dataset):
    """
    Matched exactly to BearingFaultNet.forward() signature:
        forward(mel_spec, physics_features, rpm_raw, rpm_bucket, bearing_type)
    """
    def __init__(self, spec, phys, rpm_raw, rpm_bucket, bearing_type, labels):
        self.spec         = torch.from_numpy(spec).float()
        self.phys         = torch.from_numpy(phys).float()
        self.rpm_raw      = torch.from_numpy(rpm_raw).float()
        self.rpm_bucket   = torch.from_numpy(rpm_bucket).long()
        self.bearing_type = torch.from_numpy(bearing_type).long()
        self.labels       = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "spectrogram"    : self.spec[idx],
            "physics_features": self.phys[idx],
            "rpm_raw"        : self.rpm_raw[idx],
            "rpm_bucket"     : self.rpm_bucket[idx],
            "bearing_type"   : self.bearing_type[idx],
            "label"          : self.labels[idx],
        }


def make_loader(split, idx, batch_size, shuffle):
    ds = BearingDataset(
        split["X_spec_train"][idx]        if "X_spec_train" in split else split["X_spec_test"],
        split["X_phys_train"][idx]        if "X_phys_train" in split else split["X_phys_test"],
        split["rpm_raw_train"][idx]       if "rpm_raw_train" in split else split["rpm_raw_test"],
        split["rpm_bucket_train"][idx]    if "rpm_bucket_train" in split else split["rpm_bucket_test"],
        split["bearing_type_train"][idx]  if "bearing_type_train" in split else split["bearing_type_test"],
        split["y_train"][idx]             if "y_train" in split else split["y_test"],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=DEVICE.type == "cuda")


def make_test_loader(split, batch_size):
    ds = BearingDataset(
        split["X_spec_test"],
        split["X_phys_test"],
        split["rpm_raw_test"],
        split["rpm_bucket_test"],
        split["bearing_type_test"],
        split["y_test"],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


# ── TEMPERATURE SCALING ───────────────────────────────────────────────────
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        opt = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=100)
        nll = nn.CrossEntropyLoss()
        def closure():
            opt.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss
        opt.step(closure)
        print(f"[Calibration] Temperature: {self.temperature.item():.4f}")


# ── TRAIN / EVAL ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    tot_loss, tot_ce, tot_pen, correct, total = 0, 0, 0, 0, 0
    for batch in loader:
        spec  = batch["spectrogram"].to(device)
        phys  = batch["physics_features"].to(device)
        rpm_r = batch["rpm_raw"].to(device)
        rpm_b = batch["rpm_bucket"].to(device)
        btype = batch["bearing_type"].to(device)
        lbls  = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(spec, phys, rpm_r, rpm_b, btype)
        loss, info = criterion(logits, lbls, phys)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot_loss += loss.item()
        tot_ce   += info["ce_loss"]
        tot_pen  += info["physics_penalty"]
        correct  += (logits.argmax(1) == lbls).sum().item()
        total    += len(lbls)

    n = len(loader)
    return {"loss": tot_loss/n, "ce": tot_ce/n, "penalty": tot_pen/n,
            "acc": correct/total}


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    tot_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_logits = [], [], []

    for batch in loader:
        spec  = batch["spectrogram"].to(device)
        phys  = batch["physics_features"].to(device)
        rpm_r = batch["rpm_raw"].to(device)
        rpm_b = batch["rpm_bucket"].to(device)
        btype = batch["bearing_type"].to(device)
        lbls  = batch["label"].to(device)

        logits = model(spec, phys, rpm_r, rpm_b, btype)
        loss, _ = criterion(logits, lbls, phys)

        tot_loss += loss.item()
        preds     = logits.argmax(1)
        correct  += (preds == lbls).sum().item()
        total    += len(lbls)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())
        all_logits.append(logits.cpu())

    return {
        "loss"  : tot_loss / len(loader),
        "acc"   : correct / total,
        "preds" : np.array(all_preds),
        "labels": np.array(all_labels),
        "logits": torch.cat(all_logits, dim=0),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"{SPLIT_PATH} not found — run prepare_dataset.py first")

    data = np.load(SPLIT_PATH)
    split = {k: data[k] for k in data.files}

    physics_input_dim = int(split["physics_input_dim"][0])
    n_bearing_types   = int(split["n_bearing_types"][0])

    print(f"Loaded split from {SPLIT_PATH}")
    print(f"  physics_input_dim : {physics_input_dim}")
    print(f"  n_bearing_types   : {n_bearing_types}")
    print(f"  X_spec_train      : {split['X_spec_train'].shape}")
    print(f"  y_train dist      : {np.unique(split['y_train'], return_counts=True)}\n")

    # val split from train
    N_train   = len(split["y_train"])
    perm      = np.random.permutation(N_train)
    n_val     = max(int(N_train * VAL_FRAC), 1)
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]

    train_loader = make_loader(split, train_idx, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(split, val_idx,   BATCH_SIZE, shuffle=False)
    test_loader  = make_test_loader(split, BATCH_SIZE)

    # model
    model = BearingFaultNet(
        n_classes         = N_CLASSES,
        physics_input_dim = physics_input_dim,
        n_bearing_types   = n_bearing_types,
    ).to(DEVICE)

    criterion = PhysicsInformedLoss(lambda_penalty=LAMBDA_PHYS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # training loop
    best_val_acc = 0.0
    print(f"Training {EPOCHS} epochs …\n")

    for epoch in range(1, EPOCHS + 1):
        tr  = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val = eval_epoch(model, val_loader,   criterion, DEVICE)
        scheduler.step()

        print(
            f"Ep {epoch:02d}/{EPOCHS}  "
            f"loss={tr['loss']:.4f}  acc={tr['acc']:.3f}  "
            f"phys_pen={tr['penalty']:.4f}  |  "
            f"val_loss={val['loss']:.4f}  val_acc={val['acc']:.3f}"
        )

        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  ↑ saved (val_acc={best_val_acc:.3f})")

    # load best
    model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE))
    print(f"\nBest val accuracy: {best_val_acc:.3f}")

    # temperature scaling
    val_info = eval_epoch(model, val_loader, criterion, DEVICE)
    scaler   = TemperatureScaler()
    scaler.fit(val_info["logits"], torch.tensor(val_info["labels"]))

    # OOD detector
    print("\nFitting OOD detector …")
    embs, lbls = extract_embeddings(model, train_loader, DEVICE)
    detector   = OODDetector(threshold_percentile=97.5)
    detector.fit(embs, lbls, n_classes=N_CLASSES)
    detector.save(str(OOD_OUT))

    # final test
    test = eval_epoch(model, test_loader, criterion, DEVICE)
    print(f"\nTest accuracy: {test['acc']:.3f}")
    print(classification_report(
        test["labels"], test["preds"],
        target_names=[CLASS_NAMES[i] for i in range(N_CLASSES)],
        zero_division=0,
    ))

    # OOD stats on test
    test_embs, _ = extract_embeddings(model, test_loader, DEVICE)
    ood_results  = detector.predict_batch(test_embs)
    n_ood = sum(r["ood"] for r in ood_results)
    print(f"OOD flagged on test: {n_ood}/{len(ood_results)} "
          f"({100*n_ood/max(len(ood_results),1):.1f}%)")

    print(f"\n✓ Done.  Model → {MODEL_OUT}  |  OOD → {OOD_OUT}")


if __name__ == "__main__":
    main()
