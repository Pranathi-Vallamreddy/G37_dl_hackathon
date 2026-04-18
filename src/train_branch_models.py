from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from model import MetadataMLP, PhysicsMLP, SpectrogramCNN


class SpectrogramBranch(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.backbone = SpectrogramCNN()
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


class PhysicsBranch(nn.Module):
    def __init__(self, input_dim: int, n_classes: int = 2):
        super().__init__()
        self.backbone = PhysicsMLP(input_dim)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


class MetadataBranch(nn.Module):
    def __init__(self, n_bearing_types: int, n_rpm_buckets: int = 8, n_classes: int = 2):
        super().__init__()
        self.backbone = MetadataMLP(n_bearing_types=n_bearing_types, n_rpm_buckets=n_rpm_buckets)
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, rpm_raw, rpm_bucket, bearing_type):
        x = self.backbone(rpm_raw, rpm_bucket, bearing_type)
        return self.head(x)


class MetadataDataset(Dataset):
    def __init__(self, rpm_raw, rpm_bucket, bearing_type, labels):
        self.rpm_raw = torch.tensor(rpm_raw, dtype=torch.float32)
        self.rpm_bucket = torch.tensor(rpm_bucket, dtype=torch.long)
        self.bearing_type = torch.tensor(bearing_type, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "rpm_raw": self.rpm_raw[idx],
            "rpm_bucket": self.rpm_bucket[idx],
            "bearing_type": self.bearing_type[idx],
            "label": self.labels[idx],
        }


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    unique, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique)
    if n_classes == 2:
        # Binary: weight faulty by healthy/faulty ratio
        healthy_count = counts[unique == 0]
        faulty_count = counts[unique != 0]
        if len(healthy_count) > 0 and len(faulty_count) > 0:
            weights = np.ones(n_classes, dtype=np.float32)
            weights[1] = healthy_count[0] / max(1.0, faulty_count.sum())
        else:
            weights = np.ones(n_classes, dtype=np.float32)
    else:
        # Multi-class: inverse frequency
        total = len(labels)
        weights = total / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_model(model, optimizer, loss_fn, train_loader, val_loader, device, epochs: int = 8):
    model.to(device)
    train_losses = []
    val_accs = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if isinstance(batch, dict):
                inputs = [batch[k].to(device) for k in ["rpm_raw", "rpm_bucket", "bearing_type"]]
                logits = model(*inputs)
                labels = batch["label"].to(device)
            else:
                x, labels = batch
                logits = model(x.to(device))
                labels = labels.to(device)

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)
        val_acc = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        print(f"Epoch {epoch}: loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    return model, train_losses, val_accs


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                inputs = [batch[k].to(device) for k in ["rpm_raw", "rpm_bucket", "bearing_type"]]
                logits = model(*inputs)
                labels = batch["label"].to(device)
            else:
                x, labels = batch
                logits = model(x.to(device))
                labels = labels.to(device)

            total += labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

    return correct / max(total, 1)


def evaluate_metrics(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                inputs = [batch[k].to(device) for k in ["rpm_raw", "rpm_bucket", "bearing_type"]]
                logits = model(*inputs)
                labels = batch["label"].to(device)
            else:
                x, labels = batch
                logits = model(x.to(device))
                labels = labels.to(device)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, target_names=["Healthy", "Inner race", "Ball", "Outer race"], zero_division=0)
    accuracy = np.mean(np.array(y_true) == np.array(y_pred), dtype=np.float32)
    faulty_recall = cm[1, 1] / max(cm[1, 1] + cm[1, 0], 1) if cm.shape[0] > 1 else 0.0
    return accuracy, cm, report, faulty_recall


def load_split(path: Path):
    data = np.load(path)
    return data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_path = Path("outputs") / "dataset_split.npz"
    data = load_split(split_path)

    X_spec_train = data["X_spec_train"]
    X_spec_test = data["X_spec_test"]
    X_phys_train = data["X_phys_train"]
    X_phys_test = data["X_phys_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    rpm_raw_train = data["rpm_raw_train"]
    rpm_raw_test = data["rpm_raw_test"]
    rpm_bucket_train = data["rpm_bucket_train"]
    rpm_bucket_test = data["rpm_bucket_test"]
    bearing_type_train = data["bearing_type_train"]
    bearing_type_test = data["bearing_type_test"]

    print(f"Train shapes: spec={X_spec_train.shape}, phys={X_phys_train.shape}, meta={rpm_raw_train.shape}")
    print(f"Label balance: {np.unique(y_train, return_counts=True)}")

    n_classes = len(np.unique(y_train))
    print(f"Number of classes: {n_classes}")

    class_weights = compute_class_weights(y_train, device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Spectrogram model
    spec_model_path = Path("outputs") / "spectrogram_branch.pt"
    spec_metrics_path = Path("outputs") / "spectrogram_metrics.pkl"
    if spec_model_path.exists():
        print("Loading existing Spectrogram Branch model...")
        spec_model = SpectrogramBranch(n_classes=n_classes)
        spec_model.load_state_dict(torch.load(spec_model_path, map_location=device))
        if spec_metrics_path.exists():
            import pickle
            with open(spec_metrics_path, 'rb') as f:
                spec_train_losses, spec_val_accs = pickle.load(f)
        else:
            spec_train_losses, spec_val_accs = None, None
    else:
        spec_model = SpectrogramBranch(n_classes=n_classes)
        spec_train_loader = build_dataloader(SimpleDataset(X_spec_train, y_train), batch_size=32, shuffle=True)
        spec_test_loader = build_dataloader(SimpleDataset(X_spec_test, y_test), batch_size=32, shuffle=False)
        print("\n=== Training Spectrogram Branch ===")
        spec_model, spec_train_losses, spec_val_accs = train_model(
            spec_model,
            torch.optim.Adam(spec_model.parameters(), lr=1e-3),
            loss_fn,
            spec_train_loader,
            spec_test_loader,
            device,
            epochs=8,
        )
        torch.save(spec_model.state_dict(), spec_model_path)
        import pickle
        with open(spec_metrics_path, 'wb') as f:
            pickle.dump((spec_train_losses, spec_val_accs), f)

    # Plot Spectrogram metrics if available
    if spec_train_losses and spec_val_accs:
        epochs = list(range(1, len(spec_train_losses) + 1))
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, spec_train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Spectrogram Branch: Epoch vs Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, spec_val_accs, label='Val Accuracy', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Spectrogram Branch: Epoch vs Val Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path("outputs") / "spectrogram_metrics.png")
        plt.close()
        print("Spectrogram metrics plot saved to outputs/spectrogram_metrics.png")

    # Physics model
    phys_model_path = Path("outputs") / "physics_branch.pt"
    phys_metrics_path = Path("outputs") / "physics_metrics.pkl"
    if phys_model_path.exists():
        print("Loading existing Physics Branch model...")
        phys_model = PhysicsBranch(input_dim=X_phys_train.shape[1], n_classes=n_classes)
        phys_model.load_state_dict(torch.load(phys_model_path, map_location=device))
        if phys_metrics_path.exists():
            import pickle
            with open(phys_metrics_path, 'rb') as f:
                phys_train_losses, phys_val_accs = pickle.load(f)
        else:
            phys_train_losses, phys_val_accs = None, None
    else:
        phys_model = PhysicsBranch(input_dim=X_phys_train.shape[1], n_classes=n_classes)
        phys_train_loader = build_dataloader(SimpleDataset(X_phys_train, y_train), batch_size=32, shuffle=True)
        phys_test_loader = build_dataloader(SimpleDataset(X_phys_test, y_test), batch_size=32, shuffle=False)
        print("\n=== Training Physics Branch ===")
        phys_model, phys_train_losses, phys_val_accs = train_model(
            phys_model,
            torch.optim.Adam(phys_model.parameters(), lr=1e-3),
            loss_fn,
            phys_train_loader,
            phys_test_loader,
            device,
            epochs=8,
        )
        torch.save(phys_model.state_dict(), phys_model_path)
        import pickle
        with open(phys_metrics_path, 'wb') as f:
            pickle.dump((phys_train_losses, phys_val_accs), f)

    # Plot Physics metrics if available
    if phys_train_losses and phys_val_accs:
        epochs = list(range(1, len(phys_train_losses) + 1))
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, phys_train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Physics Branch: Epoch vs Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, phys_val_accs, label='Val Accuracy', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Physics Branch: Epoch vs Val Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path("outputs") / "physics_metrics.png")
        plt.close()
        print("Physics metrics plot saved to outputs/physics_metrics.png")

    # Metadata model
    n_bearing_types = int(bearing_type_train.max()) + 1
    n_rpm_buckets = int(rpm_bucket_train.max()) + 1
    meta_model_path = Path("outputs") / "metadata_branch.pt"
    meta_metrics_path = Path("outputs") / "metadata_metrics.pkl"
    if meta_model_path.exists():
        print("Loading existing Metadata Branch model...")
        meta_model = MetadataBranch(n_bearing_types=n_bearing_types, n_rpm_buckets=n_rpm_buckets, n_classes=n_classes)
        meta_model.load_state_dict(torch.load(meta_model_path, map_location=device))
        if meta_metrics_path.exists():
            import pickle
            with open(meta_metrics_path, 'rb') as f:
                meta_train_losses, meta_val_accs = pickle.load(f)
        else:
            meta_train_losses, meta_val_accs = None, None
    else:
        meta_model = MetadataBranch(n_bearing_types=n_bearing_types, n_rpm_buckets=n_rpm_buckets, n_classes=n_classes)
        meta_train_loader = build_dataloader(
            MetadataDataset(rpm_raw_train, rpm_bucket_train, bearing_type_train, y_train),
            batch_size=32,
            shuffle=True,
        )
        meta_test_loader = build_dataloader(
            MetadataDataset(rpm_raw_test, rpm_bucket_test, bearing_type_test, y_test),
            batch_size=32,
            shuffle=False,
        )
        print("\n=== Training Metadata Branch ===")
        meta_model, meta_train_losses, meta_val_accs = train_model(
            meta_model,
            torch.optim.Adam(meta_model.parameters(), lr=1e-3),
            loss_fn,
            meta_train_loader,
            meta_test_loader,
            device,
            epochs=8,
        )
        torch.save(meta_model.state_dict(), meta_model_path)
        import pickle
        with open(meta_metrics_path, 'wb') as f:
            pickle.dump((meta_train_losses, meta_val_accs), f)

    # Plot Metadata metrics if available
    if meta_train_losses and meta_val_accs:
        epochs = list(range(1, len(meta_train_losses) + 1))
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, meta_train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Metadata Branch: Epoch vs Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, meta_val_accs, label='Val Accuracy', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Metadata Branch: Epoch vs Val Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path("outputs") / "metadata_metrics.png")
        plt.close()
        print("Metadata metrics plot saved to outputs/metadata_metrics.png")

    print("\nAll branch models trained and saved to outputs/")

    # ─────────────────────────────────────────────────────────────────────
    # TEST EVALUATION
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("TEST EVALUATION")
    print("="*55)

    # Load test data
    X_spec_test = data["X_spec_test"]
    X_phys_test = data["X_phys_test"]
    y_test = data["y_test"]
    rpm_raw_test = data["rpm_raw_test"]
    rpm_bucket_test = data["rpm_bucket_test"]
    bearing_type_test = data["bearing_type_test"]

    # Create test loaders
    spec_test_loader = build_dataloader(SimpleDataset(X_spec_test, y_test), batch_size=32, shuffle=False)
    phys_test_loader = build_dataloader(SimpleDataset(X_phys_test, y_test), batch_size=32, shuffle=False)
    meta_test_loader = build_dataloader(
        MetadataDataset(rpm_raw_test, rpm_bucket_test, bearing_type_test, y_test),
        batch_size=32,
        shuffle=False,
    )

    # Evaluate each model
    spec_test_acc, spec_cm, spec_report, spec_recall = evaluate_metrics(spec_model, spec_test_loader, device)
    phys_test_acc, phys_cm, phys_report, phys_recall = evaluate_metrics(phys_model, phys_test_loader, device)
    meta_test_acc, meta_cm, meta_report, meta_recall = evaluate_metrics(meta_model, meta_test_loader, device)

    print(f"Spectrogram Branch Test Accuracy: {spec_test_acc:.4f}")
    print(f"Physics Branch Test Accuracy: {phys_test_acc:.4f}")
    print(f"Metadata Branch Test Accuracy: {meta_test_acc:.4f}")
    print()

    print("Spectrogram Branch Confusion Matrix:")
    print(spec_cm)
    print("Spectrogram Branch Classification Report:")
    print(spec_report)
    print(f"Spectrogram Branch Faulty Recall: {spec_recall:.4f}\n")

    print("Physics Branch Confusion Matrix:")
    print(phys_cm)
    print("Physics Branch Classification Report:")
    print(phys_report)
    print(f"Physics Branch Faulty Recall: {phys_recall:.4f}\n")

    print("Metadata Branch Confusion Matrix:")
    print(meta_cm)
    print("Metadata Branch Classification Report:")
    print(meta_report)
    print(f"Metadata Branch Faulty Recall: {meta_recall:.4f}\n")

    np.savez_compressed(
        Path("outputs") / "branch_test_metrics.npz",
        spec_cm=spec_cm,
        phys_cm=phys_cm,
        meta_cm=meta_cm,
        spec_acc=spec_test_acc,
        phys_acc=phys_test_acc,
        meta_acc=meta_test_acc,
        spec_recall=spec_recall,
        phys_recall=phys_recall,
        meta_recall=meta_recall,
    )

    with open(Path("outputs") / "branch_classification_reports.txt", "w") as f:
        f.write("Spectrogram Branch\n")
        f.write(spec_report)
        f.write("\n\nPhysics Branch\n")
        f.write(phys_report)
        f.write("\n\nMetadata Branch\n")
        f.write(meta_report)

    print("\nSaved branch test metrics to outputs/branch_test_metrics.npz and outputs/branch_classification_reports.txt")
    print("\n✓ Test evaluation complete.")


if __name__ == "__main__":
    main() 