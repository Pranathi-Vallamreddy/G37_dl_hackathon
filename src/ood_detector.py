"""
PILL — ood_detector.py
Mahalanobis-distance OOD detection on the 128-d fused embedding.

Fits class-conditional Gaussians (tied covariance) on training embeddings.
At inference: min distance across classes > threshold → abstain (OOD).

Usage:
    from ood_detector import OODDetector, extract_embeddings

    # after training
    embs, lbls = extract_embeddings(model, train_loader, device)
    detector   = OODDetector()
    detector.fit(embs, lbls, n_classes=4)
    detector.save("outputs/ood_detector.pkl")

    # at inference
    result = detector.predict(embedding)   # dict with ood, class, distance
"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class OODDetector:
    def __init__(self, threshold_percentile: float = 97.5):
        self.threshold_percentile = threshold_percentile
        self.class_means_    = None
        self.shared_cov_inv_ = None
        self.threshold_      = None
        self.n_classes_      = None
        self.is_fitted_      = False

    def fit(self, embeddings: np.ndarray, labels: np.ndarray, n_classes: int):
        self.n_classes_ = n_classes
        d = embeddings.shape[1]

        # class-conditional means
        self.class_means_ = np.zeros((n_classes, d), dtype=np.float64)
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() > 0:
                self.class_means_[c] = embeddings[mask].mean(axis=0)
            else:
                print(f"  [OOD] Warning: class {c} has no training samples")

        # tied (shared) covariance
        centred = np.zeros_like(embeddings, dtype=np.float64)
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() > 0:
                centred[mask] = embeddings[mask] - self.class_means_[c]

        cov  = (centred.T @ centred) / max(len(embeddings) - n_classes, 1)
        cov += np.eye(d) * 1e-5   # regularise
        self.shared_cov_inv_ = np.linalg.inv(cov)

        # set threshold from training distribution
        train_dists    = self._min_distances(embeddings)
        self.threshold_ = float(np.percentile(train_dists, self.threshold_percentile))
        self.is_fitted_ = True

        print(f"[OOD] Fitted: {len(embeddings)} samples, {n_classes} classes")
        print(f"  threshold (p{self.threshold_percentile:.0f}): {self.threshold_:.4f}")
        return self

    def _min_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Vectorised minimum Mahalanobis distance across classes. Shape: (N,)"""
        dists = np.zeros((len(embeddings), self.n_classes_), dtype=np.float64)
        for c in range(self.n_classes_):
            delta    = embeddings - self.class_means_[c]
            dists[:, c] = np.sqrt(
                np.einsum("nd,dd,nd->n", delta, self.shared_cov_inv_, delta)
            )
        return dists.min(axis=1)

    def predict(self, embedding: np.ndarray) -> dict:
        """Single-sample prediction."""
        assert self.is_fitted_
        dists      = np.array([
            float(np.sqrt((embedding - self.class_means_[c]) @
                          self.shared_cov_inv_ @
                          (embedding - self.class_means_[c])))
            for c in range(self.n_classes_)
        ])
        min_dist   = float(dists.min())
        pred_class = int(dists.argmin())
        is_ood     = min_dist > self.threshold_
        return {
            "ood"      : is_ood,
            "class"    : None if is_ood else pred_class,
            "distance" : min_dist,
            "threshold": self.threshold_,
        }

    def predict_batch(self, embeddings: np.ndarray) -> list[dict]:
        """Batch prediction."""
        min_dists    = self._min_distances(embeddings)
        dists_all    = np.zeros((len(embeddings), self.n_classes_), dtype=np.float64)
        for c in range(self.n_classes_):
            delta = embeddings - self.class_means_[c]
            dists_all[:, c] = np.sqrt(
                np.einsum("nd,dd,nd->n", delta, self.shared_cov_inv_, delta)
            )
        pred_classes = dists_all.argmin(axis=1)
        return [
            {
                "ood"      : float(min_dists[i]) > self.threshold_,
                "class"    : None if float(min_dists[i]) > self.threshold_ else int(pred_classes[i]),
                "distance" : float(min_dists[i]),
                "threshold": self.threshold_,
            }
            for i in range(len(embeddings))
        ]

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"[OOD] Saved → {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            self.__dict__.update(pickle.load(f))
        print(f"[OOD] Loaded ← {path}")
        return self


def extract_embeddings(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 128-d fused embeddings from all batches.
    Expects batches with keys: spectrogram, physics_features,
    rpm_raw, rpm_bucket, bearing_type, label.
    """
    model.eval()
    all_embs, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            emb = model.get_embedding(
                batch["spectrogram"].to(device),
                batch["physics_features"].to(device),
                batch["rpm_raw"].to(device),
                batch["rpm_bucket"].to(device),
                batch["bearing_type"].to(device),
            )
            all_embs.append(emb.cpu().numpy())
            all_labels.append(batch["label"].numpy())

    return (
        np.concatenate(all_embs,   axis=0),
        np.concatenate(all_labels, axis=0),
    )


# ── smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    N, d, n_classes = 200, 128, 4
    embeddings = np.vstack([
        np.random.randn(N // n_classes, d) + np.array([i * 3] + [0] * (d - 1))
        for i in range(n_classes)
    ])
    labels = np.repeat(np.arange(n_classes), N // n_classes)

    det = OODDetector()
    det.fit(embeddings, labels, n_classes=n_classes)

    print("In-dist :", det.predict(embeddings[0]))
    print("OOD     :", det.predict(np.ones(d) * 50.0))
    det.save("outputs/ood_detector.pkl")
    print("✓ ood_detector.py OK")
