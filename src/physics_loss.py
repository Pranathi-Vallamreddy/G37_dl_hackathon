"""
PILL — physics_loss.py
Physics-informed loss matched exactly to pipeline.py feature vector layout.

Feature vector from compute_physics_features() — 29 dims total:
  Per fault type (BPFI, BPFO, BSF, FTF) × 3 harmonics × 2 values (raw, env):
    idx 0,1   → BPFI harmonic-1 (raw, env)
    idx 2,3   → BPFI harmonic-2 (raw, env)
    idx 4,5   → BPFI harmonic-3 (raw, env)
    idx 6,7   → BPFO harmonic-1 (raw, env)
    idx 8,9   → BPFO harmonic-2 (raw, env)
    idx 10,11 → BPFO harmonic-3 (raw, env)
    idx 12,13 → BSF  harmonic-1 (raw, env)
    idx 14,15 → BSF  harmonic-2 (raw, env)
    idx 16,17 → BSF  harmonic-3 (raw, env)
    idx 18,19 → FTF  harmonic-1 (raw, env)
    idx 20,21 → FTF  harmonic-2 (raw, env)
    idx 22,23 → FTF  harmonic-3 (raw, env)
    idx 24    → rms
    idx 25    → kurtosis
    idx 26    → crest
    idx 27    → skewness
    idx 28    → env_kurtosis

4-class label convention:
    0 = Healthy
    1 = Inner race  (BPFI) → envelope energy = mean of idx 1,3,5
    2 = Ball        (BSF)  → envelope energy = mean of idx 13,15,17
    3 = Outer race  (BPFO) → envelope energy = mean of idx 7,9,11

Usage:
    from physics_loss import PhysicsInformedLoss
    criterion = PhysicsInformedLoss(lambda_penalty=0.3)
    loss, info = criterion(logits, labels, physics_features)
"""

import torch
import torch.nn as nn


# envelope energy indices per fault class (after StandardScaler → mean=0)
# using mean of 3 harmonic envelope columns as the energy proxy
FAULT_ENV_INDICES = {
    1: [1, 3, 5],     # BPFI envelope harmonics
    2: [13, 15, 17],  # BSF  envelope harmonics
    3: [7, 9, 11],    # BPFO envelope harmonics
}


class PhysicsInformedLoss(nn.Module):
    """
    L = CrossEntropy(logits, labels)
      + lambda_penalty * physics_penalty

    Physics penalty:
        For each sample where the model predicts a fault class,
        check that the mean envelope energy at the expected fault
        frequency harmonics is above energy_threshold.
        After StandardScaler, 0.0 = population mean.
        Predicting a fault with below-mean energy is penalised.

    Args:
        lambda_penalty   : weight on physics term         (default 0.3)
        energy_threshold : scaled energy floor            (default 0.0)
        label_smoothing  : passed to CrossEntropyLoss     (default 0.05)
    """

    def __init__(
        self,
        lambda_penalty: float   = 0.3,
        energy_threshold: float = 0.0,
        label_smoothing: float  = 0.05,
    ):
        super().__init__()
        self.lambda_penalty   = lambda_penalty
        self.energy_threshold = energy_threshold
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,            # (B, 4)
        labels: torch.Tensor,            # (B,) long
        physics_features: torch.Tensor,  # (B, 29) standard-scaled
    ) -> tuple[torch.Tensor, dict]:

        ce_loss = self.ce(logits, labels)

        predicted = logits.argmax(dim=1)   # (B,)
        penalty   = torch.zeros(1, device=logits.device)
        n_penalised = 0

        for fault_class, env_idxs in FAULT_ENV_INDICES.items():
            mask = (predicted == fault_class)
            if mask.sum() == 0:
                continue

            # mean envelope energy across the 3 harmonics for masked samples
            env_energy = physics_features[mask][:, env_idxs].mean(dim=1)  # (n_masked,)

            below = env_energy < self.energy_threshold
            if below.sum() > 0:
                deficit = self.energy_threshold - env_energy[below]
                penalty = penalty + (deficit ** 2).sum()
                n_penalised += int(below.sum())

        # normalise by batch size
        penalty = penalty / max(logits.shape[0], 1)
        total   = ce_loss + self.lambda_penalty * penalty.squeeze()

        return total, {
            "ce_loss"        : ce_loss.item(),
            "physics_penalty": penalty.item(),
            "n_penalised"    : n_penalised,
        }


# ── smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 16
    logits = torch.randn(B, 4)
    labels = torch.randint(0, 4, (B,))
    phys   = torch.randn(B, 29)   # 29 dims matching pipeline.py

    criterion = PhysicsInformedLoss()
    loss, info = criterion(logits, labels, phys)

    print("PhysicsInformedLoss smoke test")
    print(f"  total            : {loss.item():.4f}")
    print(f"  ce_loss          : {info['ce_loss']:.4f}")
    print(f"  physics_penalty  : {info['physics_penalty']:.4f}")
    print(f"  n_penalised      : {info['n_penalised']}/{B}")
    print("✓ physics_loss.py OK")
