"""
fusion_model.py
───────────────
Attention-based ensemble fusion of pre-trained branch models.

The three branch models (spectrogram, physics, metadata) are loaded from
their saved checkpoints. Their logits are combined via learned attention weights
that adapt per-sample based on physics features.

Fusion pipeline
───────────────
1. Load pre-trained branch model weights from outputs/
2. Branch encoder backbones → feature embeddings (frozen or fine-tunable)
3. Physics-guided attention determines per-sample branch weights
4. Weighted ensemble of branch logits:
     fused_logit = w_spec * spec_logit + w_phys * phys_logit + w_meta * meta_logit
5. Final prediction from fused logit

Usage
─────
    from fusion_model import BearingFaultFusion, build_fusion_model
    
    model = build_fusion_model(
        physics_input_dim = 7,
        n_rpm_buckets = 8,
        n_bearing_types = 4,
        spec_ckpt = "outputs/spectrogram_branch.pt",
        phys_ckpt = "outputs/physics_branch.pt",
        meta_ckpt = "outputs/metadata_branch.pt",
    )
    logits, aux = model(mel_spec, phys_feats, rpm_raw, rpm_bucket, bearing_type)
    # aux keys: attn_weights, branch_logits
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_branch_models import SpectrogramBranch, PhysicsBranch, MetadataBranch


# ──────────────────────────────────────────────────────────────────────────────
# Constants & priors
# ──────────────────────────────────────────────────────────────────────────────

EMBED_DIM   = 128
N_HEADS     = 4
N_BRANCHES  = 3

# Suggested initial branch weights (spec, phys, meta)
_PRIOR_LOGITS = torch.tensor([0.40, 0.35, 0.25]).log()


# ──────────────────────────────────────────────────────────────────────────────
# Physics-Guided Attention Gate
# ──────────────────────────────────────────────────────────────────────────────

class AttentionWeightGate(nn.Module):
    """
    Learns per-sample attention weights for the three branches.
    Physics features guide the weighting: the model learns which branch
    is most informative based on the fault signature.
    
    Output: (B, 3) softmaxed weights for [spec, phys, meta]
    """

    def __init__(self, phys_dim: int = 7, embed_dim: int = EMBED_DIM, n_heads: int = N_HEADS):
        super().__init__()
        self.phys_dim = phys_dim
        self.embed_dim = embed_dim
        
        # Physics → query projection
        self.phys_to_query = nn.Sequential(
            nn.Linear(phys_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        
        # Multi-head attention: Q=physics, K=V=other branches
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1,
        )
        
        # Learnable prior bias
        self.prior_bias = nn.Parameter(_PRIOR_LOGITS.clone().float())  # (3,)
        
        # Output gate: embed_dim → 3 weights
        self.to_weights = nn.Linear(embed_dim, N_BRANCHES)

    def forward(
        self,
        phys_raw: torch.Tensor,       # (B, phys_dim) raw physics features
        spec_emb: torch.Tensor,       # (B, embed_dim)
        meta_emb: torch.Tensor,       # (B, embed_dim)
    ) -> torch.Tensor:
        """
        Returns
        -------
        weights : (B, 3)  softmaxed attention weights for [spec, phys, meta]
        """
        B = spec_emb.size(0)
        
        # Project physics to query
        query = self.phys_to_query(phys_raw).unsqueeze(1)  # (B, 1, D)
        
        # Stack non-physics branches as key/value
        branch_tokens = torch.stack([spec_emb, meta_emb], dim=1)  # (B, 2, D)
        
        # Cross-attention: Q=physics, K=V=branches
        ctx, _ = self.mha(query, branch_tokens, branch_tokens)  # (B, 1, D)
        ctx = ctx.squeeze(1)  # (B, D)
        
        # Gate to 3 weights
        logits = self.to_weights(ctx)  # (B, 3)
        
        # Add prior bias
        prior = F.softmax(self.prior_bias, dim=0)
        logits = logits + prior.unsqueeze(0).log()
        weights = F.softmax(logits, dim=-1)  # (B, 3)
        
        return weights


# ──────────────────────────────────────────────────────────────────────────────
# Fusion Loss
# ──────────────────────────────────────────────────────────────────────────────

def fusion_loss(
    fused_logits: torch.Tensor,       # (B, n_classes)
    labels: torch.Tensor,             # (B,)
    attn_weights: torch.Tensor,       # (B, 3)
    lambda_attn: float = 0.05,
) -> torch.Tensor:
    """
    L = CE(logits, labels) + λ_attn * attention_entropy_penalty
    
    Entropy penalty encourages the model to be decisive about which
    branch to trust rather than uniformly distributing attention.
    """
    ce_loss = F.cross_entropy(fused_logits, labels)
    
    # Attention entropy penalty
    eps = 1e-8
    attn_entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=1)
    attn_penalty = attn_entropy.mean()
    
    return ce_loss + lambda_attn * attn_penalty


# ──────────────────────────────────────────────────────────────────────────────
# Auxiliary outputs
# ──────────────────────────────────────────────────────────────────────────────

class FusionAux(NamedTuple):
    attn_weights: torch.Tensor        # (B, 3)  [spec, phys, meta]
    branch_logits: torch.Tensor       # (B, 3, n_classes)
    branch_embeddings: torch.Tensor   # (B, 3, embed_dim)


# ──────────────────────────────────────────────────────────────────────────────
# BearingFaultFusion – Main Model
# ──────────────────────────────────────────────────────────────────────────────

class BearingFaultFusion(nn.Module):
    """
    Attention-weighted ensemble of pre-trained branch models.
    
    Forward pass
    ────────────
    (mel_spec, physics_feats, rpm_raw, rpm_bucket, bearing_type)
        → spec_branch  → spec_logit  (B, n_classes)
        → phys_branch  → phys_logit  (B, n_classes)
        → meta_branch  → meta_logit  (B, n_classes)
        
        → AttentionWeightGate(phys_feats) → weights (B, 3)
        
        → fused_logit = w_spec*spec_logit + w_phys*phys_logit + w_meta*meta_logit
        
    Returns
    -------
    logits : (B, n_classes)  fused prediction logits
    aux    : FusionAux(attn_weights, branch_logits, branch_embeddings)
    """

    def __init__(
        self,
        physics_input_dim: int = 7,
        n_rpm_buckets: int = 8,
        n_bearing_types: int = 4,
        n_classes: int = 2,
        freeze_branches: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.freeze_branches = freeze_branches
        
        # Load or initialize pre-trained branch models
        self.spec_branch = SpectrogramBranch(n_classes=n_classes)
        self.phys_branch = PhysicsBranch(input_dim=physics_input_dim, n_classes=n_classes)
        self.meta_branch = MetadataBranch(
            n_bearing_types=n_bearing_types,
            n_rpm_buckets=n_rpm_buckets,
            n_classes=n_classes,
        )
        
        # Attention gate to weight branches
        self.attention_gate = AttentionWeightGate(
            phys_dim=physics_input_dim,
            embed_dim=EMBED_DIM,
            n_heads=N_HEADS,
        )
        
        # Optional: freeze branch parameters if desired
        if freeze_branches:
            for param in self.spec_branch.parameters():
                param.requires_grad = False
            for param in self.phys_branch.parameters():
                param.requires_grad = False
            for param in self.meta_branch.parameters():
                param.requires_grad = False

    def load_branch_checkpoints(
        self,
        spec_ckpt: str | Path | None = None,
        phys_ckpt: str | Path | None = None,
        meta_ckpt: str | Path | None = None,
    ):
        """Load pre-trained branch weights from checkpoints."""
        device = next(self.parameters()).device
        
        if spec_ckpt:
            spec_path = Path(spec_ckpt)
            if spec_path.exists():
                state_dict = torch.load(spec_path, map_location=device)
                self.spec_branch.load_state_dict(state_dict, strict=False)
                print(f"  ✓ Loaded SpectrogramBranch from {spec_path.name}")
            else:
                print(f"  ⚠ SpectrogramBranch checkpoint not found: {spec_path}")
        
        if phys_ckpt:
            phys_path = Path(phys_ckpt)
            if phys_path.exists():
                state_dict = torch.load(phys_path, map_location=device)
                self.phys_branch.load_state_dict(state_dict, strict=False)
                print(f"  ✓ Loaded PhysicsBranch from {phys_path.name}")
            else:
                print(f"  ⚠ PhysicsBranch checkpoint not found: {phys_path}")
        
        if meta_ckpt:
            meta_path = Path(meta_ckpt)
            if meta_path.exists():
                state_dict = torch.load(meta_path, map_location=device)
                self.meta_branch.load_state_dict(state_dict, strict=False)
                print(f"  ✓ Loaded MetadataBranch from {meta_path.name}")
            else:
                print(f"  ⚠ MetadataBranch checkpoint not found: {meta_path}")

    def forward(
        self,
        mel_spec: torch.Tensor,        # (B, 1, F, T)
        physics_feats: torch.Tensor,   # (B, phys_dim)
        rpm_raw: torch.Tensor,         # (B,)
        rpm_bucket: torch.Tensor,      # (B,)
        bearing_type: torch.Tensor,    # (B,)
    ) -> tuple[torch.Tensor, FusionAux]:
        """
        Returns
        -------
        logits : (B, n_classes)
        aux    : FusionAux with attention weights, branch logits, embeddings
        """
        
        # ── Forward through branches (get logits) ───────────────────
        spec_logits = self.spec_branch(mel_spec)                    # (B, n_classes)
        phys_logits = self.phys_branch(physics_feats)               # (B, n_classes)
        meta_logits = self.meta_branch(rpm_raw, rpm_bucket, bearing_type)  # (B, n_classes)
        
        # ── Also get embeddings from backbones for attention ────────
        spec_emb = self.spec_branch.backbone(mel_spec)              # (B, 128)
        phys_emb = self.phys_branch.backbone(physics_feats)         # (B, 128)
        meta_emb = self.meta_branch.backbone(rpm_raw, rpm_bucket, bearing_type)  # (B, 128)
        
        # ── Compute attention weights ──────────────────────────────
        attn_weights = self.attention_gate(physics_feats, spec_emb, meta_emb)  # (B, 3)
        
        # ── Fused prediction: weighted ensemble of logits ──────────
        # weights: (B, 3) → expand to (B, 3, 1) for broadcasting
        weighted_logits = (
            attn_weights[:, 0:1, None] * spec_logits +
            attn_weights[:, 1:2, None] * phys_logits +
            attn_weights[:, 2:3, None] * meta_logits
        )  # (B, n_classes)
        
        # ── Pack auxiliary outputs ─────────────────────────────────
        branch_logits = torch.stack([spec_logits, phys_logits, meta_logits], dim=1)  # (B, 3, n_classes)
        branch_embeddings = torch.stack([spec_emb, phys_emb, meta_emb], dim=1)  # (B, 3, 128)
        
        aux = FusionAux(
            attn_weights=attn_weights,
            branch_logits=branch_logits,
            branch_embeddings=branch_embeddings,
        )
        
        return weighted_logits, aux


# ──────────────────────────────────────────────────────────────────────────────
# Training & Evaluation Helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_fusion_model(
    physics_input_dim: int = 7,
    n_rpm_buckets: int = 8,
    n_bearing_types: int = 4,
    n_classes: int = 2,
    spec_ckpt: str | Path | None = None,
    phys_ckpt: str | Path | None = None,
    meta_ckpt: str | Path | None = None,
    freeze_branches: bool = False,
) -> BearingFaultFusion:
    """Build and optionally load pre-trained branch models."""
    model = BearingFaultFusion(
        physics_input_dim=physics_input_dim,
        n_rpm_buckets=n_rpm_buckets,
        n_bearing_types=n_bearing_types,
        n_classes=n_classes,
        freeze_branches=freeze_branches,
    )
    
    # Load branch checkpoints if provided
    model.load_branch_checkpoints(spec_ckpt, phys_ckpt, meta_ckpt)
    
    return model


def train_step_fusion(
    model: BearingFaultFusion,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_attn: float = 0.05,
) -> float:
    """Single training step. Returns loss."""
    model.train()
    
    mel = batch["mel_spec"].to(device)
    phys = batch["physics_features"].to(device)
    rpm_r = batch["rpm_raw"].to(device)
    rpm_b = batch["rpm_bucket"].to(device)
    bear = batch["bearing_type"].to(device)
    lbl = batch["label"].to(device)
    
    optimizer.zero_grad(set_to_none=True)
    logits, aux = model(mel, phys, rpm_r, rpm_b, bear)
    
    loss = fusion_loss(logits, lbl, aux.attn_weights, lambda_attn=lambda_attn)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def eval_fusion(
    model: BearingFaultFusion,
    loader,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    """
    Returns (accuracy, attn_weights_all) where
    attn_weights_all : (N, 3) attention weights across the test set.
    """
    model.eval()
    correct = total = 0
    all_attn = []
    
    for batch in loader:
        mel = batch["mel_spec"].to(device)
        phys = batch["physics_features"].to(device)
        rpm_r = batch["rpm_raw"].to(device)
        rpm_b = batch["rpm_bucket"].to(device)
        bear = batch["bearing_type"].to(device)
        lbl = batch["label"].to(device)
        
        logits, aux = model(mel, phys, rpm_r, rpm_b, bear)
        preds = logits.argmax(dim=1)
        correct += (preds == lbl).sum().item()
        total += lbl.size(0)
        all_attn.append(aux.attn_weights.cpu())
    
    acc = correct / max(total, 1)
    attn = torch.cat(all_attn, dim=0) if all_attn else torch.zeros(0, 3)
    return acc, attn
