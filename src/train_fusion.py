"""
train_fusion.py
───────────────
Training pipeline for the attention-based fusion model.

The fusion model combines pre-trained branch models (loaded from outputs/)
via learned attention weights. The branches can be optionally frozen or
fine-tuned during fusion training.

Usage
─────
  python train_fusion.py --data_dir "SCA bearing dataset" --epochs 20

Optional: freeze branch models
  python train_fusion.py --data_dir "SCA bearing dataset" --epochs 20 --freeze_branches
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BearingFaultDataset
from fusion_model import (
    BearingFaultFusion,
    build_fusion_model,
    eval_fusion,
    train_step_fusion,
)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_fusion(cfg: dict):
    """Main training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── 1. Data ────────────────────────────────────────────────────────
    print(f"Loading dataset from: {cfg['data_dir']}")
    dataset = BearingFaultDataset(dataset_root=cfg["data_dir"], label_mode="binary")
    
    # 80/20 train-test split with fixed seed
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    test_len = total_len - train_len
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_len, test_len],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    
    print(f"Train: {len(train_ds)}  Test: {len(test_ds)}\n")
    
    # ── 2. Model ───────────────────────────────────────────────────────
    print("Building BearingFaultFusion model...")
    model = build_fusion_model(
        physics_input_dim=dataset.physics_input_dim,
        n_rpm_buckets=8,
        n_bearing_types=dataset.n_bearing_types,
        n_classes=2,
        spec_ckpt="outputs/spectrogram_branch.pt",
        phys_ckpt="outputs/physics_branch.pt",
        meta_ckpt="outputs/metadata_branch.pt",
        freeze_branches=cfg.get("freeze_branches", False),
    ).to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / Total: {total_params:,}\n")
    
    # ── 3. Optimizer & scheduler ───────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["lr"],
        epochs=cfg["epochs"],
        steps_per_epoch=max(len(train_loader), 1),
        pct_start=0.3,
    )
    
    # ── 4. Training loop ───────────────────────────────────────────────
    best_acc = 0.0
    best_state = None
    print(f"{'═'*70}")
    print(f"Training BearingFaultFusion for {cfg['epochs']} epochs")
    print(f"{'═'*70}\n")
    
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            loss = train_step_fusion(
                model, batch, optimizer, device,
                lambda_attn=cfg.get("lambda_attn", 0.05)
            )
            running_loss += loss * batch["label"].size(0)
            scheduler.step()
        
        train_loss = running_loss / max(len(train_ds), 1)
        test_acc, attn = eval_fusion(model, test_loader, device)
        
        mean_attn = attn.mean(0) if attn.size(0) > 0 else torch.zeros(3)
        attn_str = "  ".join(
            f"{n}={w:.2f}" for n, w in zip(
                ["spec", "phys", "meta"], mean_attn.tolist()
            )
        )
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = "  ★"
        else:
            marker = ""
        
        print(f"Epoch {epoch:02d}/{cfg['epochs']}  "
              f"loss={train_loss:.4f}  test_acc={test_acc:.4f}  "
              f"attn=[{attn_str}]{marker}")
    
    # ── 5. Final evaluation ────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    model.to(device)
    test_acc, attn_all = eval_fusion(model, test_loader, device)
    
    print(f"\n{'─'*70}")
    print(f"Best Test Accuracy: {best_acc:.4f}\n")
    
    # Attention weight summary
    if attn_all.size(0) > 0:
        mean_attn = attn_all.mean(0)
        print("Mean Attention Weights Across Test Set:")
        for name, w in zip(["Spectrogram", "Physics", "Metadata"], mean_attn.tolist()):
            bar = "█" * int(w * 40)
            print(f"  {name:<12} {w:.4f}  {bar}")
    
    # ── 6. Save ───────────────────────────────────────────────────────
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / "bearing_fault_fusion.pt"
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg,
        "best_acc": best_acc,
    }, ckpt_path)
    print(f"\nCheckpoint saved → {ckpt_path}")
    
    return model, best_acc


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train BearingFaultFusion model")
    p.add_argument("--data_dir",        required=True, default="SCA bearing dataset")
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--lambda_attn",     type=float, default=0.05)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--freeze_branches", action="store_true",
                   help="Freeze pre-trained branch models")
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    torch.manual_seed(cfg["seed"])
    model, best_acc = train_fusion(cfg)


