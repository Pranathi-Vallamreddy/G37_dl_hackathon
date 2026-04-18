from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_c, max(out_c // 4, 1)),
            nn.ReLU(),
            nn.Linear(max(out_c // 4, 1), out_c),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.conv(x)
        se_w = self.se(feat).unsqueeze(-1).unsqueeze(-1)
        return feat * se_w


class SpectrogramCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, 128)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return self.proj(x)


class PhysicsMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

    def forward(self, x):
        return self.net(x)


class MetadataMLP(nn.Module):
    def __init__(self, n_bearing_types: int, n_rpm_buckets: int = 8):
        super().__init__()
        self.rpm_embed = nn.Embedding(n_rpm_buckets, 16)
        self.type_embed = nn.Embedding(n_bearing_types, 16)
        self.net = nn.Sequential(
            nn.Linear(33, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

    def forward(self, rpm_raw, rpm_bucket, bearing_type):
        r = self.rpm_embed(rpm_bucket)
        t = self.type_embed(bearing_type)
        x = torch.cat([r, t, rpm_raw.unsqueeze(1)], dim=1)
        return self.net(x)


class FusionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(320, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

    def forward(self, cnn_feat, physics_feat, meta_feat):
        phys_q = physics_feat.unsqueeze(1)
        cnn_kv = cnn_feat.unsqueeze(1)
        attended, _ = self.cross_attn(phys_q, cnn_kv, cnn_kv)
        attended = attended.squeeze(1)

        fused = torch.cat([attended, physics_feat, meta_feat], dim=1)
        return self.proj(fused)


class BearingFaultNet(nn.Module):
    def __init__(self, n_classes: int, physics_input_dim: int, n_bearing_types: int):
        super().__init__()
        self.cnn = SpectrogramCNN()
        self.physics = PhysicsMLP(physics_input_dim)
        self.metadata = MetadataMLP(n_bearing_types=n_bearing_types)
        self.fusion = FusionModule()
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, mel_spec, physics_features, rpm_raw, rpm_bucket, bearing_type):
        cnn_feat = self.cnn(mel_spec)
        physics_feat = self.physics(physics_features)
        meta_feat = self.metadata(rpm_raw, rpm_bucket, bearing_type)
        fused = self.fusion(cnn_feat, physics_feat, meta_feat)
        return self.head(fused)

    def get_embedding(self, mel_spec, physics_features, rpm_raw, rpm_bucket, bearing_type):
        """Extract 128-d fused embedding without classification head."""
        cnn_feat = self.cnn(mel_spec)
        physics_feat = self.physics(physics_features)
        meta_feat = self.metadata(rpm_raw, rpm_bucket, bearing_type)
        fused = self.fusion(cnn_feat, physics_feat, meta_feat)
        return fused  # (B, 128)


def physics_informed_loss(logits, labels, physics_features, fault_energies, lambda_phys=0.1):
    ce_loss = nn.CrossEntropyLoss()(logits, labels)

    pred_classes = logits.argmax(dim=1)
    faulty_mask = pred_classes > 0

    if faulty_mask.sum() > 0:
        expected_elevation = fault_energies[faulty_mask]
        physics_penalty = torch.relu(0.01 - expected_elevation).mean()
    else:
        physics_penalty = torch.tensor(0.0, device=logits.device)

    return ce_loss + lambda_phys * physics_penalty
