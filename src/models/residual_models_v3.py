"""Experimental SRAF-ID repair-v3-light modules.

This module is intentionally separate from the frozen v1/v2 implementations.
It adds a lightweight train-only time-of-day profile candidate, observed-aware
top-k spatial repair, and softmax fusion over temporal/spatial/profile repair
candidates. It is for diagnostic experiments only unless later promoted.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.residual_models import basic_temporal_repair, bidirectional_temporal_repair, finite_mask, spatial_repair
from src.models.residual_models_v2 import _local_variance, _stuck_duration, reliability_features_v2


def observed_aware_topk_spatial_repair(
    x_filled: torch.Tensor,
    observed_mask: torch.Tensor,
    adjacency: torch.Tensor,
    k: int = 5,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Top-k adjacency repair with neighbor-observation filtering."""

    sensors = x_filled.shape[2]
    kk = min(max(1, k), sensors)
    weights, idx = torch.topk(adjacency, k=kk, dim=1)
    x_neighbors = x_filled[:, :, idx, :]  # [B,L,N,K,1]
    m_neighbors = observed_mask[:, :, idx, :]
    w = weights.view(1, 1, sensors, kk, 1).to(dtype=x_filled.dtype, device=x_filled.device)
    w = w * m_neighbors
    denom = w.sum(dim=3).clamp_min(epsilon)
    spatial = (x_neighbors * w).sum(dim=3) / denom
    fallback = x_filled
    return torch.where(denom > epsilon, spatial, fallback)


class SRAFRepairV3Light(nn.Module):
    """Lightweight multi-candidate repair bank for SRAF-ID diagnostics."""

    def __init__(
        self,
        sensors: int,
        tod_profile: torch.Tensor,
        topk: int = 5,
        fusion_hidden_dim: int = 32,
        observed_input_blend: float = 0.5,
        epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if tod_profile.ndim != 3:
            raise ValueError(f"Expected tod_profile [288,N,1], got {tuple(tod_profile.shape)}")
        self.sensors = sensors
        self.topk = topk
        self.observed_input_blend = observed_input_blend
        self.epsilon = epsilon
        self.register_buffer("tod_profile", tod_profile.float())
        self.fusion_feature_names = [
            "x_filled",
            "observed_mask",
            "local_variance",
            "stuck_duration",
            "missing_ratio",
            "abs_temporal_spatial",
            "abs_temporal_profile",
            "abs_spatial_profile",
            "tod_norm",
            "dow_norm",
        ]
        self.fusion_mlp = nn.Sequential(
            nn.Linear(len(self.fusion_feature_names), fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 3),
        )

    def _profile_candidate(self, tod_norm: torch.Tensor) -> torch.Tensor:
        idx = torch.floor(tod_norm[..., :1] * 288.0).long().clamp(0, 287)
        # tod_profile: [288,N,1], idx: [B,L,N,1]
        profile_by_tod = self.tod_profile[:, None, :, :]  # [288,1,N,1]
        idx_flat = idx.reshape(-1)
        sensor_idx = torch.arange(self.sensors, device=idx.device).view(1, 1, self.sensors, 1).expand_as(idx).reshape(-1)
        values = self.tod_profile[idx_flat, sensor_idx, 0]
        return values.reshape_as(idx).to(dtype=tod_norm.dtype)

    def repair_components(
        self,
        speed: torch.Tensor,
        identity: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if speed.ndim != 4 or speed.shape[-1] != 1:
            raise ValueError(f"Expected speed [B,L,N,1], got {tuple(speed.shape)}")
        if identity.ndim != 4 or identity.shape[-1] < 2:
            raise ValueError(f"Expected identity [B,L,N,2+], got {tuple(identity.shape)}")
        mask = finite_mask(speed, observed_mask=observed_mask)[..., :1]
        x_filled = torch.nan_to_num(speed, nan=0.0)
        if adjacency is None:
            adjacency = torch.eye(self.sensors, device=x_filled.device, dtype=x_filled.dtype)
        adjacency = adjacency.to(device=x_filled.device, dtype=x_filled.dtype)

        x_temp, _ = bidirectional_temporal_repair(speed, mask)
        x_sp = observed_aware_topk_spatial_repair(x_filled, mask, adjacency, k=self.topk, epsilon=self.epsilon)
        x_profile = self._profile_candidate(identity[..., 0:1])

        missing_ratio = (1.0 - mask).mean(dim=(1, 2, 3), keepdim=True).expand_as(x_filled)
        feats = torch.cat(
            [
                x_filled,
                mask,
                _local_variance(x_filled, window=3),
                _stuck_duration(mask, x_filled),
                missing_ratio,
                torch.abs(x_temp - x_sp),
                torch.abs(x_temp - x_profile),
                torch.abs(x_sp - x_profile),
                identity[..., 0:1],
                identity[..., 1:2],
            ],
            dim=-1,
        )
        logits = self.fusion_mlp(feats)
        weights = torch.softmax(logits, dim=-1)
        x_rep = (
            weights[..., 0:1] * x_temp
            + weights[..., 1:2] * x_sp
            + weights[..., 2:3] * x_profile
        )
        observed_blend = torch.full_like(mask, self.observed_input_blend)
        repaired = torch.where(mask > 0.5, observed_blend * x_filled + (1.0 - observed_blend) * x_rep, x_rep)
        return {
            "repaired_input": repaired,
            "mask": mask,
            "x_filled": x_filled,
            "temporal_repair": x_temp,
            "spatial_repair": x_sp,
            "profile_repair": x_profile,
            "repair_blend": x_rep,
            "candidate_weights": weights,
            "candidate_names": ["bidirectional_temporal", "observed_topk_spatial", "tod_profile"],
            "fusion_feature_names": self.fusion_feature_names,
            "reliability": torch.zeros_like(mask),
            "alpha": weights[..., 0:1],
        }


class SRAFRepairFactorAblation(nn.Module):
    """Controlled repair-factor ablation used for A1-A5 diagnostics."""

    def __init__(
        self,
        sensors: int,
        tod_profile: torch.Tensor,
        temporal_mode: str = "basic",
        spatial_mode: str = "adjacency",
        fusion_mode: str = "alpha",
        use_profile: bool = False,
        topk: int = 5,
        hidden_dim: int = 16,
        fixed_profile_weight: float = 0.10,
        observed_input_blend: float = 0.5,
        epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.sensors = sensors
        self.temporal_mode = temporal_mode
        self.spatial_mode = spatial_mode
        self.fusion_mode = fusion_mode
        self.use_profile = use_profile
        self.topk = topk
        self.fixed_profile_weight = fixed_profile_weight
        self.observed_input_blend = observed_input_blend
        self.epsilon = epsilon
        self.register_buffer("tod_profile", tod_profile.float())
        dummy = torch.zeros((1, 3, 2, 1))
        feats, _ = reliability_features_v2(
            dummy,
            torch.ones_like(dummy),
            dummy,
            dummy,
            include_stuck_features=True,
            include_second_delta=True,
            include_flatness=False,
            include_repair_disagreement=True,
        )
        feature_dim = feats.shape[-1]
        if fusion_mode == "softmax":
            out_dim = 3 if use_profile else 2
            self.fusion = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
            self.alpha = None
        elif fusion_mode in {"temporal_only", "spatial_only", "fixed"}:
            self.fusion = None
            self.alpha = None
        else:
            self.alpha = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.fusion = None

    def _profile_candidate(self, tod_norm: torch.Tensor) -> torch.Tensor:
        idx = torch.floor(tod_norm[..., :1] * 288.0).long().clamp(0, 287)
        idx_flat = idx.reshape(-1)
        sensor_idx = torch.arange(self.sensors, device=idx.device).view(1, 1, self.sensors, 1).expand_as(idx).reshape(-1)
        values = self.tod_profile[idx_flat, sensor_idx, 0]
        return values.reshape_as(idx).to(dtype=tod_norm.dtype)

    def repair_components(
        self,
        speed: torch.Tensor,
        identity: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mask = finite_mask(speed, observed_mask=observed_mask)[..., :1]
        x_filled = torch.nan_to_num(speed, nan=0.0)
        if adjacency is None:
            adjacency = torch.eye(self.sensors, dtype=x_filled.dtype, device=x_filled.device)
        adjacency = adjacency.to(device=x_filled.device, dtype=x_filled.dtype)
        if self.temporal_mode == "bidir":
            x_temp, _ = bidirectional_temporal_repair(speed, mask)
        else:
            x_temp, _ = basic_temporal_repair(speed, mask)
        if self.spatial_mode == "topk":
            x_sp = observed_aware_topk_spatial_repair(x_filled, mask, adjacency, k=self.topk, epsilon=self.epsilon)
        else:
            x_sp = spatial_repair(x_filled, adjacency, epsilon=self.epsilon)
        x_profile = self._profile_candidate(identity[..., 0:1])
        feats, _ = reliability_features_v2(
            x_filled,
            mask,
            x_temp,
            x_sp,
            include_stuck_features=True,
            include_second_delta=True,
            include_flatness=False,
            include_repair_disagreement=True,
        )
        if self.fusion_mode == "temporal_only":
            x_rep = x_temp
            alpha = torch.ones_like(x_filled)
            weights = torch.cat([alpha, torch.zeros_like(alpha)], dim=-1)
            names = ["temporal", "spatial"]
        elif self.fusion_mode == "spatial_only":
            x_rep = x_sp
            alpha = torch.zeros_like(x_filled)
            weights = torch.cat([alpha, torch.ones_like(alpha)], dim=-1)
            names = ["temporal", "spatial"]
        elif self.fusion_mode == "fixed":
            alpha = torch.full_like(x_filled, 0.5)
            x_rep = 0.5 * x_temp + 0.5 * x_sp
            weights = torch.cat([alpha, alpha], dim=-1)
            names = ["temporal", "spatial"]
        elif self.fusion_mode == "softmax":
            candidates = [x_temp, x_sp]
            names = ["temporal", "spatial"]
            if self.use_profile:
                candidates.append(x_profile)
                names.append("profile")
            weights = torch.softmax(self.fusion(feats), dim=-1)
            x_rep = torch.zeros_like(x_filled)
            for idx, cand in enumerate(candidates):
                x_rep = x_rep + weights[..., idx : idx + 1] * cand
            alpha = weights[..., 0:1]
        else:
            alpha = self.alpha(feats)
            base = alpha * x_temp + (1.0 - alpha) * x_sp
            if self.use_profile:
                w = self.fixed_profile_weight
                x_rep = (1.0 - w) * base + w * x_profile
                weights = torch.cat(
                    [
                        (1.0 - w) * alpha,
                        (1.0 - w) * (1.0 - alpha),
                        torch.full_like(alpha, w),
                    ],
                    dim=-1,
                )
                names = ["temporal", "spatial", "profile"]
            else:
                x_rep = base
                weights = torch.cat([alpha, 1.0 - alpha], dim=-1)
                names = ["temporal", "spatial"]
        repaired = torch.where(mask > 0.5, self.observed_input_blend * x_filled + (1.0 - self.observed_input_blend) * x_rep, x_rep)
        return {
            "repaired_input": repaired,
            "mask": mask,
            "x_filled": x_filled,
            "temporal_repair": x_temp,
            "spatial_repair": x_sp,
            "profile_repair": x_profile,
            "repair_blend": x_rep,
            "candidate_weights": weights,
            "candidate_names": names,
            "reliability": torch.zeros_like(mask),
            "alpha": alpha,
        }
