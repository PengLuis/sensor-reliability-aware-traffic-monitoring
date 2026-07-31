"""Experimental repair-v3-light wrapper for OfficialStyleSTID.

This file is diagnostic-only and does not alter frozen SRAF-ID v1/v2 behavior.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.residual_models_v3 import SRAFRepairFactorAblation, SRAFRepairV3Light
from src.models.strong_backbones import OfficialStyleSTID


class SRAFOfficialStyleSTIDWrapperV3Light(nn.Module):
    """Speed-only repair-v3-light wrapper with identity-feature bypass."""

    def __init__(
        self,
        sensors: int,
        backbone: OfficialStyleSTID,
        tod_profile: torch.Tensor,
        topk: int = 5,
        fusion_hidden_dim: int = 32,
        observed_input_blend: float = 0.5,
    ) -> None:
        super().__init__()
        self.repairer = SRAFRepairV3Light(
            sensors=sensors,
            tod_profile=tod_profile,
            topk=topk,
            fusion_hidden_dim=fusion_hidden_dim,
            observed_input_blend=observed_input_blend,
        )
        self.backbone = backbone

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError(f"Expected x [B,L,N,3], got {tuple(x.shape)}")
        speed = x[..., :1]
        identity = x[..., 1:]
        comps = self.repairer.repair_components(speed, identity, adjacency=adjacency, observed_mask=observed_mask)
        repaired_speed = comps["repaired_input"][..., :1]
        x_backbone = torch.cat([repaired_speed, identity], dim=-1)
        pred = self.backbone(x_backbone, adjacency=None)
        if return_components:
            return pred, {**comps, "repaired_input_speed": repaired_speed, "backbone_input": x_backbone, "identity_features": identity}
        return pred


class SRAFOfficialStyleSTIDWrapperFactorAblation(nn.Module):
    """Wrapper for controlled temporal/spatial/MLP/profile ablations."""

    def __init__(
        self,
        sensors: int,
        backbone: OfficialStyleSTID,
        tod_profile: torch.Tensor,
        temporal_mode: str = "basic",
        spatial_mode: str = "adjacency",
        fusion_mode: str = "alpha",
        use_profile: bool = False,
        topk: int = 5,
        fusion_hidden_dim: int = 16,
        fixed_profile_weight: float = 0.10,
        observed_input_blend: float = 0.5,
    ) -> None:
        super().__init__()
        self.repairer = SRAFRepairFactorAblation(
            sensors=sensors,
            tod_profile=tod_profile,
            temporal_mode=temporal_mode,
            spatial_mode=spatial_mode,
            fusion_mode=fusion_mode,
            use_profile=use_profile,
            topk=topk,
            hidden_dim=fusion_hidden_dim,
            fixed_profile_weight=fixed_profile_weight,
            observed_input_blend=observed_input_blend,
        )
        self.backbone = backbone

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError(f"Expected x [B,L,N,3], got {tuple(x.shape)}")
        speed = x[..., :1]
        identity = x[..., 1:]
        comps = self.repairer.repair_components(speed, identity, adjacency=adjacency, observed_mask=observed_mask)
        repaired_speed = comps["repaired_input"][..., :1]
        x_backbone = torch.cat([repaired_speed, identity], dim=-1)
        pred = self.backbone(x_backbone, adjacency=None)
        if return_components:
            return pred, {**comps, "repaired_input_speed": repaired_speed, "backbone_input": x_backbone, "identity_features": identity}
        return pred
