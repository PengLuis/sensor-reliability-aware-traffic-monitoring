"""Stable public entry point for the SRAF-ID paper model.

The formal artifact was historically implemented through the factor-ablation
wrapper with the fixed paper configuration below.  This module gives users a
paper-facing class name without changing the frozen implementation or reported
results.
"""

from __future__ import annotations

import torch

from src.models.strong_backbones import OfficialStyleSTID
from src.models.strong_backbones_v3 import SRAFOfficialStyleSTIDWrapperFactorAblation


class SRAFID(SRAFOfficialStyleSTIDWrapperFactorAblation):
    """Paper-facing SRAF-ID model with the frozen formal configuration.

    Configuration:
    - basic same-sensor temporal repair;
    - supplied-adjacency spatial repair;
    - learned two-way softmax fusion;
    - no time-of-day profile candidate;
    - no additional learned reliability gate;
    - fixed observed-input blend of 0.5 by default.
    """

    def __init__(
        self,
        sensors: int,
        backbone: OfficialStyleSTID,
        tod_profile: torch.Tensor,
        topk: int = 5,
        fusion_hidden_dim: int = 16,
        observed_input_blend: float = 0.5,
    ) -> None:
        super().__init__(
            sensors=sensors,
            backbone=backbone,
            tod_profile=tod_profile,
            temporal_mode="basic",
            spatial_mode="adjacency",
            fusion_mode="softmax",
            use_profile=False,
            topk=topk,
            fusion_hidden_dim=fusion_hidden_dim,
            observed_input_blend=observed_input_blend,
        )


__all__ = ["SRAFID"]
