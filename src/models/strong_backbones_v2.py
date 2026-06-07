"""SRAF-ID-v2 wrapper on top of OfficialStyleSTID backbone."""

from __future__ import annotations

import torch
from torch import nn

from src.models.residual_models_v2 import SRAFResidualGRUV2
from src.models.strong_backbones import OfficialStyleSTID


class SRAFOfficialStyleSTIDWrapperV2(nn.Module):
    """Speed-only repair wrapper with adaptive alpha and rich reliability features."""

    def __init__(
        self,
        sensors: int,
        backbone: OfficialStyleSTID,
        rel_hidden_dim: int = 32,
        alpha_hidden_dim: int = 16,
        alpha_adaptive: bool = True,
        use_reliability_gate: bool = True,
        include_stuck_features: bool = False,
        include_neighbor_disagreement: bool = False,
        include_second_delta: bool = True,
        include_flatness: bool = True,
        include_repair_disagreement: bool = True,
        use_base_features_for_reliability: bool = False,
        residual_clamp_k: float | None = None,
        safe_fallback_enable: bool = False,
        safe_fallback_eta: float = 0.5,
        safe_fallback_uncertainty_threshold: float = 0.25,
        use_temporal_attention_candidate: bool = False,
        temporal_attention_hidden_dim: int = 16,
        use_bidirectional_temporal_candidate: bool = False,
        use_light_graph_message_candidate: bool = False,
        use_candidate_softmax_fusion: bool = False,
        stuck_fallback_enable: bool = False,
        stuck_duration_threshold: float = 0.4,
        spatial_disagreement_threshold: float = 0.2,
        stuck_fallback_eta: float = 0.5,
    ) -> None:
        super().__init__()
        self.repairer = SRAFResidualGRUV2(
            sensors=sensors,
            hidden_dim=rel_hidden_dim,
            alpha_hidden_dim=alpha_hidden_dim,
            alpha_adaptive=alpha_adaptive,
            use_reliability_gate=use_reliability_gate,
            include_stuck_features=include_stuck_features,
            include_neighbor_disagreement=include_neighbor_disagreement,
            include_second_delta=include_second_delta,
            include_flatness=include_flatness,
            include_repair_disagreement=include_repair_disagreement,
            use_base_features_for_reliability=use_base_features_for_reliability,
            residual_clamp_k=residual_clamp_k,
            safe_fallback_enable=safe_fallback_enable,
            safe_fallback_eta=safe_fallback_eta,
            safe_fallback_uncertainty_threshold=safe_fallback_uncertainty_threshold,
            use_temporal_attention_candidate=use_temporal_attention_candidate,
            temporal_attention_hidden_dim=temporal_attention_hidden_dim,
            use_bidirectional_temporal_candidate=use_bidirectional_temporal_candidate,
            use_light_graph_message_candidate=use_light_graph_message_candidate,
            use_candidate_softmax_fusion=use_candidate_softmax_fusion,
            stuck_fallback_enable=stuck_fallback_enable,
            stuck_duration_threshold=stuck_duration_threshold,
            spatial_disagreement_threshold=spatial_disagreement_threshold,
            stuck_fallback_eta=stuck_fallback_eta,
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
        identity_before = identity
        comps = self.repairer.repair_components(speed, adjacency=adjacency, observed_mask=observed_mask)
        repaired_speed = comps["repaired_input"][..., :1]
        x_backbone = torch.cat([repaired_speed, identity], dim=-1)
        if not torch.equal(identity, identity_before):
            raise RuntimeError("SRAF-ID-v2 modified identity features.")
        pred = self.backbone(x_backbone, adjacency=None)
        if return_components:
            out = {
                **comps,
                "repaired_input_speed": repaired_speed,
                "backbone_input": x_backbone,
                "identity_features": identity,
            }
            return pred, out
        return pred
