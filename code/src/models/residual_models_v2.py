"""SRAF v2 repair modules.

This file is intentionally separate from v1 implementations so existing behavior
is unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.residual_models import basic_temporal_repair, bidirectional_temporal_repair, finite_mask, spatial_repair


def _local_variance(x: torch.Tensor, window: int = 3) -> torch.Tensor:
    if window <= 1:
        return torch.zeros_like(x)
    pad = window // 2
    xp = x.permute(0, 2, 3, 1)  # B,N,F,L
    mean = nn.functional.avg_pool1d(
        nn.functional.pad(xp.reshape(-1, 1, xp.shape[-1]), (pad, pad), mode="replicate"),
        kernel_size=window,
        stride=1,
    ).reshape(xp.shape)
    sq = nn.functional.avg_pool1d(
        nn.functional.pad((xp**2).reshape(-1, 1, xp.shape[-1]), (pad, pad), mode="replicate"),
        kernel_size=window,
        stride=1,
    ).reshape(xp.shape)
    var = (sq - mean**2).clamp_min(0.0)
    return var.permute(0, 3, 1, 2)


def _flatness(x: torch.Tensor) -> torch.Tensor:
    left = torch.roll(x, shifts=1, dims=1)
    right = torch.roll(x, shifts=-1, dims=1)
    return (torch.abs(x - left) + torch.abs(right - x)) * 0.5


def _stuck_duration(mask: torch.Tensor, x_filled: torch.Tensor) -> torch.Tensor:
    # Approximate run-length of repeated values while observed.
    b, l, n, f = x_filled.shape
    out = torch.zeros_like(x_filled)
    run = torch.zeros((b, n, f), device=x_filled.device, dtype=x_filled.dtype)
    for t in range(1, l):
        same = (torch.abs(x_filled[:, t] - x_filled[:, t - 1]) < 1.0e-6).to(x_filled.dtype)
        obs = mask[:, t]
        run = (run + 1.0) * same * obs
        out[:, t] = run
    norm = out / max(1.0, float(l - 1))
    return norm


def reliability_features_v2(
    x_filled: torch.Tensor,
    mask: torch.Tensor,
    x_temp: torch.Tensor,
    x_sp: torch.Tensor,
    include_stuck_features: bool = False,
    include_neighbor_disagreement: bool = False,
    include_second_delta: bool = True,
    include_flatness: bool = True,
    include_repair_disagreement: bool = True,
) -> tuple[torch.Tensor, list[str]]:
    delta1 = torch.zeros_like(x_filled)
    delta1[:, 1:] = torch.abs(x_filled[:, 1:] - x_filled[:, :-1])
    delta2 = torch.zeros_like(x_filled)
    delta2[:, 2:] = torch.abs((x_filled[:, 2:] - x_filled[:, 1:-1]) - (x_filled[:, 1:-1] - x_filled[:, :-2]))
    var_local = _local_variance(x_filled, window=3)
    d_x_sp = torch.abs(x_filled - x_sp)
    d_x_temp = torch.abs(x_filled - x_temp)
    d_temp_sp = torch.abs(x_temp - x_sp)
    flatness = _flatness(x_filled)
    speed_mag = torch.abs(x_filled)
    feats = [x_filled, mask, delta1, var_local, speed_mag]
    names = ["x_filled", "observed_mask", "abs_delta1", "local_var", "speed_magnitude"]
    if include_second_delta:
        feats.append(delta2)
        names.append("abs_delta2")
    if include_repair_disagreement:
        feats.extend([d_x_sp, d_x_temp, d_temp_sp])
        names.extend(["abs_x_minus_xsp", "abs_x_minus_xtemp", "abs_xtemp_minus_xsp"])
    if include_flatness:
        feats.append(flatness)
        names.append("local_flatness")
    if include_stuck_features:
        stuck = _stuck_duration(mask, x_filled)
        feats.append(stuck)
        names.append("stuck_duration_estimate")
    if include_neighbor_disagreement:
        # Lightweight proxy without new graph ops: disagreement between temporal and spatial candidates.
        feats.append(d_temp_sp)
        names.append("neighbor_trend_disagreement_proxy")
    return torch.cat(feats, dim=-1), names


class SRAFResidualGRUV2(nn.Module):
    """SRAF v2 repairer with adaptive alpha and richer reliability features."""

    def __init__(
        self,
        sensors: int,
        hidden_dim: int = 32,
        alpha_hidden_dim: int = 16,
        alpha_adaptive: bool = True,
        use_reliability_gate: bool = True,
        use_temporal_repair: bool = True,
        use_spatial_repair: bool = True,
        alpha_fixed: float = 0.5,
        epsilon: float = 1.0e-6,
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
        self.sensors = sensors
        self.hidden_dim = hidden_dim
        self.alpha_hidden_dim = alpha_hidden_dim
        self.alpha_adaptive = alpha_adaptive
        self.use_reliability_gate = use_reliability_gate
        self.use_temporal_repair = use_temporal_repair
        self.use_spatial_repair = use_spatial_repair
        self.alpha_fixed = alpha_fixed
        self.epsilon = epsilon
        self.include_stuck_features = include_stuck_features
        self.include_neighbor_disagreement = include_neighbor_disagreement
        self.include_second_delta = include_second_delta
        self.include_flatness = include_flatness
        self.include_repair_disagreement = include_repair_disagreement
        self.use_base_features_for_reliability = use_base_features_for_reliability
        self.residual_clamp_k = residual_clamp_k
        self.safe_fallback_enable = safe_fallback_enable
        self.safe_fallback_eta = safe_fallback_eta
        self.safe_fallback_uncertainty_threshold = safe_fallback_uncertainty_threshold
        self.use_temporal_attention_candidate = use_temporal_attention_candidate
        self.temporal_attention_hidden_dim = temporal_attention_hidden_dim
        self.use_bidirectional_temporal_candidate = use_bidirectional_temporal_candidate
        self.use_light_graph_message_candidate = use_light_graph_message_candidate
        self.use_candidate_softmax_fusion = use_candidate_softmax_fusion
        self.stuck_fallback_enable = stuck_fallback_enable
        self.stuck_duration_threshold = stuck_duration_threshold
        self.spatial_disagreement_threshold = spatial_disagreement_threshold
        self.stuck_fallback_eta = stuck_fallback_eta

        # Feature dims are computed from builder.
        self._feature_dim_full = len(self.feature_names_full)
        self._feature_dim_rel = len(self.feature_names_reliability)
        self.reliability = nn.Sequential(
            nn.Linear(self._feature_dim_rel, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        if alpha_adaptive:
            self.alpha_mlp = nn.Sequential(
                nn.Linear(self._feature_dim_full, alpha_hidden_dim),
                nn.ReLU(),
                nn.Linear(alpha_hidden_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.alpha_mlp = None

        if self.use_temporal_attention_candidate:
            h = temporal_attention_hidden_dim
            self.attn_q = nn.Linear(1, h)
            self.attn_k = nn.Linear(1, h)
            self.attn_v = nn.Linear(1, h)
            self.attn_o = nn.Linear(h, 1)
        else:
            self.attn_q = None
            self.attn_k = None
            self.attn_v = None
            self.attn_o = None
        if self.use_light_graph_message_candidate:
            self.graph_message_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.graph_message_scale = None
        self.candidate_weight_mlp: nn.Linear | None = None

    @property
    def feature_names(self) -> list[str]:
        return self.feature_names_reliability

    @property
    def feature_names_full(self) -> list[str]:
        _, names = reliability_features_v2(
            torch.zeros((1, 3, 2, 1)),
            torch.ones((1, 3, 2, 1)),
            torch.zeros((1, 3, 2, 1)),
            torch.zeros((1, 3, 2, 1)),
            include_stuck_features=self.include_stuck_features,
            include_neighbor_disagreement=self.include_neighbor_disagreement,
            include_second_delta=self.include_second_delta,
            include_flatness=self.include_flatness,
            include_repair_disagreement=self.include_repair_disagreement,
        )
        return names

    @property
    def feature_names_reliability(self) -> list[str]:
        if not self.use_base_features_for_reliability:
            return self.feature_names_full
        return ["x_filled", "observed_mask", "abs_delta1"]

    def repair_components(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected x [B,L,N,F], got {tuple(x.shape)}")
        mask = finite_mask(x, observed_mask=observed_mask)[..., :1]
        x_filled = torch.nan_to_num(x, nan=0.0)
        adjacency_eff = adjacency
        if adjacency_eff is None:
            adjacency_eff = torch.eye(self.sensors, dtype=x_filled.dtype, device=x_filled.device)
        if self.use_temporal_repair:
            x_temp, _ = basic_temporal_repair(x, mask)
        else:
            x_temp = x_filled
        if self.use_spatial_repair:
            x_sp = spatial_repair(x_filled, adjacency_eff, epsilon=self.epsilon)
        else:
            x_sp = x_filled
        x_bidir = None
        if self.use_bidirectional_temporal_candidate:
            x_bidir, _ = bidirectional_temporal_repair(x, mask)
        x_attn = None
        if self.use_temporal_attention_candidate and self.attn_q is not None and self.attn_k is not None and self.attn_v is not None and self.attn_o is not None:
            # [B,L,N,1] -> [B*N,L,1]
            bln = x_filled.permute(0, 2, 1, 3).reshape(-1, x_filled.shape[1], 1)
            q = self.attn_q(bln)
            k = self.attn_k(bln)
            v = self.attn_v(bln)
            scale = float(self.temporal_attention_hidden_dim) ** -0.5
            score = torch.matmul(q, k.transpose(1, 2)) * scale
            obs = mask.permute(0, 2, 1, 3).reshape(-1, x_filled.shape[1], 1)
            obs2 = obs.transpose(1, 2)
            score = score.masked_fill(obs2 < 0.5, -1.0e4)
            w = torch.softmax(score, dim=-1)
            out = torch.matmul(w, v)
            out = self.attn_o(out)
            x_attn = out.reshape(x_filled.shape[0], x_filled.shape[2], x_filled.shape[1], 1).permute(0, 2, 1, 3)
        x_gmp = None
        if self.use_light_graph_message_candidate and self.graph_message_scale is not None:
            denom = adjacency_eff.sum(dim=1, keepdim=True).clamp_min(self.epsilon).view(1, 1, self.sensors, 1)
            x_gmp = torch.einsum("ij,btjf->btif", adjacency_eff, x_filled) / denom
            x_gmp = self.graph_message_scale * x_gmp

        feats_full, feat_names = reliability_features_v2(
            x_filled,
            mask,
            x_temp,
            x_sp,
            include_stuck_features=self.include_stuck_features,
            include_neighbor_disagreement=self.include_neighbor_disagreement,
            include_second_delta=self.include_second_delta,
            include_flatness=self.include_flatness,
            include_repair_disagreement=self.include_repair_disagreement,
        )
        if self.use_base_features_for_reliability:
            feats_rel = torch.cat(
                [
                    x_filled,
                    mask,
                    torch.abs(torch.cat([torch.zeros_like(x_filled[:, :1]), x_filled[:, 1:] - x_filled[:, :-1]], dim=1)),
                ],
                dim=-1,
            )
        else:
            feats_rel = feats_full
        candidates = [x_temp, x_sp]
        candidate_names = ["temporal", "spatial"]
        if x_attn is not None:
            candidates.append(x_attn)
            candidate_names.append("temporal_attention")
        if x_bidir is not None:
            candidates.append(x_bidir)
            candidate_names.append("bidirectional_temporal")
        if x_gmp is not None:
            candidates.append(x_gmp)
            candidate_names.append("graph_message")

        if self.use_candidate_softmax_fusion and len(candidates) >= 3:
            if self.candidate_weight_mlp is None or self.candidate_weight_mlp.out_features != len(candidates):
                self.candidate_weight_mlp = nn.Linear(self._feature_dim_full, len(candidates)).to(x_filled.device)
            logits = self.candidate_weight_mlp(feats_full)
            weight = torch.softmax(logits, dim=-1)
            x_rep = torch.zeros_like(x_filled)
            for idx, cand in enumerate(candidates):
                x_rep = x_rep + weight[..., idx : idx + 1] * cand
            alpha = weight[..., 0:1]  # compatibility alias
            candidate_weights = weight
        else:
            if self.alpha_mlp is not None and self.use_temporal_repair and self.use_spatial_repair:
                alpha = self.alpha_mlp(feats_full)
            else:
                alpha = torch.full_like(x_filled[..., :1], self.alpha_fixed)
            x_rep = alpha * x_temp + (1.0 - alpha) * x_sp
            candidate_weights = None

        if self.use_reliability_gate:
            reliability = self.reliability(feats_rel)
            reliability = reliability * mask
        else:
            reliability = torch.full_like(x_filled[..., :1], 0.5) * mask
        repaired = reliability * x_filled + (1.0 - reliability) * x_rep

        if self.residual_clamp_k is not None:
            delta = repaired - x_filled
            delta = torch.clamp(delta, min=-self.residual_clamp_k, max=self.residual_clamp_k)
            repaired = x_filled + delta

        if self.safe_fallback_enable:
            # Uncertainty is high when reliability is close to 0.5 or alpha is close to 0.5.
            rel_unc = 1.0 - torch.abs(2.0 * reliability - 1.0)
            alpha_unc = 1.0 - torch.abs(2.0 * alpha - 1.0)
            uncertain = ((rel_unc + alpha_unc) * 0.5) > self.safe_fallback_uncertainty_threshold
            v1_like = 0.5 * x_temp + 0.5 * x_sp
            safe = self.safe_fallback_eta * repaired + (1.0 - self.safe_fallback_eta) * v1_like
            repaired = torch.where(uncertain, safe, repaired)

        if self.stuck_fallback_enable:
            # Trigger fallback on likely stuck positions with high temporal-spatial disagreement.
            stuck_feat = _stuck_duration(mask, x_filled)
            disagreement = torch.abs(x_temp - x_sp)
            cond = (stuck_feat > self.stuck_duration_threshold) & (disagreement > self.spatial_disagreement_threshold)
            v1_like = 0.5 * x_temp + 0.5 * x_sp
            safe = self.stuck_fallback_eta * repaired + (1.0 - self.stuck_fallback_eta) * v1_like
            repaired = torch.where(cond, safe, repaired)

        return {
            "repaired_input": repaired,
            "mask": mask,
            "reliability": reliability,
            "temporal_repair": x_temp,
            "spatial_repair": x_sp,
            "repair_blend": x_rep,
            "alpha": alpha,
            "x_filled": x_filled,
            "feature_names": feat_names,
            "feature_names_reliability": self.feature_names_reliability,
            "candidate_names": candidate_names,
            "candidate_weights": candidate_weights,
            "temporal_attention_repair": x_attn,
            "bidirectional_temporal_repair": x_bidir,
            "graph_message_repair": x_gmp,
        }
