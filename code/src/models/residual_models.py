"""Persistence-anchored residual models for traffic sensor forecasting."""

from __future__ import annotations

import torch
from torch import nn


class ResidualGRU(nn.Module):
    """Shared per-sensor GRU that predicts residuals around persistence."""

    def __init__(
        self,
        sensors: int,
        features: int,
        horizon: int,
        hidden_dim: int = 64,
        sensor_embedding_dim: int = 8,
        use_mask_channel: bool = True,
        output_features: int | None = None,
    ) -> None:
        super().__init__()
        self.sensors = sensors
        self.features = features
        self.output_features = features if output_features is None else output_features
        self.horizon = horizon
        self.use_mask_channel = use_mask_channel
        input_dim = features + (1 if use_mask_channel else 0)
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.sensor_embedding = nn.Embedding(sensors, sensor_embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + sensor_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * self.output_features),
        )
        self._init_residual_head()

    def _init_residual_head(self) -> None:
        last = self.decoder[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def repair_input(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del adjacency
        return basic_temporal_repair(x, observed_mask)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        repaired, mask = self.repair_input(x, adjacency=adjacency, observed_mask=observed_mask)
        pred, components = residual_decode(
            repaired=repaired,
            mask=mask,
            encoder=self.encoder,
            sensor_embedding=self.sensor_embedding,
            decoder=self.decoder,
            horizon=self.horizon,
            features=self.features,
            output_features=self.output_features,
            use_mask_channel=self.use_mask_channel,
        )
        if return_components:
            components["repaired_input"] = repaired
            return pred, components
        return pred


class SRAFResidualGRU(nn.Module):
    """SRAF repaired input followed by a persistence-anchored residual GRU."""

    def __init__(
        self,
        sensors: int,
        features: int,
        horizon: int,
        hidden_dim: int = 64,
        sensor_embedding_dim: int = 8,
        alpha: float = 0.5,
        epsilon: float = 1.0e-6,
        use_mask_channel: bool = True,
        reliability_bias_init: float | None = None,
        output_features: int | None = None,
        use_reliability_gate: bool = True,
        reliability_uses_mask: bool = True,
        use_temporal_repair: bool = True,
        use_spatial_repair: bool = True,
        hard_missing_gate: bool = False,
        enhanced_reliability_features: bool = False,
        stronger_stuck_features: bool = False,
        adaptive_adjacency_repair: bool = False,
        adaptive_adjacency_dim: int = 8,
        adaptive_adjacency_eta: float = 0.7,
        bidirectional_temporal_repair: bool = False,
        adaptive_repair_blending: bool = False,
        horizon_aware_decoder: bool = False,
        horizon_embedding_dim: int = 4,
        missing_severity_gate: bool = False,
        conditional_stuck_gate: bool = False,
        stuck_gate_beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.sensors = sensors
        self.features = features
        self.output_features = features if output_features is None else output_features
        self.horizon = horizon
        self.alpha = alpha
        self.epsilon = epsilon
        self.use_mask_channel = use_mask_channel
        self.use_reliability_gate = use_reliability_gate
        self.reliability_uses_mask = reliability_uses_mask
        self.use_temporal_repair = use_temporal_repair
        self.use_spatial_repair = use_spatial_repair
        self.hard_missing_gate = hard_missing_gate
        self.enhanced_reliability_features = enhanced_reliability_features
        self.stronger_stuck_features = stronger_stuck_features
        self.adaptive_adjacency_repair = adaptive_adjacency_repair
        self.adaptive_adjacency_eta = adaptive_adjacency_eta
        self.bidirectional_temporal_repair = bidirectional_temporal_repair
        self.adaptive_repair_blending = adaptive_repair_blending
        self.horizon_aware_decoder = horizon_aware_decoder
        self.missing_severity_gate = missing_severity_gate
        self.conditional_stuck_gate = conditional_stuck_gate
        self.stuck_gate_beta = stuck_gate_beta
        reliability_inputs = reliability_feature_dim(
            features,
            reliability_uses_mask,
            enhanced_reliability_features,
            stronger_stuck_features,
        )
        if adaptive_adjacency_repair:
            self.adaptive_adj_source = nn.Embedding(sensors, adaptive_adjacency_dim)
            self.adaptive_adj_target = nn.Embedding(sensors, adaptive_adjacency_dim)
        else:
            self.adaptive_adj_source = None
            self.adaptive_adj_target = None
        self.reliability = nn.Sequential(
            nn.Linear(reliability_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        if adaptive_repair_blending:
            self.alpha_gate = nn.Sequential(nn.Linear(reliability_inputs, 1), nn.Sigmoid())
        else:
            self.alpha_gate = None
        if reliability_bias_init is not None:
            final = self.reliability[2]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.constant_(final.bias, reliability_bias_init)
        if not use_reliability_gate:
            for param in self.reliability.parameters():
                param.requires_grad_(False)
        input_dim = features + (1 if use_mask_channel else 0)
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.sensor_embedding = nn.Embedding(sensors, sensor_embedding_dim)
        if horizon_aware_decoder:
            self.horizon_embedding = nn.Embedding(horizon, horizon_embedding_dim)
            decoder_input_dim = hidden_dim + sensor_embedding_dim + horizon_embedding_dim
            decoder_output_dim = self.output_features
        else:
            self.horizon_embedding = None
            decoder_input_dim = hidden_dim + sensor_embedding_dim
            decoder_output_dim = horizon * self.output_features
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, decoder_output_dim),
        )
        self._init_residual_head()

    def _init_residual_head(self) -> None:
        last = self.decoder[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def repair_components(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected x [B,L,N,F], got {tuple(x.shape)}")
        batch, length, sensors, features = x.shape
        del batch, length
        if sensors != self.sensors or features != self.features:
            raise ValueError("Input sensor/feature dimensions do not match model")
        mask = finite_mask(x, observed_mask)
        x_filled = torch.nan_to_num(x, nan=0.0)
        if self.bidirectional_temporal_repair:
            x_temp, _ = bidirectional_temporal_repair(x, mask)
        else:
            x_temp, _ = basic_temporal_repair(x, mask)
        if adjacency is None:
            adjacency = torch.eye(sensors, device=x.device, dtype=x.dtype)
        if self.adaptive_adjacency_repair:
            adjacency_for_repair = adaptive_adjacency_mix(
                adjacency,
                self.adaptive_adj_source,
                self.adaptive_adj_target,
                self.adaptive_adjacency_eta,
            )
        else:
            adjacency_for_repair = adjacency
        x_sp = spatial_repair(x_filled, adjacency_for_repair, self.epsilon)
        if self.use_temporal_repair and self.use_spatial_repair:
            rel_features = reliability_features(
                x_filled,
                mask,
                use_mask=self.reliability_uses_mask,
                enhanced=self.enhanced_reliability_features,
                stronger_stuck=self.stronger_stuck_features,
                spatial_estimate=x_sp,
            )
            if self.alpha_gate is None:
                alpha = torch.full_like(x_filled[..., :1], self.alpha)
            else:
                alpha = self.alpha_gate(rel_features)
                alpha = torch.where(mask[..., :1] < 0.5, torch.ones_like(alpha), alpha)
            x_rep = alpha * x_temp + (1.0 - alpha) * x_sp
        elif self.use_temporal_repair:
            rel_features = reliability_features(
                x_filled,
                mask,
                use_mask=self.reliability_uses_mask,
                enhanced=self.enhanced_reliability_features,
                stronger_stuck=self.stronger_stuck_features,
                spatial_estimate=x_sp,
            )
            alpha = torch.ones_like(x_filled[..., :1])
            x_rep = x_temp
        elif self.use_spatial_repair:
            rel_features = reliability_features(
                x_filled,
                mask,
                use_mask=self.reliability_uses_mask,
                enhanced=self.enhanced_reliability_features,
                stronger_stuck=self.stronger_stuck_features,
                spatial_estimate=x_sp,
            )
            alpha = torch.zeros_like(x_filled[..., :1])
            x_rep = x_sp
        else:
            rel_features = reliability_features(
                x_filled,
                mask,
                use_mask=self.reliability_uses_mask,
                enhanced=self.enhanced_reliability_features,
                stronger_stuck=self.stronger_stuck_features,
                spatial_estimate=x_sp,
            )
            alpha = torch.full_like(x_filled[..., :1], self.alpha)
            x_rep = x_filled
        if self.use_reliability_gate:
            reliability = self.reliability(rel_features)
            if self.hard_missing_gate:
                reliability = reliability * mask[..., :1]
        else:
            reliability = torch.full_like(x_filled[..., :1], 0.5)
        if self.missing_severity_gate:
            missing_ratio = 1.0 - mask[..., :1].mean(dim=1, keepdim=True)
            reliability = torch.where(
                mask[..., :1] < 0.5,
                reliability * (1.0 - 0.3 * missing_ratio),
                reliability + (1.0 - reliability) * 0.1 * (1.0 - missing_ratio),
            )
            reliability = reliability.clamp(0.0, 1.0)
        if self.conditional_stuck_gate:
            stuck_score = conditional_stuck_score(x_filled, mask, x_sp)
            s_final = mask[..., :1] * stuck_score
            reliability = reliability * (1.0 - self.stuck_gate_beta * s_final)
        else:
            stuck_score = torch.zeros_like(x_filled[..., :1])
        repaired = reliability * x_filled + (1.0 - reliability) * x_rep
        return {
            "repaired_input": repaired,
            "mask": mask,
            "reliability": reliability,
            "temporal_repair": x_temp,
            "spatial_repair": x_sp,
            "repair_blend": x_rep,
            "alpha": alpha,
            "stuck_score": stuck_score,
        }

    def repair_input(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        components = self.repair_components(x, adjacency=adjacency, observed_mask=observed_mask)
        repaired = components["repaired_input"]
        mask = components["mask"]
        reliability = components["reliability"]
        return repaired, mask, reliability

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        repair_components = self.repair_components(x, adjacency=adjacency, observed_mask=observed_mask)
        repaired = repair_components["repaired_input"]
        mask = repair_components["mask"]
        pred, components = residual_decode(
            repaired=repaired,
            mask=mask,
            encoder=self.encoder,
            sensor_embedding=self.sensor_embedding,
            decoder=self.decoder,
            horizon=self.horizon,
            features=self.features,
            output_features=self.output_features,
            use_mask_channel=self.use_mask_channel,
            horizon_embedding=self.horizon_embedding,
        )
        if return_components:
            components.update(repair_components)
            return pred, components
        return pred


def finite_mask(x: torch.Tensor, observed_mask: torch.Tensor | None = None) -> torch.Tensor:
    if observed_mask is None:
        return torch.isfinite(x).to(x.dtype)
    return observed_mask.to(device=x.device, dtype=x.dtype)


def basic_temporal_repair(
    x: torch.Tensor,
    observed_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = finite_mask(x, observed_mask)
    filled = torch.nan_to_num(x, nan=0.0)
    repaired = filled.clone()
    last = filled[:, 0]
    for step in range(filled.shape[1]):
        current_mask = mask[:, step]
        last = torch.where(current_mask > 0.5, filled[:, step], last)
        repaired[:, step] = last
    return repaired, mask


def bidirectional_temporal_repair(
    x: torch.Tensor,
    observed_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = finite_mask(x, observed_mask)
    forward, _ = basic_temporal_repair(x, mask)
    reverse_repaired, _ = basic_temporal_repair(torch.flip(x, dims=[1]), torch.flip(mask, dims=[1]))
    backward = torch.flip(reverse_repaired, dims=[1])
    observed = torch.nan_to_num(x, nan=0.0)
    repaired = 0.5 * (forward + backward)
    repaired = torch.where(mask > 0.5, observed, repaired)
    return repaired, mask


def adaptive_adjacency_mix(
    adjacency: torch.Tensor,
    source_embedding: nn.Embedding | None,
    target_embedding: nn.Embedding | None,
    eta: float,
) -> torch.Tensor:
    if source_embedding is None or target_embedding is None:
        return adjacency
    sensors = adjacency.shape[0]
    ids = torch.arange(sensors, device=adjacency.device)
    source = source_embedding(ids)
    target = target_embedding(ids)
    adaptive_logits = torch.relu(source @ target.transpose(0, 1))
    adaptive = torch.softmax(adaptive_logits, dim=-1)
    physical = adjacency.to(device=adaptive.device, dtype=adaptive.dtype)
    eta_clamped = max(0.0, min(1.0, float(eta)))
    return eta_clamped * physical + (1.0 - eta_clamped) * adaptive


def spatial_repair(x: torch.Tensor, adjacency: torch.Tensor, epsilon: float) -> torch.Tensor:
    weights = adjacency.to(device=x.device, dtype=x.dtype)
    denom = weights.sum(dim=-1).clamp_min(epsilon)
    spatial = torch.einsum("ij,btjf->btif", weights, x)
    return spatial / denom.view(1, 1, -1, 1)


def conditional_stuck_score(
    x: torch.Tensor,
    observed_mask: torch.Tensor,
    spatial_estimate: torch.Tensor,
) -> torch.Tensor:
    speed = x[..., :1]
    abs_delta = torch.zeros_like(speed)
    abs_delta[:, 1:] = torch.abs(speed[:, 1:] - speed[:, :-1])
    mean_abs_diff = torch.zeros_like(speed)
    for step in range(speed.shape[1]):
        start = max(0, step - 3)
        mean_abs_diff[:, step] = abs_delta[:, start : step + 1].mean(dim=1)
    low_change_score = torch.exp(-20.0 * mean_abs_diff.clamp_min(0.0))
    spatial_disagreement = torch.abs(speed - spatial_estimate[..., :1])
    disagreement_score = torch.sigmoid(4.0 * (spatial_disagreement - 0.1))
    observed = observed_mask[..., :1]
    return (low_change_score * disagreement_score * observed).clamp(0.0, 1.0)



def reliability_feature_dim(
    features: int,
    use_mask: bool = True,
    enhanced: bool = False,
    stronger_stuck: bool = False,
) -> int:
    dim = features + 1
    if use_mask:
        dim += 1
    if enhanced:
        dim += 3
    if stronger_stuck:
        dim += 4
    return dim


def reliability_features(
    x: torch.Tensor,
    observed_mask: torch.Tensor,
    use_mask: bool = True,
    enhanced: bool = False,
    stronger_stuck: bool = False,
    spatial_estimate: torch.Tensor | None = None,
) -> torch.Tensor:
    delta = torch.zeros_like(x[..., :1])
    delta[:, 1:] = torch.abs(x[:, 1:, :, :1] - x[:, :-1, :, :1])
    parts = [x, delta]
    if use_mask:
        parts.append(observed_mask[..., :1])
    if enhanced:
        speed = x[..., :1]
        local_mean = torch.zeros_like(speed)
        local_var = torch.zeros_like(speed)
        for step in range(speed.shape[1]):
            start = max(0, step - 2)
            window = speed[:, start : step + 1]
            local_mean[:, step] = window.mean(dim=1)
            local_var[:, step] = window.var(dim=1, unbiased=False)
        rolling_deviation = torch.abs(speed - local_mean)
        low_variance_indicator = torch.exp(-10.0 * local_var.clamp_min(0.0))
        if spatial_estimate is None:
            spatial_disagreement = torch.zeros_like(speed)
        else:
            spatial_disagreement = torch.abs(speed - spatial_estimate[..., :1])
        parts.extend([rolling_deviation, low_variance_indicator, spatial_disagreement])
    if stronger_stuck:
        speed = x[..., :1]
        abs_delta = torch.zeros_like(speed)
        abs_delta[:, 1:] = torch.abs(speed[:, 1:] - speed[:, :-1])
        rolling_variance = torch.zeros_like(speed)
        mean_abs_temporal_diff = torch.zeros_like(speed)
        low_change_fraction = torch.zeros_like(speed)
        for step in range(speed.shape[1]):
            start = max(0, step - 3)
            window = speed[:, start : step + 1]
            delta_window = abs_delta[:, start : step + 1]
            rolling_variance[:, step] = window.var(dim=1, unbiased=False)
            mean_abs_temporal_diff[:, step] = delta_window.mean(dim=1)
            low_change_fraction[:, step] = (delta_window < 0.02).to(speed.dtype).mean(dim=1)
        if spatial_estimate is None:
            neighbor_trend_disagreement = torch.zeros_like(speed)
        else:
            spatial_delta = torch.zeros_like(speed)
            spatial_delta[:, 1:] = torch.abs(spatial_estimate[:, 1:, :, :1] - spatial_estimate[:, :-1, :, :1])
            neighbor_trend_disagreement = torch.abs(abs_delta - spatial_delta)
        parts.extend([rolling_variance, mean_abs_temporal_diff, low_change_fraction, neighbor_trend_disagreement])
    return torch.cat(parts, dim=-1)


def residual_decode(
    repaired: torch.Tensor,
    mask: torch.Tensor,
    encoder: nn.GRU,
    sensor_embedding: nn.Embedding,
    decoder: nn.Module,
    horizon: int,
    features: int,
    output_features: int,
    use_mask_channel: bool,
    horizon_embedding: nn.Embedding | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if repaired.ndim != 4:
        raise ValueError(f"Expected repaired [B,L,N,F], got {tuple(repaired.shape)}")
    batch, length, sensors, input_features = repaired.shape
    if input_features != features:
        raise ValueError("Feature dimension mismatch")
    if output_features > features:
        raise ValueError("output_features cannot exceed input features")
    base = repaired[:, -1, :, :output_features]
    parts = [repaired]
    if use_mask_channel:
        parts.append(mask[..., :1])
    encoded_input = torch.cat(parts, dim=-1)
    per_sensor = encoded_input.permute(0, 2, 1, 3).reshape(batch * sensors, length, -1)
    _, hidden = encoder(per_sensor)
    hidden_last = hidden[-1].reshape(batch, sensors, -1)
    sensor_ids = torch.arange(sensors, device=repaired.device)
    emb = sensor_embedding(sensor_ids).unsqueeze(0).expand(batch, -1, -1)
    if horizon_embedding is None:
        decoded = decoder(torch.cat([hidden_last, emb], dim=-1))
        residual_delta = decoded.reshape(batch, sensors, horizon, output_features).permute(0, 2, 1, 3)
    else:
        horizon_ids = torch.arange(horizon, device=repaired.device)
        horizon_emb = horizon_embedding(horizon_ids).view(1, 1, horizon, -1).expand(batch, sensors, -1, -1)
        hidden_expand = hidden_last.unsqueeze(2).expand(-1, -1, horizon, -1)
        emb_expand = emb.unsqueeze(2).expand(-1, -1, horizon, -1)
        decoded = decoder(torch.cat([hidden_expand, emb_expand, horizon_emb], dim=-1))
        residual_delta = decoded.permute(0, 2, 1, 3)
    prediction = base.unsqueeze(1).expand(-1, horizon, -1, -1) + residual_delta
    return prediction, {"base": base, "residual_delta": residual_delta}
