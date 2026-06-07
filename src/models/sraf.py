"""Lightweight SRAF model."""

from __future__ import annotations

import torch
from torch import nn


SRAF_MODULES = (
    "mask_aware_input_encoding",
    "lightweight_reliability_score_estimation",
    "temporal_repair",
    "adjacency_based_spatial_repair",
    "reliability_gated_forecasting_with_gru_or_tcn",
)


def list_sraf_modules() -> tuple[str, ...]:
    """Return required SRAF modules."""

    return SRAF_MODULES


class SRAFModel(nn.Module):
    """Sensor-Reliability-Aware Forecasting with GRU or TCN backbone."""

    def __init__(
        self,
        sensors: int,
        features: int,
        horizon: int,
        hidden_dim: int = 16,
        backbone: str = "GRU",
        alpha: float = 0.5,
        epsilon: float = 1.0e-6,
        no_reliability_gating: bool = False,
        no_mask_encoding: bool = False,
        no_temporal_repair: bool = False,
        no_spatial_repair: bool = False,
    ) -> None:
        super().__init__()
        if backbone not in {"GRU", "TCN"}:
            raise ValueError("backbone must be GRU or TCN")
        self.sensors = sensors
        self.features = features
        self.horizon = horizon
        self.backbone = backbone
        self.alpha = alpha
        self.epsilon = epsilon
        self.no_reliability_gating = no_reliability_gating
        self.no_mask_encoding = no_mask_encoding
        self.no_temporal_repair = no_temporal_repair
        self.no_spatial_repair = no_spatial_repair
        reliability_inputs = features + 1
        if not no_mask_encoding:
            reliability_inputs += 1
        self.reliability = nn.Sequential(
            nn.Linear(reliability_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        flat_dim = sensors * features
        output_dim = horizon * flat_dim
        if backbone == "GRU":
            self.forecaster = nn.GRU(flat_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, output_dim)
        else:
            self.forecaster = nn.Sequential(
                nn.Conv1d(flat_dim, hidden_dim, kernel_size=3, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2),
                nn.ReLU(),
            )
            self.head = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecast Y with shape [B,H,N,F] from X [B,L,N,F]."""

        if x.ndim != 4:
            raise ValueError(f"Expected x [B,L,N,F], got {tuple(x.shape)}")
        batch, length, sensors, features = x.shape
        if sensors != self.sensors or features != self.features:
            raise ValueError("Input sensor/feature dimensions do not match model")
        if observed_mask is None:
            observed_mask = torch.isfinite(x).to(x.dtype)
        else:
            observed_mask = observed_mask.to(x.dtype)
        x_filled = torch.nan_to_num(x, nan=0.0)
        x_temp = self._temporal_repair(x_filled) if not self.no_temporal_repair else x_filled
        if adjacency is None:
            adjacency = torch.eye(sensors, device=x.device, dtype=x.dtype)
        x_sp = self._spatial_repair(x_filled, adjacency) if not self.no_spatial_repair else x_filled
        x_rep = self.alpha * x_temp + (1.0 - self.alpha) * x_sp
        r = self._reliability_score(x_filled, observed_mask)
        if self.no_reliability_gating:
            x_tilde = x_rep
        else:
            x_tilde = r * x_filled + (1.0 - r) * x_rep
        flat = x_tilde.reshape(batch, length, sensors * features)
        if self.backbone == "GRU":
            _, hidden = self.forecaster(flat)
            out = self.head(hidden[-1])
        else:
            features_t = self.forecaster(flat.transpose(1, 2))[..., :length]
            pooled = features_t.mean(dim=-1)
            out = self.head(pooled)
        return out.reshape(batch, self.horizon, sensors, features)

    def _reliability_score(self, x: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(x[..., :1])
        delta[:, 1:] = torch.abs(x[:, 1:, :, :1] - x[:, :-1, :, :1])
        parts = [x, delta]
        if not self.no_mask_encoding:
            parts.append(observed_mask[..., :1])
        features = torch.cat(parts, dim=-1)
        return self.reliability(features)

    @staticmethod
    def _temporal_repair(x: torch.Tensor) -> torch.Tensor:
        repaired = x.clone()
        repaired[:, 1:] = x[:, :-1]
        return repaired

    def _spatial_repair(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        weights = adjacency.to(device=x.device, dtype=x.dtype)
        denom = weights.sum(dim=-1).clamp_min(self.epsilon)
        spatial = torch.einsum("ij,btjf->btif", weights, x)
        return spatial / denom.view(1, 1, -1, 1)
