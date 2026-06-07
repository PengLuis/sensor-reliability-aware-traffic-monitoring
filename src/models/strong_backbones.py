"""Stronger clean backbones and SRAF-repair integration wrappers."""

from __future__ import annotations

import torch
from torch import nn

from src.models.residual_models import SRAFResidualGRU


class ResidualConvMLP(nn.Module):
    """Residual 1x1 Conv2d MLP block used by official-style STID."""

    def __init__(self, dim: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim, dim, kernel_size=(1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class STIDBackbone(nn.Module):
    """STID-style lightweight backbone with sensor/time identities and horizon decoder."""

    def __init__(
        self,
        sensors: int,
        input_length: int,
        input_features: int,
        horizon: int,
        hidden_dim: int = 128,
        sensor_embedding_dim: int = 16,
        horizon_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.sensors = sensors
        self.input_length = input_length
        self.input_features = input_features
        self.horizon = horizon
        flat_dim = input_length * input_features
        self.sensor_embedding = nn.Embedding(sensors, sensor_embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim + sensor_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.spatial_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.horizon_embedding = nn.Embedding(horizon, horizon_embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + horizon_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x [B,L,N,F], got {tuple(x.shape)}")
        bsz, length, sensors, features = x.shape
        if sensors != self.sensors or length != self.input_length or features != self.input_features:
            raise ValueError("Input dimensions do not match STIDBackbone configuration.")
        per_sensor = x.permute(0, 2, 1, 3).reshape(bsz, sensors, length * features)
        sensor_ids = torch.arange(sensors, device=x.device)
        sensor_emb = self.sensor_embedding(sensor_ids).unsqueeze(0).expand(bsz, -1, -1)
        latent = self.encoder(torch.cat([per_sensor, sensor_emb], dim=-1))
        if adjacency is None:
            adjacency = torch.eye(sensors, device=x.device, dtype=x.dtype)
        adj = adjacency.to(device=x.device, dtype=latent.dtype)
        denom = adj.sum(dim=-1).clamp_min(1.0e-6)
        latent_sp = torch.einsum("ij,bjh->bih", adj, latent) / denom.view(1, -1, 1)
        latent = self.spatial_fuse(torch.cat([latent, latent_sp], dim=-1))
        horizon_ids = torch.arange(self.horizon, device=x.device)
        h_emb = self.horizon_embedding(horizon_ids).view(1, 1, self.horizon, -1).expand(bsz, sensors, -1, -1)
        latent_h = latent.unsqueeze(2).expand(-1, -1, self.horizon, -1)
        pred = self.decoder(torch.cat([latent_h, h_emb], dim=-1))  # [B,N,H,1]
        return pred.permute(0, 2, 1, 3).contiguous()


class OfficialStyleSTID(nn.Module):
    """Official-style STID backbone with Conv2d embeddings and identity lookups.

    This is a local implementation of the architecture pattern, not a copied
    dependency. It intentionally ignores adjacency and target-horizon metadata.
    """

    def __init__(
        self,
        sensors: int,
        input_length: int,
        input_dim: int,
        horizon: int,
        embed_dim: int = 32,
        node_dim: int = 32,
        temp_dim_tid: int = 32,
        temp_dim_diw: int = 32,
        num_layers: int = 3,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        use_node: bool = True,
        use_time_in_day: bool = True,
        use_day_in_week: bool = True,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.sensors = sensors
        self.input_length = input_length
        self.input_dim = input_dim
        self.horizon = horizon
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.use_node = use_node
        self.use_time_in_day = use_time_in_day
        self.use_day_in_week = use_day_in_week

        self.time_series_emb_layer = nn.Conv2d(input_dim * input_length, embed_dim, kernel_size=(1, 1))
        self.node_emb = nn.Embedding(sensors, node_dim) if use_node else None
        self.time_in_day_emb = nn.Embedding(time_of_day_size, temp_dim_tid) if use_time_in_day else None
        self.day_in_week_emb = nn.Embedding(day_of_week_size, temp_dim_diw) if use_day_in_week else None
        hidden_dim = embed_dim
        if use_node:
            hidden_dim += node_dim
        if use_time_in_day:
            hidden_dim += temp_dim_tid
        if use_day_in_week:
            hidden_dim += temp_dim_diw
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(*(ResidualConvMLP(hidden_dim, dropout=dropout) for _ in range(num_layers)))
        self.regression_layer = nn.Conv2d(hidden_dim, horizon, kernel_size=(1, 1))

    def _derive_tod_index(self, x: torch.Tensor, tod_index: torch.Tensor | None) -> torch.Tensor:
        if tod_index is None:
            if x.shape[-1] <= 1:
                raise ValueError("OfficialStyleSTID requires tod_index or x[...,1] containing tod_norm in [0,1).")
            tod_norm = x[:, -1, :, 1]
            tod_index = torch.floor(tod_norm * self.time_of_day_size).long()
        if tod_index.ndim == 1:
            tod_index = tod_index[:, None].expand(-1, self.sensors)
        if tod_index.ndim != 2:
            raise ValueError(f"tod_index must have shape [B,N] or [B], got {tuple(tod_index.shape)}")
        return tod_index.to(device=x.device).clamp(0, self.time_of_day_size - 1)

    def _derive_dow_index(self, x: torch.Tensor, dow_index: torch.Tensor | None) -> torch.Tensor:
        if dow_index is None:
            if x.shape[-1] <= 2:
                raise ValueError("OfficialStyleSTID requires dow_index or x[...,2] containing dow_norm in [0,1).")
            dow_norm = x[:, -1, :, 2]
            dow_index = torch.floor(dow_norm * self.day_of_week_size).long()
        if dow_index.ndim == 1:
            dow_index = dow_index[:, None].expand(-1, self.sensors)
        if dow_index.ndim != 2:
            raise ValueError(f"dow_index must have shape [B,N] or [B], got {tuple(dow_index.shape)}")
        return dow_index.to(device=x.device).clamp(0, self.day_of_week_size - 1)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        tod_index: torch.Tensor | None = None,
        dow_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del adjacency
        if x.ndim != 4:
            raise ValueError(f"OfficialStyleSTID expected x [B,L,N,C], got {tuple(x.shape)}")
        bsz, length, sensors, channels = x.shape
        if length != self.input_length:
            raise ValueError(f"OfficialStyleSTID expected input length {self.input_length}, got {length}")
        if sensors != self.sensors:
            raise ValueError(f"OfficialStyleSTID expected {self.sensors} sensors, got {sensors}")
        if channels < self.input_dim:
            raise ValueError(f"OfficialStyleSTID expected at least {self.input_dim} channels, got {channels}")
        if not torch.isfinite(x).all():
            raise ValueError("OfficialStyleSTID received non-finite input values.")

        x_ts = x[..., : self.input_dim]
        x_ts = x_ts.permute(0, 1, 3, 2).reshape(bsz, length * self.input_dim, sensors, 1)
        embeddings = [self.time_series_emb_layer(x_ts)]

        if self.node_emb is not None:
            node_ids = torch.arange(sensors, device=x.device)
            node_emb = self.node_emb(node_ids).transpose(0, 1).view(1, -1, sensors, 1).expand(bsz, -1, -1, -1)
            embeddings.append(node_emb)

        if self.time_in_day_emb is not None:
            tid = self._derive_tod_index(x, tod_index)
            tid_emb = self.time_in_day_emb(tid).permute(0, 2, 1).unsqueeze(-1)
            embeddings.append(tid_emb)

        if self.day_in_week_emb is not None:
            diw = self._derive_dow_index(x, dow_index)
            diw_emb = self.day_in_week_emb(diw).permute(0, 2, 1).unsqueeze(-1)
            embeddings.append(diw_emb)

        hidden = torch.cat(embeddings, dim=1)
        if hidden.shape[1] != self.hidden_dim:
            raise RuntimeError(f"OfficialStyleSTID hidden dim mismatch: expected {self.hidden_dim}, got {hidden.shape[1]}")
        hidden = self.encoder(hidden)
        output = self.regression_layer(hidden)
        if output.shape != (bsz, self.horizon, sensors, 1):
            raise RuntimeError(f"OfficialStyleSTID output shape mismatch: got {tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise RuntimeError("OfficialStyleSTID produced non-finite outputs.")
        return output.contiguous()


class TCNTimeStrongBackbone(nn.Module):
    """Dilated TCN-time backbone with sensor/horizon identity embeddings."""

    def __init__(
        self,
        sensors: int,
        input_features: int,
        horizon: int,
        hidden_dim: int = 96,
        sensor_embedding_dim: int = 16,
        horizon_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.sensors = sensors
        self.input_features = input_features
        self.horizon = horizon
        self.sensor_embedding = nn.Embedding(sensors, sensor_embedding_dim)
        self.tcn = nn.Sequential(
            nn.Conv1d(input_features, hidden_dim, kernel_size=3, dilation=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=4, padding=8),
            nn.ReLU(),
        )
        self.horizon_embedding = nn.Embedding(horizon, horizon_embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + sensor_embedding_dim + horizon_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.spatial_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2 + sensor_embedding_dim, hidden_dim + sensor_embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x [B,L,N,F], got {tuple(x.shape)}")
        bsz, length, sensors, features = x.shape
        if sensors != self.sensors or features != self.input_features:
            raise ValueError("Input dimensions do not match TCNTimeStrongBackbone configuration.")
        per_sensor = x.permute(0, 2, 3, 1).reshape(bsz * sensors, features, length)
        latent_seq = self.tcn(per_sensor)[..., :length]
        latent = latent_seq[..., -1].reshape(bsz, sensors, -1)
        sensor_ids = torch.arange(sensors, device=x.device)
        s_emb = self.sensor_embedding(sensor_ids).unsqueeze(0).expand(bsz, -1, -1)
        if adjacency is None:
            adjacency = torch.eye(sensors, device=x.device, dtype=x.dtype)
        adj = adjacency.to(device=x.device, dtype=latent.dtype)
        denom = adj.sum(dim=-1).clamp_min(1.0e-6)
        latent_sp = torch.einsum("ij,bjh->bih", adj, latent) / denom.view(1, -1, 1)
        base = self.spatial_fuse(torch.cat([latent, latent_sp, s_emb], dim=-1))
        horizon_ids = torch.arange(self.horizon, device=x.device)
        h_emb = self.horizon_embedding(horizon_ids).view(1, 1, self.horizon, -1).expand(bsz, sensors, -1, -1)
        base_h = base.unsqueeze(2).expand(-1, -1, self.horizon, -1)
        pred = self.decoder(torch.cat([base_h, h_emb], dim=-1))
        return pred.permute(0, 2, 1, 3).contiguous()


class SRAFBackboneWrapper(nn.Module):
    """Prepend SRAF repair to a stronger backbone while keeping the backbone lightweight."""

    def __init__(
        self,
        sensors: int,
        input_features: int,
        horizon: int,
        repair_hidden_dim: int,
        repair_sensor_embedding_dim: int,
        backbone: nn.Module,
    ) -> None:
        super().__init__()
        self.repairer = SRAFResidualGRU(
            sensors=sensors,
            features=input_features,
            output_features=1,
            horizon=horizon,
            hidden_dim=repair_hidden_dim,
            sensor_embedding_dim=repair_sensor_embedding_dim,
            horizon_aware_decoder=False,
        )
        self.backbone = backbone

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components = self.repairer.repair_components(x, adjacency=adjacency)
        repaired_speed = components["repaired_input"][..., :1]
        x_backbone = torch.cat([repaired_speed, x[..., 1:]], dim=-1)
        pred = self.backbone(x_backbone, adjacency=adjacency)
        if return_components:
            return pred, components
        return pred


class SRAFOfficialStyleSTIDWrapper(nn.Module):
    """SRAF speed-channel repair followed by OfficialStyleSTID.

    The wrapper intentionally repairs only the normalized speed channel and
    preserves STID identity features exactly.
    """

    def __init__(
        self,
        sensors: int,
        horizon: int,
        repair_hidden_dim: int,
        repair_sensor_embedding_dim: int,
        backbone: OfficialStyleSTID,
        use_reliability_gate: bool = True,
        use_temporal_repair: bool = True,
        use_spatial_repair: bool = True,
    ) -> None:
        super().__init__()
        self.repairer = SRAFResidualGRU(
            sensors=sensors,
            features=1,
            output_features=1,
            horizon=horizon,
            hidden_dim=repair_hidden_dim,
            sensor_embedding_dim=repair_sensor_embedding_dim,
            use_reliability_gate=use_reliability_gate,
            use_temporal_repair=use_temporal_repair,
            use_spatial_repair=use_spatial_repair,
            hard_missing_gate=True,
            horizon_aware_decoder=False,
        )
        # The GRU forecast path is not used by this wrapper; keep only repair
        # parameters trainable so complexity reflects the active SRAF-STID path.
        for module in (self.repairer.encoder, self.repairer.sensor_embedding, self.repairer.decoder):
            for param in module.parameters():
                param.requires_grad_(False)
        self.backbone = backbone

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 4:
            raise ValueError(f"Expected x [B,L,N,3], got {tuple(x.shape)}")
        if x.shape[-1] != 3:
            raise ValueError(f"SRAFOfficialStyleSTIDWrapper expects 3 channels [speed,tod,dow], got {x.shape[-1]}")
        speed = x[..., :1]
        identities = x[..., 1:]
        identities_before = identities
        components = self.repairer.repair_components(speed, adjacency=adjacency, observed_mask=observed_mask)
        repaired_speed = components["repaired_input"][..., :1]
        x_backbone = torch.cat([repaired_speed, identities], dim=-1)
        if not torch.equal(identities, identities_before):
            raise RuntimeError("SRAF repair modified STID identity features.")
        pred = self.backbone(x_backbone, adjacency=None)
        if return_components:
            out = {
                **components,
                "repaired_input_speed": repaired_speed,
                "backbone_input": x_backbone,
                "identity_features": identities,
            }
            return pred, out
        return pred
