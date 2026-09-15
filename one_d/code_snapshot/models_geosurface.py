# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
from torch import nn


class GeoSurfaceMLP(nn.Module):
    """Query-time engineering surface-depth surrogate scaffold.

    ``query`` is expected to contain x, y, z, t, depth, sdf, thickness and any
    local descriptors. ``process`` is [B, 3, Nt]. ``geometry_params`` and
    ``material`` are per-case vectors.
    """

    def __init__(
        self,
        query_channels: int,
        geometry_channels: int,
        material_channels: int,
        process_channels: int = 3,
        descriptor_channels: int = 0,
        shape_channels: int = 0,
        latent: int = 96,
    ) -> None:
        super().__init__()
        self.descriptor_channels = int(descriptor_channels)
        self.shape_channels = int(shape_channels)
        self.process_encoder = nn.Sequential(
            nn.Conv1d(process_channels, latent // 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(latent // 2, latent // 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.case_encoder = nn.Sequential(
            nn.Linear(
                geometry_channels + material_channels + descriptor_channels + shape_channels + latent // 2,
                latent,
            ),
            nn.GELU(),
            nn.Linear(latent, latent),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(query_channels + latent, latent),
            nn.GELU(),
            nn.Linear(latent, latent),
            nn.GELU(),
            nn.Linear(latent, 1),
        )

    def forward(
        self,
        query: torch.Tensor,
        process: torch.Tensor,
        geometry_params: torch.Tensor,
        material: torch.Tensor,
        process_descriptors: torch.Tensor | None = None,
        shape_onehot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        proc = self.process_encoder(process)
        parts = [geometry_params, material]
        if self.descriptor_channels:
            if process_descriptors is None:
                process_descriptors = torch.zeros(
                    geometry_params.shape[0],
                    self.descriptor_channels,
                    dtype=geometry_params.dtype,
                    device=geometry_params.device,
                )
            parts.append(process_descriptors)
        if self.shape_channels:
            if shape_onehot is None:
                shape_onehot = torch.zeros(
                    geometry_params.shape[0],
                    self.shape_channels,
                    dtype=geometry_params.dtype,
                    device=geometry_params.device,
                )
            parts.append(shape_onehot)
        parts.append(proc)
        case = self.case_encoder(torch.cat(parts, dim=-1))
        case = case[:, None, :].expand(-1, query.shape[1], -1)
        return self.decoder(torch.cat([query, case], dim=-1))


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class GeoSurfaceResidualTCN(nn.Module):
    """Average-Cp residual corrector with query-conditioned temporal features."""

    def __init__(
        self,
        query_channels: int,
        temporal_channels: int,
        geometry_channels: int,
        material_channels: int,
        descriptor_channels: int = 0,
        shape_channels: int = 0,
        latent: int = 192,
        residual_scale: float = 0.20,
        temporal_mode: str = "query",
        temporal_steps: int = 61,
        global_history_mode: str = "pooled",
        temporal_blend_decay: float = 18.0,
        output_mode: str = "residual",
        time_channel_index: int = 3,
        depth_channel_index: int = 4,
        residual_envelope: str = "none",
        residual_depth_decay: float = 18.0,
    ) -> None:
        super().__init__()
        self.descriptor_channels = int(descriptor_channels)
        self.shape_channels = int(shape_channels)
        self.residual_scale = float(residual_scale)
        if output_mode not in {"residual", "absolute"}:
            raise ValueError(f"Unsupported output_mode={output_mode!r}")
        self.output_mode = output_mode
        self.time_channel_index = int(time_channel_index)
        self.depth_channel_index = int(depth_channel_index)
        if not 0 <= self.time_channel_index < query_channels:
            raise ValueError("time_channel_index must reference a query channel")
        if not 0 <= self.depth_channel_index < query_channels:
            raise ValueError("depth_channel_index must reference a query channel")
        if residual_envelope not in {"none", "diffusion"}:
            raise ValueError(f"Unsupported residual_envelope={residual_envelope!r}")
        self.residual_envelope = residual_envelope
        self.residual_depth_decay = float(residual_depth_decay)
        if temporal_mode not in {"query", "global", "depth_blend"}:
            raise ValueError(f"Unsupported temporal_mode={temporal_mode!r}")
        self.temporal_mode = temporal_mode
        self.temporal_blend_decay = float(temporal_blend_decay)
        if global_history_mode not in {"pooled", "position_mlp"}:
            raise ValueError(f"Unsupported global_history_mode={global_history_mode!r}")
        self.global_history_mode = global_history_mode
        temporal_latent = max(48, latent // 2)
        case_latent = max(48, latent // 2)
        self.temporal_in = nn.Sequential(
            nn.Conv1d(temporal_channels, temporal_latent, kernel_size=1),
            nn.GELU(),
        )
        self.temporal_blocks = nn.Sequential(
            TemporalResidualBlock(temporal_latent, dilation=1),
            TemporalResidualBlock(temporal_latent, dilation=2),
            TemporalResidualBlock(temporal_latent, dilation=4),
            TemporalResidualBlock(temporal_latent, dilation=8),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.position_global = (
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(temporal_channels * temporal_steps, temporal_latent * 2),
                nn.GELU(),
                nn.Linear(temporal_latent * 2, temporal_latent),
                nn.GELU(),
            )
            if global_history_mode == "position_mlp"
            else None
        )
        self.case_encoder = nn.Sequential(
            nn.Linear(
                geometry_channels + material_channels + descriptor_channels + shape_channels + temporal_latent,
                case_latent,
            ),
            nn.GELU(),
            nn.Linear(case_latent, case_latent),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(query_channels + temporal_latent + case_latent, latent),
            nn.GELU(),
            nn.Linear(latent, latent),
            nn.GELU(),
            nn.Linear(latent, 1),
        )
        if self.output_mode == "absolute":
            # Centre a direct predictor near the 0.20 wt% bulk concentration.
            # A zero-centred tanh output avoids the severe sigmoid saturation
            # observed with an otherwise identical absolute baseline.
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)

    @staticmethod
    def gather_query_time(encoded: torch.Tensor, time_norm: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of encoded process features at query times."""

        # encoded: [B, L, Nt], time_norm: [B, Q]
        encoded_time = encoded.transpose(1, 2)
        nt = encoded_time.shape[1]
        pos = torch.clamp(time_norm, 0.0, 1.0) * max(nt - 1, 1)
        i0 = torch.floor(pos).long()
        i1 = torch.clamp(i0 + 1, max=nt - 1)
        frac = (pos - i0.to(pos.dtype)).unsqueeze(-1)
        gather_i0 = i0.unsqueeze(-1).expand(-1, -1, encoded_time.shape[-1])
        gather_i1 = i1.unsqueeze(-1).expand(-1, -1, encoded_time.shape[-1])
        h0 = torch.gather(encoded_time, 1, gather_i0)
        h1 = torch.gather(encoded_time, 1, gather_i1)
        return (1.0 - frac) * h0 + frac * h1

    def forward(
        self,
        query: torch.Tensor,
        temporal_features: torch.Tensor,
        geometry_params: torch.Tensor,
        material: torch.Tensor,
        process_descriptors: torch.Tensor | None = None,
        shape_onehot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        temporal = self.temporal_blocks(self.temporal_in(temporal_features))
        temporal_global = (
            self.position_global(temporal_features)
            if self.position_global is not None
            else self.global_pool(temporal).squeeze(-1)
        )
        if self.temporal_mode == "global":
            temporal_q = temporal_global[:, None, :].expand(-1, query.shape[1], -1)
        else:
            temporal_local = self.gather_query_time(temporal, query[..., self.time_channel_index])
            if self.temporal_mode == "depth_blend":
                depth_norm = torch.clamp(
                    query[..., self.depth_channel_index : self.depth_channel_index + 1], 0.0, 1.0
                )
                local_weight = torch.exp(-self.temporal_blend_decay * depth_norm)
                global_q = temporal_global[:, None, :].expand_as(temporal_local)
                temporal_q = local_weight * temporal_local + (1.0 - local_weight) * global_q
            else:
                temporal_q = temporal_local
        parts = [geometry_params, material]
        if self.descriptor_channels:
            if process_descriptors is None:
                process_descriptors = torch.zeros(
                    geometry_params.shape[0],
                    self.descriptor_channels,
                    dtype=geometry_params.dtype,
                    device=geometry_params.device,
                )
            parts.append(process_descriptors)
        if self.shape_channels:
            if shape_onehot is None:
                shape_onehot = torch.zeros(
                    geometry_params.shape[0],
                    self.shape_channels,
                    dtype=geometry_params.dtype,
                    device=geometry_params.device,
                )
            parts.append(shape_onehot)
        parts.append(temporal_global)
        case = self.case_encoder(torch.cat(parts, dim=-1))
        case = case[:, None, :].expand(-1, query.shape[1], -1)
        raw = self.decoder(torch.cat([query, temporal_q, case], dim=-1))
        if self.output_mode == "absolute":
            return 0.20 + torch.tanh(raw)
        residual = self.residual_scale * torch.tanh(raw)
        if self.residual_envelope == "diffusion":
            time_norm = torch.clamp(query[..., self.time_channel_index : self.time_channel_index + 1], 0.0, 1.0)
            depth_norm = torch.clamp(query[..., self.depth_channel_index : self.depth_channel_index + 1], 0.0, 1.0)
            initial_envelope = 1.0 - torch.exp(-8.0 * time_norm)
            depth_envelope = torch.exp(-self.residual_depth_decay * depth_norm)
            residual = residual * initial_envelope * depth_envelope
        return residual


def _mlp(widths: list[int], activation: type[nn.Module] = nn.GELU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(widths) - 2):
        layers.append(nn.Linear(widths[i], widths[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(widths[-2], widths[-1]))
    return nn.Sequential(*layers)


class GeoSurfaceDeepONet(nn.Module):
    """Branch-trunk DeepONet baseline for GeoSurface query data.

    The branch network encodes the case-level function input: the full process
    history plus material, geometry, process descriptors and shape label.  The
    trunk network encodes query coordinates/time/depth features.  Their inner
    product predicts either the absolute carbon concentration or, when
    ``output_mode="residual"``, the dynamic residual on top of the average-Cp
    physical prior handled by the training script.
    """

    def __init__(
        self,
        query_channels: int,
        temporal_channels: int,
        temporal_steps: int,
        geometry_channels: int,
        material_channels: int,
        descriptor_channels: int = 0,
        shape_channels: int = 0,
        basis: int = 160,
        hidden: int = 256,
        output_mode: str = "absolute",
        residual_scale: float = 0.20,
        time_channel_index: int = 0,
        depth_channel_index: int = 1,
        residual_envelope: str = "none",
        residual_depth_decay: float = 18.0,
    ) -> None:
        super().__init__()
        if output_mode not in {"absolute", "residual"}:
            raise ValueError(f"Unsupported output_mode={output_mode!r}")
        self.output_mode = output_mode
        self.residual_scale = float(residual_scale)
        self.time_channel_index = int(time_channel_index)
        self.depth_channel_index = int(depth_channel_index)
        if residual_envelope not in {"none", "diffusion"}:
            raise ValueError(f"Unsupported residual_envelope={residual_envelope!r}")
        self.residual_envelope = residual_envelope
        self.residual_depth_decay = float(residual_depth_decay)
        self.basis = int(basis)
        self.descriptor_channels = int(descriptor_channels)
        self.shape_channels = int(shape_channels)
        branch_in = (
            temporal_channels * temporal_steps
            + geometry_channels
            + material_channels
            + descriptor_channels
            + shape_channels
        )
        self.branch = _mlp([branch_in, hidden, hidden, basis])
        self.trunk = _mlp([query_channels, hidden, hidden, basis])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        query: torch.Tensor,
        temporal_features: torch.Tensor,
        geometry_params: torch.Tensor,
        material: torch.Tensor,
        process_descriptors: torch.Tensor | None = None,
        shape_onehot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [temporal_features.flatten(start_dim=1), geometry_params, material]
        if self.descriptor_channels:
            if process_descriptors is None:
                process_descriptors = torch.zeros(
                    geometry_params.shape[0],
                    self.descriptor_channels,
                    dtype=geometry_params.dtype,
                    device=geometry_params.device,
                )
            parts.append(process_descriptors)
        if self.shape_channels:
            if shape_onehot is None:
                shape_onehot = torch.zeros(
                    geometry_params.shape[0],
                    self.shape_channels,
                    dtype=geometry_params.dtype,
                    device=geometry_params.device,
                )
            parts.append(shape_onehot)
        branch = self.branch(torch.cat(parts, dim=-1))
        trunk = self.trunk(query)
        out = (branch[:, None, :] * trunk).sum(dim=-1, keepdim=True) / math.sqrt(float(self.basis))
        out = out + self.bias
        if self.output_mode == "residual":
            residual = self.residual_scale * torch.tanh(out)
            if self.residual_envelope == "diffusion":
                time_norm = torch.clamp(
                    query[..., self.time_channel_index : self.time_channel_index + 1], 0.0, 1.0
                )
                depth_norm = torch.clamp(
                    query[..., self.depth_channel_index : self.depth_channel_index + 1], 0.0, 1.0
                )
                residual = residual * (1.0 - torch.exp(-8.0 * time_norm))
                residual = residual * torch.exp(-self.residual_depth_decay * depth_norm)
            return residual
        return out
