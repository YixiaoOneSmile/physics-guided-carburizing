#!/usr/bin/env python3
"""Matched temporal-spatial backbones for the physics-residual matrix."""

from __future__ import annotations

import torch
from torch import nn
from physicsnemo.models.fno import FNO


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class QueryTimeTCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, hidden, 1), nn.GELU())
        self.blocks = nn.Sequential(
            TemporalResidualBlock(hidden, 1),
            TemporalResidualBlock(hidden, 2),
            TemporalResidualBlock(hidden, 4),
            TemporalResidualBlock(hidden, 8),
        )
        self.projection = nn.Conv1d(hidden, latent, 1)

    def forward(self, process: torch.Tensor, query_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.projection(self.blocks(self.stem(process)))
        query_index = query_index.long().clamp(0, sequence.shape[-1] - 1)
        gather_index = query_index[:, None, None].expand(-1, sequence.shape[1], 1)
        query_latent = torch.gather(sequence, 2, gather_index).squeeze(-1)
        return query_latent, sequence.mean(dim=-1)


class PointwiseMLP3D(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, layers: int = 4) -> None:
        super().__init__()
        modules: list[nn.Module] = [nn.Conv3d(in_channels, hidden, 1), nn.GELU()]
        for _ in range(max(layers - 2, 0)):
            modules.extend([nn.Conv3d(hidden, hidden, 1), nn.GELU()])
        modules.append(nn.Conv3d(hidden, 1, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialResidualBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class LocalResNet3D(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(in_channels, hidden, 3, padding=1), nn.GELU())
        self.blocks = nn.Sequential(*(SpatialResidualBlock3D(hidden) for _ in range(blocks)))
        self.head = nn.Sequential(nn.Conv3d(hidden, hidden, 1), nn.GELU(), nn.Conv3d(hidden, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class MatrixPredictor3D(nn.Module):
    """Common TCN/depth fusion with a selectable spatial backbone."""

    def __init__(
        self,
        backbone: str,
        grid_channels: int,
        process_channels: int = 4,
        process_hidden_channels: int = 48,
        process_latent_channels: int = 16,
        depth_gate_lambda: float = 9.0,
        mlp_hidden_channels: int = 64,
        mlp_layers: int = 4,
        resnet_hidden_channels: int = 32,
        resnet_blocks: int = 4,
        fno_latent_channels: int = 32,
        fno_layers: int = 4,
        fno_modes: list[int] | tuple[int, int, int] = (8, 8, 8),
        decoder_layer_size: int = 64,
        padding: int = 4,
    ) -> None:
        super().__init__()
        if backbone not in {"mlp", "resnet", "fno"}:
            raise ValueError(f"unsupported backbone={backbone}")
        self.backbone_name = backbone
        self.depth_gate_lambda = float(depth_gate_lambda)
        self.temporal = QueryTimeTCN(process_channels, process_hidden_channels, process_latent_channels)
        spatial_in = grid_channels + process_latent_channels
        if backbone == "mlp":
            self.spatial = PointwiseMLP3D(spatial_in, mlp_hidden_channels, mlp_layers)
        elif backbone == "resnet":
            self.spatial = LocalResNet3D(spatial_in, resnet_hidden_channels, resnet_blocks)
        else:
            self.spatial = FNO(
                in_channels=spatial_in,
                out_channels=1,
                decoder_layers=2,
                decoder_layer_size=decoder_layer_size,
                dimension=3,
                latent_channels=fno_latent_channels,
                num_fno_layers=fno_layers,
                num_fno_modes=fno_modes,
                padding=padding,
            )

    def forward(self, grid: torch.Tensor, process: torch.Tensor, query_index: torch.Tensor) -> torch.Tensor:
        query_latent, global_latent = self.temporal(process, query_index)
        depth_fraction = torch.relu(grid[:, 1:2]) * 0.5
        gate = torch.exp(-self.depth_gate_lambda * depth_fraction)
        query_grid = query_latent[:, :, None, None, None]
        global_grid = global_latent[:, :, None, None, None]
        blended = gate * query_grid + (1.0 - gate) * global_grid
        blended = blended.expand(-1, -1, *grid.shape[-3:])
        return self.spatial(torch.cat([grid, blended], dim=1))
