#!/usr/bin/env python3
"""Matched-input dataset for direct, prior-conditioned direct, and residual targets."""

from __future__ import annotations

from pathlib import Path

import torch

from residual_datapipe3d_base import RESIDUAL_GRID_CHANNELS, Residual3DQueryDataset


TARGET_MODES = ("direct", "prior_direct", "residual")


class Matrix3DQueryDataset(Residual3DQueryDataset):
    """Return a full 32^3 query-time field with a target-mode-specific label.

    All modes keep the same 17-channel grid width. In ``direct`` mode the
    Average-Cp channel is set to zero, so model parameter counts remain exactly
    matched without leaking the physical prior.
    """

    def __init__(
        self,
        source_path: str | Path,
        sidecar_path: str | Path,
        stats_path: str | Path,
        target_mode: str,
        random_time: bool,
        samples_per_case: int = 4,
        preload: bool = False,
    ) -> None:
        if target_mode not in TARGET_MODES:
            raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {target_mode}")
        self.target_mode = target_mode
        super().__init__(
            source_path=source_path,
            sidecar_path=sidecar_path,
            stats_path=stats_path,
            random_time=random_time,
            samples_per_case=samples_per_case,
            preload=preload,
        )

    def get_case_query(self, case_id: int, query_index: int) -> dict[str, torch.Tensor]:
        sample = super().get_case_query(case_id, query_index)
        grid = sample["grid"].clone()
        if self.target_mode == "direct":
            grid[-1].zero_()
        if self.target_mode == "residual":
            target = sample["residual"]
        else:
            target = (
                sample["truth"] - float(self.stats["carbon_mean"])
            ) / float(self.stats["carbon_std"])
        sample["grid"] = grid
        sample["target"] = target.to(torch.float32)
        sample["target_mode_id"] = torch.tensor(TARGET_MODES.index(self.target_mode), dtype=torch.long)
        return sample


GRID_CHANNELS = RESIDUAL_GRID_CHANNELS
