#!/usr/bin/env python3
"""Dataset utilities for strict-Average-Cp residual learning."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


RESIDUAL_GRID_CHANNELS = (
    "mask",
    "sdf",
    "surface_count",
    "x",
    "y",
    "z",
    "query_time_norm",
    "c0_wt",
    "log10_h_m",
    "log10_d_ref",
    "activation_norm",
    "cp_time_average",
    "cp_std",
    "exposure_norm",
    "cp_at_query",
    "temperature_norm",
    "avgcp_prior_normalized",
)


def integrate_trapezoid(values: np.ndarray, time_grid: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, time_grid))
    return float(np.trapz(values, time_grid))


def strict_time_average(values: np.ndarray, time_grid: np.ndarray) -> float:
    duration = max(float(time_grid[-1] - time_grid[0]), 1e-12)
    return integrate_trapezoid(values.astype(np.float64), time_grid.astype(np.float64)) / duration


def cumulative_trapezoid(values: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if len(values) > 1:
        increments = 0.5 * (values[1:] + values[:-1]) * np.diff(time_grid)
        out[1:] = np.cumsum(increments)
    return out


def compute_residual_stats(source_path: Path, sidecar_path: Path, output_path: Path) -> dict:
    """Streaming statistics over active training voxels only."""
    count = 0
    residual_sum = 0.0
    residual_sq_sum = 0.0
    carbon_sum = 0.0
    carbon_sq_sum = 0.0
    process_samples = []
    residual_abs_samples = []
    with h5py.File(source_path, "r") as source, h5py.File(sidecar_path, "r") as sidecar:
        complete = np.asarray(sidecar["complete"][:], dtype=bool)
        if not np.all(complete):
            raise RuntimeError(f"incomplete Average-Cp sidecar: {sidecar_path}")
        time_grid = np.asarray(source["time"][:], dtype=np.float64)
        for case_id in range(source["carbon"].shape[0]):
            truth = np.asarray(source["carbon"][case_id], dtype=np.float32)
            prior = np.asarray(sidecar["avg_carbon"][case_id], dtype=np.float32)
            mask = np.asarray(source["mask"][case_id, 0], dtype=bool)
            residual = (truth - prior)[:, mask].astype(np.float64).reshape(-1)
            carbon = truth[:, mask].astype(np.float64).reshape(-1)
            count += residual.size
            residual_sum += float(residual.sum())
            residual_sq_sum += float(np.dot(residual, residual))
            carbon_sum += float(carbon.sum())
            carbon_sq_sum += float(np.dot(carbon, carbon))
            process = np.asarray(source["process"][case_id], dtype=np.float64)
            cp_avg = strict_time_average(process[0], time_grid)
            centered = process[0] - cp_avg
            cumulative = cumulative_trapezoid(centered, time_grid) / max(float(time_grid[-1]), 1.0)
            process_samples.append(np.stack([process[0], process[1], centered, cumulative], axis=0))
            stride = max(1, residual.size // 20000)
            residual_abs_samples.append(np.abs(residual[::stride]))

    residual_mean = residual_sum / max(count, 1)
    residual_var = residual_sq_sum / max(count, 1) - residual_mean**2
    carbon_mean = carbon_sum / max(count, 1)
    carbon_var = carbon_sq_sum / max(count, 1) - carbon_mean**2
    process_array = np.stack(process_samples, axis=0)
    abs_values = np.concatenate(residual_abs_samples)
    stats = {
        "residual_mean": float(residual_mean),
        "residual_std": float(max(np.sqrt(max(residual_var, 0.0)), 1e-7)),
        "residual_abs_p50": float(np.quantile(abs_values, 0.50)),
        "residual_abs_p90": float(np.quantile(abs_values, 0.90)),
        "residual_abs_p99": float(np.quantile(abs_values, 0.99)),
        "carbon_mean": float(carbon_mean),
        "carbon_std": float(max(np.sqrt(max(carbon_var, 0.0)), 1e-7)),
        "process_mean": process_array.mean(axis=(0, 2)).astype(float).tolist(),
        "process_std": np.maximum(process_array.std(axis=(0, 2)), 1e-7).astype(float).tolist(),
        "active_value_count": int(count),
        "grid_channels": list(RESIDUAL_GRID_CHANNELS),
        "process_channels": ["Cp", "temperature", "Cp-Cp_time_average", "cumulative_centered_Cp"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


class Residual3DQueryDataset(Dataset):
    def __init__(
        self,
        source_path: str | Path,
        sidecar_path: str | Path,
        stats_path: str | Path,
        random_time: bool,
        samples_per_case: int = 4,
        preload: bool = False,
    ) -> None:
        self.source_path = str(source_path)
        self.sidecar_path = str(sidecar_path)
        self.random_time = bool(random_time)
        self.samples_per_case = int(samples_per_case)
        with h5py.File(self.source_path, "r") as source:
            self.n_cases = int(source["carbon"].shape[0])
            self.nt = int(source["carbon"].shape[1])
            self.time = np.asarray(source["time"][:], dtype=np.float32)
            self.coords = np.asarray(source["coords"][:], dtype=np.float32)
        with h5py.File(self.sidecar_path, "r") as sidecar:
            if not np.all(np.asarray(sidecar["complete"][:], dtype=bool)):
                raise RuntimeError(f"sidecar is incomplete: {self.sidecar_path}")
        self.stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        self.process_mean = np.asarray(self.stats["process_mean"], dtype=np.float32)[:, None]
        self.process_std = np.asarray(self.stats["process_std"], dtype=np.float32)[:, None]
        self.cache: dict[str, np.ndarray] | None = None
        if preload:
            with h5py.File(self.source_path, "r") as source, h5py.File(self.sidecar_path, "r") as sidecar:
                self.cache = {
                    "carbon": np.asarray(source["carbon"][:]),
                    "mask": np.asarray(source["mask"][:]),
                    "sdf": np.asarray(source["sdf"][:]),
                    "surface": np.asarray(source["surface"][:]),
                    "process": np.asarray(source["process"][:]),
                    "material": np.asarray(source["material"][:]),
                    "prior": np.asarray(sidecar["avg_carbon"][:]),
                }

    def __len__(self) -> int:
        return self.n_cases * self.samples_per_case

    def choose_query_index(self, item_index: int) -> int:
        if self.random_time:
            return int(np.random.randint(0, self.nt))
        return int((item_index // self.n_cases) % self.nt)

    def make_process_features(self, process: np.ndarray) -> np.ndarray:
        cp_avg = strict_time_average(process[0], self.time)
        centered = process[0] - cp_avg
        cumulative = cumulative_trapezoid(centered, self.time) / max(float(self.time[-1]), 1.0)
        features = np.stack([process[0], process[1], centered, cumulative], axis=0).astype(np.float32)
        return (features - self.process_mean) / np.maximum(self.process_std, 1e-7)

    def make_grid(
        self,
        mask: np.ndarray,
        sdf: np.ndarray,
        surface: np.ndarray,
        process: np.ndarray,
        material: np.ndarray,
        prior: np.ndarray,
        query_index: int,
    ) -> np.ndarray:
        n = mask.shape[-1]
        cp = process[0]
        temp = process[1]
        cp_avg = strict_time_average(cp, self.time)
        time_norm = float(self.time[query_index] / max(float(self.time[-1]), 1.0))
        exposure = integrate_trapezoid(cp[: query_index + 1], self.time[: query_index + 1])
        exposure_norm = exposure / max(float(self.time[-1]), 1.0)
        c0, h_m, d_ref, activation, _domain_size = material
        surface_count = np.sum(surface, axis=0) / 6.0
        scalars = (
            time_norm,
            float(c0),
            float(np.log10(max(float(h_m), 1e-20))),
            float(np.log10(max(float(d_ref), 1e-20))),
            float(activation / 150000.0),
            cp_avg,
            float(np.std(cp)),
            exposure_norm,
            float(cp[query_index]),
            float((temp[query_index] - 900.0) / 60.0),
        )
        channels = [mask[0], sdf[0], surface_count, self.coords[0], self.coords[1], self.coords[2]]
        channels.extend(np.full((n, n, n), value, dtype=np.float32) for value in scalars)
        prior_norm = (prior - float(self.stats["carbon_mean"])) / float(self.stats["carbon_std"])
        channels.append(prior_norm.astype(np.float32))
        return np.stack(channels, axis=0).astype(np.float32)

    def get_case_query(self, case_id: int, query_index: int) -> dict[str, torch.Tensor]:
        if self.cache is None:
            with h5py.File(self.source_path, "r") as source, h5py.File(self.sidecar_path, "r") as sidecar:
                mask = np.asarray(source["mask"][case_id], dtype=np.float32)
                sdf = np.asarray(source["sdf"][case_id], dtype=np.float32)
                surface = np.asarray(source["surface"][case_id], dtype=np.float32)
                process = np.asarray(source["process"][case_id], dtype=np.float32)
                material = np.asarray(source["material"][case_id], dtype=np.float32)
                truth = np.asarray(source["carbon"][case_id, query_index], dtype=np.float32)
                prior = np.asarray(sidecar["avg_carbon"][case_id, query_index], dtype=np.float32)
        else:
            mask = np.asarray(self.cache["mask"][case_id], dtype=np.float32)
            sdf = np.asarray(self.cache["sdf"][case_id], dtype=np.float32)
            surface = np.asarray(self.cache["surface"][case_id], dtype=np.float32)
            process = np.asarray(self.cache["process"][case_id], dtype=np.float32)
            material = np.asarray(self.cache["material"][case_id], dtype=np.float32)
            truth = np.asarray(self.cache["carbon"][case_id, query_index], dtype=np.float32)
            prior = np.asarray(self.cache["prior"][case_id, query_index], dtype=np.float32)
        grid = self.make_grid(mask, sdf, surface, process, material, prior, query_index)
        process_features = self.make_process_features(process)
        residual = truth - prior
        residual_norm = (residual - float(self.stats["residual_mean"])) / float(self.stats["residual_std"])
        return {
            "grid": torch.from_numpy(grid),
            "process": torch.from_numpy(process_features),
            "residual": torch.from_numpy(residual_norm[None].astype(np.float32)),
            "prior": torch.from_numpy(prior[None]),
            "truth": torch.from_numpy(truth[None]),
            "mask": torch.from_numpy(mask),
            "surface": torch.from_numpy(surface),
            "query_index": torch.tensor(query_index, dtype=torch.long),
            "case_id": torch.tensor(case_id, dtype=torch.long),
        }

    def __getitem__(self, item_index: int) -> dict[str, torch.Tensor]:
        case_id = int(item_index % self.n_cases)
        return self.get_case_query(case_id, self.choose_query_index(item_index))
