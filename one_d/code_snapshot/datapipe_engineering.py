# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from physics import (
    diffusivity_carbon_austenite,
    diffusivity_carbon_austenite_agren,
    equilibrium_surface_carbon,
    equilibrium_surface_carbon_from_potential,
)


GEOSURFACE_QUERY_CHANNELS = (
    "x_mm_scaled",
    "y_mm_scaled",
    "z_mm_scaled",
    "time_norm",
    "depth_to_surface_norm",
    "sdf_norm",
    "local_thickness_norm",
    "mask",
)

GEOSURFACE_RESIDUAL_QUERY_CHANNELS = (
    *GEOSURFACE_QUERY_CHANNELS,
    "avg_carbon_wt",
)

GEOSURFACE_TEMPORAL_CHANNELS = (
    "Cp",
    "Cp_minus_mean",
    "T_scaled",
    "log10_hm_scaled",
    "boost_stage",
    "diffuse_stage",
    "Ceq",
    "D_scaled",
    "cum_Ceq_norm",
    "cum_D_norm",
    "cum_surface_drive_norm",
)


def _normalize_process(process: np.ndarray) -> np.ndarray:
    out = process.astype(np.float32).copy()
    out[0] = out[0]
    out[1] = (out[1] - 880.0) / 80.0
    out[2] = np.log10(np.maximum(out[2], 1e-12)) + 9.0
    return out


def _normalize_geometry_params(params: np.ndarray) -> np.ndarray:
    out = params.astype(np.float32).copy()
    if out.size:
        out[:8] /= 100.0
    if out.size > 8:
        out[8] /= 5.0
    if out.size > 9:
        out[9] /= 50.0
    if out.size > 10:
        out[10:16] /= 100.0
    if out.size > 16:
        out[16] /= 30.0
    return out


def _normalize_material(material: np.ndarray) -> np.ndarray:
    out = material.astype(np.float32).copy()
    if out.size == 3:
        out[1] = np.log10(max(float(out[1]), 1e-20)) + 7.0
        out[2] = np.log10(max(float(out[2]), 1e-20)) + 7.0
    elif out.size >= 5:
        out[1] = np.log10(max(float(out[1]), 1e-20)) + 11.0
        out[2] = (out[2] - 145000.0) / 10000.0
        out[3] = np.log10(max(float(out[3]), 1e-20)) + 8.0
        out[4] = np.log10(max(float(out[4]), 1e-20)) + 9.0
    return out


def _normalize_descriptors(descriptors: np.ndarray) -> np.ndarray:
    out = descriptors.astype(np.float32).copy()
    if out.size >= 9:
        out[1] /= 0.20
        out[7] *= 1e7
        out[8] *= 1e4
    return out


def _build_temporal_features(
    process: np.ndarray,
    material: np.ndarray,
    time_s: np.ndarray,
    stage: np.ndarray | None = None,
    diffusivity_model: str = "legacy_parametric",
) -> np.ndarray:
    """Build query-gatherable process features for residual GeoSurface models."""

    process = np.asarray(process, dtype=np.float32)
    time_s = np.asarray(time_s, dtype=np.float32)
    cp = process[0].astype(np.float32)
    temp = process[1].astype(np.float32)
    hm = np.maximum(process[2].astype(np.float32), 1e-12)
    c0 = float(material[0]) if material.size else 0.20

    if stage is None:
        threshold = float(np.percentile(hm, 55.0))
        boost = (hm >= threshold).astype(np.float32)
    else:
        boost = (np.asarray(stage).astype(np.int32) == 1).astype(np.float32)
    diffuse = 1.0 - boost

    if diffusivity_model == "agren_binary":
        ceq = equilibrium_surface_carbon_from_potential(cp).astype(np.float32)
        diffusivity = diffusivity_carbon_austenite_agren(temp, c0).astype(np.float32)
    else:
        ceq = equilibrium_surface_carbon(cp, temp).astype(np.float32)
        d_ref = float(material[1]) if material.size > 1 else 2.0e-11
        activation = float(material[2]) if material.size > 2 else 145_000.0
        diffusivity = diffusivity_carbon_austenite(temp, c0, d_ref, activation).astype(np.float32)

    dt = np.diff(time_s, prepend=time_s[:1]).astype(np.float32)
    cum_ceq = np.cumsum(ceq * dt).astype(np.float32)
    cum_d = np.cumsum(diffusivity * dt).astype(np.float32)
    cum_drive = np.cumsum(hm * (ceq - c0) * dt).astype(np.float32)

    def norm_cum(values: np.ndarray) -> np.ndarray:
        scale = float(np.max(np.abs(values)))
        return values / max(scale, 1e-8)

    features = np.stack(
        [
            cp,
            cp - float(np.mean(cp)),
            (temp - 880.0) / 80.0,
            np.log10(hm) + 9.0,
            boost,
            diffuse,
            ceq,
            diffusivity * 1.0e11,
            norm_cum(cum_ceq),
            norm_cum(cum_d),
            norm_cum(cum_drive),
        ],
        axis=0,
    ).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class Engineering1DDataset(Dataset):
    def __init__(self, file_path: str, device: str | torch.device = "cpu") -> None:
        self.file_path = str(file_path)
        with h5py.File(self.file_path, "r") as f:
            self.length = int(f["carbon"].shape[0])
            self.nd = int(f["carbon"].shape[2])
            self.nt = int(f["carbon"].shape[3])
        self.device = torch.device(device)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        with h5py.File(self.file_path, "r") as f:
            sample = {
                "invar": torch.from_numpy(np.asarray(f["invar"][idx], dtype=np.float32)),
                "carbon": torch.from_numpy(np.asarray(f["carbon"][idx], dtype=np.float32)),
                "process": torch.from_numpy(np.asarray(f["process"][idx], dtype=np.float32)),
                "material": torch.from_numpy(np.asarray(f["material"][idx], dtype=np.float32)),
                "depth": torch.from_numpy(np.asarray(f["depth"][:], dtype=np.float32)),
                "time": torch.from_numpy(np.asarray(f["time"][:], dtype=np.float32)),
            }
        if self.device.type == "cuda":
            sample = {key: value.to(self.device) for key, value in sample.items()}
        return sample


class Engineering2DSectionDataset(Dataset):
    def __init__(self, file_path: str, device: str | torch.device = "cpu") -> None:
        self.file_path = str(file_path)
        with h5py.File(self.file_path, "r") as f:
            self.length = int(f["carbon"].shape[0])
            self.nt = int(f["carbon"].shape[1])
            self.n = int(f["carbon"].shape[2])
        self.device = torch.device(device)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        with h5py.File(self.file_path, "r") as f:
            sample = {
                "carbon": torch.from_numpy(np.asarray(f["carbon"][idx], dtype=np.float32)),
                "mask": torch.from_numpy(np.asarray(f["mask"][idx], dtype=np.float32)),
                "depth_to_surface_mm": torch.from_numpy(np.asarray(f["depth_to_surface_mm"][idx], dtype=np.float32)),
                "geometry_params": torch.from_numpy(np.asarray(f["geometry_params"][idx], dtype=np.float32)),
                "shape_id": torch.tensor(int(f["shape_id"][idx]), dtype=torch.long),
                "process": torch.from_numpy(np.asarray(f["process"][idx], dtype=np.float32)),
                "material": torch.from_numpy(np.asarray(f["material"][idx], dtype=np.float32)),
                "time": torch.from_numpy(np.asarray(f["time"][:], dtype=np.float32)),
            }
        if self.device.type == "cuda":
            sample = {key: value.to(self.device) for key, value in sample.items()}
        return sample


class GeoSurfaceDataset(Dataset):
    def __init__(self, file_path: str, device: str | torch.device = "cpu") -> None:
        self.file_path = str(file_path)
        with h5py.File(self.file_path, "r") as f:
            self.length = int(f["carbon_profile"].shape[0])
            self.nt = int(f["carbon_profile"].shape[1])
            self.nd = int(f["carbon_profile"].shape[2])
            self.n = int(f["lowres_sdf_mm"].shape[-1])
        self.device = torch.device(device)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        with h5py.File(self.file_path, "r") as f:
            sample = {
                "lowres_mask": torch.from_numpy(np.asarray(f["lowres_mask"][idx], dtype=np.float32)),
                "lowres_sdf_mm": torch.from_numpy(np.asarray(f["lowres_sdf_mm"][idx], dtype=np.float32)),
                "depth_to_surface_mm": torch.from_numpy(np.asarray(f["depth_to_surface_mm"][idx], dtype=np.float32)),
                "local_thickness_mm": torch.from_numpy(np.asarray(f["local_thickness_mm"][idx], dtype=np.float32)),
                "coords_mm": torch.from_numpy(np.asarray(f["coords_mm"][idx], dtype=np.float32)),
                "carbon_profile": torch.from_numpy(np.asarray(f["carbon_profile"][idx], dtype=np.float32)),
                "depth_grid_mm": torch.from_numpy(np.asarray(f["depth_grid_mm"][:], dtype=np.float32)),
                "geometry_params": torch.from_numpy(np.asarray(f["geometry_params"][idx], dtype=np.float32)),
                "shape_id": torch.tensor(int(f["shape_id"][idx]), dtype=torch.long),
                "process": torch.from_numpy(np.asarray(f["process"][idx], dtype=np.float32)),
                "material": torch.from_numpy(np.asarray(f["material"][idx], dtype=np.float32)),
                "time": torch.from_numpy(np.asarray(f["time"][:], dtype=np.float32)),
            }
        if self.device.type == "cuda":
            sample = {key: value.to(self.device) for key, value in sample.items()}
        return sample


class GeoSurfaceQueryDataset(Dataset):
    """Random query-point dataset for engineering-scale GeoSurface surrogates."""

    def __init__(
        self,
        file_path: str,
        queries_per_case: int = 4096,
        random_query: bool = True,
        seed: int = 1234,
        preload: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        self.file_path = str(file_path)
        self.queries_per_case = int(queries_per_case)
        self.random_query = bool(random_query)
        self.seed = int(seed)
        self.preload = bool(preload)
        self.cache: dict[str, np.ndarray] = {}
        with h5py.File(self.file_path, "r") as f:
            self.length = int(f["carbon_profile"].shape[0])
            self.nt = int(f["carbon_profile"].shape[1])
            self.nd = int(f["carbon_profile"].shape[2])
            self.n = int(f["lowres_mask"].shape[-1])
            self.num_shapes = int(len(f["shape_names"])) if "shape_names" in f else 6
            self.depth_max_mm = float(np.asarray(f["depth_grid_mm"][:], dtype=np.float32)[-1])
            raw_diffusivity_model = f.attrs.get("diffusivity_model", "legacy_parametric")
            if isinstance(raw_diffusivity_model, bytes):
                raw_diffusivity_model = raw_diffusivity_model.decode("utf-8")
            self.diffusivity_model = str(raw_diffusivity_model)
            if self.preload:
                for key in (
                    "lowres_mask",
                    "depth_to_surface_mm",
                    "lowres_sdf_mm",
                    "local_thickness_mm",
                    "coords_mm",
                    "carbon_profile",
                    "avg_carbon_profile",
                    "depth_grid_mm",
                    "time",
                    "process",
                    "avg_process",
                    "process_descriptors",
                    "geometry_params",
                    "material",
                    "shape_id",
                    "pair_id",
                    "variant_id",
                    "stage",
                    "process_family_id",
                    "dynamic_sensitivity_score",
                ):
                    if key in f:
                        self.cache[key] = np.asarray(f[key])
        self.device = torch.device(device)

    def __len__(self) -> int:
        return self.length

    def _rng(self, idx: int) -> np.random.Generator:
        if self.random_query:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + int(idx))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = self._rng(idx)
        if self.cache:
            mask = np.asarray(self.cache["lowres_mask"][idx, 0], dtype=np.float32) > 0.5
            depth_map = np.asarray(self.cache["depth_to_surface_mm"][idx, 0], dtype=np.float32)
            sdf = np.asarray(self.cache["lowres_sdf_mm"][idx, 0], dtype=np.float32)
            thickness = np.asarray(self.cache["local_thickness_mm"][idx, 0], dtype=np.float32)
            coords = np.asarray(self.cache["coords_mm"][idx], dtype=np.float32)
            profile = np.asarray(self.cache["carbon_profile"][idx], dtype=np.float32)
            avg_profile = np.asarray(self.cache.get("avg_carbon_profile", self.cache["carbon_profile"])[idx], dtype=np.float32)
            depth_grid = np.asarray(self.cache["depth_grid_mm"], dtype=np.float32)
            time = np.asarray(self.cache["time"], dtype=np.float32)
            process = np.asarray(self.cache["process"][idx], dtype=np.float32)
            avg_process = np.asarray(self.cache.get("avg_process", self.cache["process"])[idx], dtype=np.float32)
            descriptors = np.asarray(
                self.cache["process_descriptors"][idx] if "process_descriptors" in self.cache else np.zeros(9),
                dtype=np.float32,
            )
            geometry_params = np.asarray(self.cache["geometry_params"][idx], dtype=np.float32)
            material = np.asarray(self.cache["material"][idx], dtype=np.float32)
            material_raw = material.copy()
            shape_id = int(self.cache["shape_id"][idx])
            pair_id = int(self.cache["pair_id"][idx]) if "pair_id" in self.cache else idx
            variant_id = int(self.cache["variant_id"][idx]) if "variant_id" in self.cache else 0
            stage = np.asarray(self.cache["stage"][idx], dtype=np.int32) if "stage" in self.cache else None
            family_id = int(self.cache["process_family_id"][idx]) if "process_family_id" in self.cache else 0
            sensitivity = float(self.cache["dynamic_sensitivity_score"][idx]) if "dynamic_sensitivity_score" in self.cache else 0.0
        else:
            with h5py.File(self.file_path, "r") as f:
                mask = np.asarray(f["lowres_mask"][idx, 0], dtype=np.float32) > 0.5
                depth_map = np.asarray(f["depth_to_surface_mm"][idx, 0], dtype=np.float32)
                sdf = np.asarray(f["lowres_sdf_mm"][idx, 0], dtype=np.float32)
                thickness = np.asarray(f["local_thickness_mm"][idx, 0], dtype=np.float32)
                coords = np.asarray(f["coords_mm"][idx], dtype=np.float32)
                profile = np.asarray(f["carbon_profile"][idx], dtype=np.float32)
                avg_profile = (
                    np.asarray(f["avg_carbon_profile"][idx], dtype=np.float32)
                    if "avg_carbon_profile" in f
                    else profile.copy()
                )
                depth_grid = np.asarray(f["depth_grid_mm"][:], dtype=np.float32)
                time = np.asarray(f["time"][:], dtype=np.float32)
                process = np.asarray(f["process"][idx], dtype=np.float32)
                avg_process = np.asarray(f["avg_process"][idx], dtype=np.float32) if "avg_process" in f else process.copy()
                descriptors = (
                    np.asarray(f["process_descriptors"][idx], dtype=np.float32)
                    if "process_descriptors" in f
                    else np.zeros(9, dtype=np.float32)
                )
                geometry_params = np.asarray(f["geometry_params"][idx], dtype=np.float32)
                material = np.asarray(f["material"][idx], dtype=np.float32)
                material_raw = material.copy()
                shape_id = int(f["shape_id"][idx])
                pair_id = int(f["pair_id"][idx]) if "pair_id" in f else idx
                variant_id = int(f["variant_id"][idx]) if "variant_id" in f else 0
                stage = np.asarray(f["stage"][idx], dtype=np.int32) if "stage" in f else None
                family_id = int(f["process_family_id"][idx]) if "process_family_id" in f else 0
                sensitivity = float(f["dynamic_sensitivity_score"][idx]) if "dynamic_sensitivity_score" in f else 0.0

        active = np.flatnonzero(mask.reshape(-1))
        if active.size == 0:
            raise ValueError(f"No active voxels in sample {idx}")
        active_depth = depth_map.reshape(-1)[active]
        band_mm = max(0.35, float(np.nanpercentile(active_depth, 8.0)))
        surface_active = active[active_depth <= band_mm]
        if surface_active.size == 0:
            surface_active = active

        q = self.queries_per_case
        if self.random_query:
            n_surface = q // 2
            surf_ids = rng.choice(surface_active, size=n_surface, replace=surface_active.size < n_surface)
            bulk_ids = rng.choice(active, size=q - n_surface, replace=active.size < q - n_surface)
            flat_ids = np.concatenate([surf_ids, bulk_ids])
            surface_query = np.concatenate([np.ones(n_surface, dtype=bool), np.zeros(q - n_surface, dtype=bool)])
            order = rng.permutation(q)
            flat_ids = flat_ids[order]
            surface_query = surface_query[order]
            time_ids = rng.integers(0, self.nt, size=q, endpoint=False)
            time_ids[: max(1, q // 8)] = self.nt - 1
        else:
            n_surface = q // 2
            surf_take = np.linspace(0, surface_active.size - 1, n_surface).round().astype(np.int64)
            bulk_take = np.linspace(0, active.size - 1, q - n_surface).round().astype(np.int64)
            flat_ids = np.concatenate([surface_active[surf_take % surface_active.size], active[bulk_take % active.size]])
            surface_query = np.concatenate([np.ones(n_surface, dtype=bool), np.zeros(q - n_surface, dtype=bool)])
            time_ids = np.linspace(0, self.nt - 1, q).round().astype(np.int64)

        depth_values = np.nan_to_num(depth_map.reshape(-1)[flat_ids], nan=0.0).astype(np.float32)
        n_surface_values = int(surface_query.sum())
        if self.random_query:
            surface_depths = rng.uniform(0.0, min(0.65, self.depth_max_mm), size=n_surface_values).astype(np.float32)
            surface_depths[: n_surface_values // 2] = 0.0
            depth_values[surface_query] = surface_depths
        else:
            surface_depths = np.linspace(0.0, min(0.65, self.depth_max_mm), n_surface_values, dtype=np.float32)
            surface_depths[: n_surface_values // 2] = 0.0
            depth_values[surface_query] = surface_depths
        spacing = float(depth_grid[1] - depth_grid[0]) if len(depth_grid) > 1 else max(self.depth_max_mm, 1.0)
        x = np.clip(depth_values / max(spacing, 1e-8), 0.0, len(depth_grid) - 1.0)
        i0 = np.floor(x).astype(np.int64)
        i1 = np.clip(i0 + 1, 0, len(depth_grid) - 1)
        frac = (x - i0).astype(np.float32)
        tid = time_ids.astype(np.int64)
        target = ((1.0 - frac) * profile[tid, i0] + frac * profile[tid, i1]).astype(np.float32)
        avg_target = ((1.0 - frac) * avg_profile[tid, i0] + frac * avg_profile[tid, i1]).astype(np.float32)

        coord_flat = coords.reshape(3, -1)[:, flat_ids].T
        query = np.column_stack(
            [
                coord_flat[:, 0] / 100.0,
                coord_flat[:, 1] / 100.0,
                coord_flat[:, 2] / 100.0,
                time[time_ids] / max(float(time[-1]), 1.0),
                depth_values / max(self.depth_max_mm, 1e-6),
                np.nan_to_num(sdf.reshape(-1)[flat_ids], nan=0.0) / 100.0,
                np.nan_to_num(thickness.reshape(-1)[flat_ids], nan=0.0) / 100.0,
                np.ones(q, dtype=np.float32),
            ]
        ).astype(np.float32)
        surface_weight = (depth_values <= max(0.65, band_mm)).astype(np.float32)
        final_weight = (time_ids == self.nt - 1).astype(np.float32)
        shape_onehot = np.zeros(self.num_shapes, dtype=np.float32)
        shape_onehot[shape_id] = 1.0
        temporal_features = _build_temporal_features(
            process,
            material_raw,
            time,
            stage,
            diffusivity_model=self.diffusivity_model,
        )

        sample = {
            "query": torch.from_numpy(query),
            "carbon": torch.from_numpy(target[:, None]),
            "avg_carbon": torch.from_numpy(avg_target[:, None]),
            "surface_weight": torch.from_numpy(surface_weight[:, None]),
            "final_weight": torch.from_numpy(final_weight[:, None]),
            "depth_mm": torch.from_numpy(depth_values[:, None]),
            "time_id": torch.from_numpy(time_ids.astype(np.int64)),
            "time": torch.from_numpy(time),
            "process_raw": torch.from_numpy(process),
            "process": torch.from_numpy(_normalize_process(process)),
            "avg_process": torch.from_numpy(_normalize_process(avg_process)),
            "temporal_features": torch.from_numpy(temporal_features),
            "process_descriptors": torch.from_numpy(_normalize_descriptors(descriptors)),
            "geometry_params": torch.from_numpy(_normalize_geometry_params(geometry_params)),
            "material": torch.from_numpy(_normalize_material(material)),
            "material_raw": torch.from_numpy(material_raw),
            "shape_onehot": torch.from_numpy(shape_onehot),
            "shape_id": torch.tensor(shape_id, dtype=torch.long),
            "pair_id": torch.tensor(pair_id, dtype=torch.long),
            "variant_id": torch.tensor(variant_id, dtype=torch.long),
            "process_family_id": torch.tensor(family_id, dtype=torch.long),
            "dynamic_sensitivity_score": torch.tensor(sensitivity, dtype=torch.float32),
        }
        if self.device.type == "cuda":
            sample = {key: value.to(self.device) for key, value in sample.items()}
        return sample


class GeoSurfaceResidualQueryDataset(GeoSurfaceQueryDataset):
    """GeoSurface query dataset with average-Cp residual targets.

    The learning target is ``carbon - avg_carbon``. The model input appends the
    local average-Cp prediction to the ordinary spatial/time query, making the
    physical prior explicit. ``dynamic_sensitivity_score`` remains evaluation
    metadata only because it is derived from the unknown target field.
    """

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        sample["residual"] = sample["carbon"] - sample["avg_carbon"]
        sample["residual_query"] = torch.cat([sample["query"], sample["avg_carbon"]], dim=-1)
        return sample


class CarburizingProfileQueryDataset(Dataset):
    """Depth--time queries for the publication task.

    The physical labels are one-dimensional depth profiles.  This dataset
    therefore samples directly in ``(time, depth)`` and keeps engineering
    geometry only as optional case metadata.  It avoids presenting arbitrary
    voxel coordinates as if they affected labels that were generated solely by
    a depth solver.
    """

    def __init__(
        self,
        file_path: str,
        queries_per_case: int = 2048,
        random_query: bool = True,
        full_grid: bool = False,
        seed: int = 1234,
    ) -> None:
        self.file_path = str(file_path)
        self.queries_per_case = int(queries_per_case)
        self.random_query = bool(random_query)
        self.full_grid = bool(full_grid)
        self.seed = int(seed)
        self.epoch = 0
        with h5py.File(self.file_path, "r") as handle:
            self.cache = {
                key: np.asarray(handle[key])
                for key in (
                    "carbon_profile",
                    "avg_carbon_profile",
                    "depth_grid_mm",
                    "time",
                    "process",
                    "process_descriptors",
                    "material",
                    "stage",
                    "geometry_params",
                    "shape_id",
                    "pair_id",
                    "variant_id",
                    "process_family_id",
                    "dynamic_sensitivity_score",
                )
                if key in handle
            }
            self.length = int(handle["carbon_profile"].shape[0])
            self.nt = int(handle["carbon_profile"].shape[1])
            self.nd = int(handle["carbon_profile"].shape[2])
            self.num_shapes = int(len(handle["shape_names"])) if "shape_names" in handle else 6
            raw_model = handle.attrs.get("diffusivity_model", "legacy_parametric")
            if isinstance(raw_model, bytes):
                raw_model = raw_model.decode("utf-8")
            self.diffusivity_model = str(raw_model)
        self.depth_max_mm = float(self.cache["depth_grid_mm"][-1])

    def __len__(self) -> int:
        return self.length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(idx)]))

    def _queries(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if self.full_grid:
            time_ids = np.repeat(np.arange(self.nt, dtype=np.int64), self.nd)
            depth_values = np.tile(self.cache["depth_grid_mm"].astype(np.float32), self.nt)
            return time_ids, depth_values

        q = self.queries_per_case
        if q <= 0:
            raise ValueError("queries_per_case must be positive unless full_grid=True")
        if self.random_query:
            rng = self._rng(idx)
            n_surface = q // 2
            depth_values = np.concatenate(
                [
                    rng.uniform(0.0, min(0.65, self.depth_max_mm), size=n_surface),
                    rng.uniform(0.0, self.depth_max_mm, size=q - n_surface),
                ]
            ).astype(np.float32)
            time_ids = rng.integers(0, self.nt, size=q, endpoint=False, dtype=np.int64)
            time_ids[: max(1, q // 8)] = self.nt - 1
            time_ids[max(1, q // 8) : max(2, 3 * q // 16)] = 0
            order = rng.permutation(q)
            return time_ids[order], depth_values[order]

        n_surface = q // 2
        depth_values = np.concatenate(
            [
                np.linspace(0.0, min(0.65, self.depth_max_mm), n_surface, dtype=np.float32),
                np.linspace(0.0, self.depth_max_mm, q - n_surface, dtype=np.float32),
            ]
        )
        time_ids = np.linspace(0, self.nt - 1, q).round().astype(np.int64)
        return time_ids, depth_values

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        profile = np.asarray(self.cache["carbon_profile"][idx], dtype=np.float32)
        avg_profile = np.asarray(self.cache["avg_carbon_profile"][idx], dtype=np.float32)
        depth_grid = np.asarray(self.cache["depth_grid_mm"], dtype=np.float32)
        time = np.asarray(self.cache["time"], dtype=np.float32)
        process = np.asarray(self.cache["process"][idx], dtype=np.float32)
        material_raw = np.asarray(self.cache["material"][idx], dtype=np.float32)
        stage = np.asarray(self.cache["stage"][idx], dtype=np.int32)
        descriptors = np.asarray(self.cache["process_descriptors"][idx], dtype=np.float32)
        geometry = np.asarray(self.cache["geometry_params"][idx], dtype=np.float32)
        shape_id = int(self.cache["shape_id"][idx])

        time_ids, depth_values = self._queries(idx)
        position = np.clip(depth_values / max(float(depth_grid[1] - depth_grid[0]), 1.0e-8), 0.0, self.nd - 1.0)
        i0 = np.floor(position).astype(np.int64)
        i1 = np.clip(i0 + 1, 0, self.nd - 1)
        frac = (position - i0).astype(np.float32)
        target = ((1.0 - frac) * profile[time_ids, i0] + frac * profile[time_ids, i1]).astype(np.float32)
        avg_target = (
            (1.0 - frac) * avg_profile[time_ids, i0] + frac * avg_profile[time_ids, i1]
        ).astype(np.float32)
        time_norm = time[time_ids] / max(float(time[-1]), 1.0)
        query = np.column_stack(
            [time_norm, depth_values / max(self.depth_max_mm, 1.0e-8), avg_target]
        ).astype(np.float32)
        shape_onehot = np.zeros(self.num_shapes, dtype=np.float32)
        shape_onehot[shape_id] = 1.0

        sample = {
            "residual_query": torch.from_numpy(query),
            "carbon": torch.from_numpy(target[:, None]),
            "avg_carbon": torch.from_numpy(avg_target[:, None]),
            "residual": torch.from_numpy((target - avg_target)[:, None]),
            "surface_weight": torch.from_numpy((depth_values <= 0.65).astype(np.float32)[:, None]),
            "exact_surface_weight": torch.from_numpy((depth_values <= 0.5 * depth_grid[1]).astype(np.float32)[:, None]),
            "final_weight": torch.from_numpy((time_ids == self.nt - 1).astype(np.float32)[:, None]),
            "initial_weight": torch.from_numpy((time_ids == 0).astype(np.float32)[:, None]),
            "depth_mm": torch.from_numpy(depth_values[:, None]),
            "time_id": torch.from_numpy(time_ids),
            "time": torch.from_numpy(time),
            "depth_grid_mm": torch.from_numpy(depth_grid),
            "carbon_profile_full": torch.from_numpy(profile),
            "avg_profile_full": torch.from_numpy(avg_profile),
            "process_raw": torch.from_numpy(process),
            "temporal_features": torch.from_numpy(
                _build_temporal_features(
                    process,
                    material_raw,
                    time,
                    stage,
                    diffusivity_model=self.diffusivity_model,
                )
            ),
            "process_descriptors": torch.from_numpy(_normalize_descriptors(descriptors)),
            "geometry_params": torch.from_numpy(_normalize_geometry_params(geometry)),
            "material": torch.from_numpy(_normalize_material(material_raw)),
            "material_raw": torch.from_numpy(material_raw),
            "shape_onehot": torch.from_numpy(shape_onehot),
            "shape_id": torch.tensor(shape_id, dtype=torch.long),
            "pair_id": torch.tensor(int(self.cache["pair_id"][idx]), dtype=torch.long),
            "variant_id": torch.tensor(int(self.cache["variant_id"][idx]), dtype=torch.long),
            "process_family_id": torch.tensor(int(self.cache["process_family_id"][idx]), dtype=torch.long),
            "dynamic_sensitivity_score": torch.tensor(
                float(self.cache["dynamic_sensitivity_score"][idx]), dtype=torch.float32
            ),
        }
        return sample


class CarburizingPairedProfileDataset(Dataset):
    """Complete process pairs evaluated at identical depth--time queries."""

    def __init__(self, file_path: str, queries_per_case: int = 2048, seed: int = 1234) -> None:
        self.dataset = CarburizingProfileQueryDataset(
            file_path,
            queries_per_case=queries_per_case,
            random_query=False,
            full_grid=False,
            seed=seed,
        )
        pair_ids = np.asarray(self.dataset.cache["pair_id"], dtype=np.int64)
        variant_ids = np.asarray(self.dataset.cache["variant_id"], dtype=np.int64)
        grouped: dict[int, list[int]] = {}
        for sample_index, pair_id in enumerate(pair_ids):
            grouped.setdefault(int(pair_id), []).append(sample_index)
        self.pairs = []
        for indices in grouped.values():
            if len(indices) != 2:
                continue
            ordered = sorted(indices, key=lambda sample_index: int(variant_ids[sample_index]))
            self.pairs.append((ordered[0], ordered[1]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        index_a, index_b = self.pairs[idx]
        sample_a = self.dataset[index_a]
        sample_b = self.dataset[index_b]
        return {f"a_{key}": value for key, value in sample_a.items()} | {
            f"b_{key}": value for key, value in sample_b.items()
        }


class GeoSurfacePairedResidualDataset(Dataset):
    """Pairs equal-mean dynamic histories and returns shared query points.

    The v2 pilot stores paired variants with the same geometry/material/mean Cp
    but different timing. Using deterministic queries gives both variants the
    same query coordinates/depths/times, so the residual difference is a clean
    timing-history contrast.
    """

    def __init__(
        self,
        file_path: str,
        queries_per_case: int = 4096,
        seed: int = 1234,
        preload: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        self.dataset = GeoSurfaceResidualQueryDataset(
            file_path,
            queries_per_case=queries_per_case,
            random_query=False,
            seed=seed,
            preload=preload,
            device=device,
        )
        with h5py.File(str(file_path), "r") as f:
            if "pair_id" not in f:
                self.pairs = []
            else:
                pair_ids = np.asarray(f["pair_id"][:], dtype=np.int64)
                variant_ids = np.asarray(f["variant_id"][:], dtype=np.int64) if "variant_id" in f else np.arange(len(pair_ids))
                grouped: dict[int, list[int]] = {}
                for i, pid in enumerate(pair_ids):
                    grouped.setdefault(int(pid), []).append(i)
                pairs: list[tuple[int, int]] = []
                for indices in grouped.values():
                    if len(indices) < 2:
                        continue
                    indices = sorted(indices, key=lambda i: int(variant_ids[i]))
                    pairs.append((indices[0], indices[1]))
                self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ia, ib = self.pairs[idx]
        a = self.dataset[ia]
        b = self.dataset[ib]
        return {f"a_{key}": value for key, value in a.items()} | {f"b_{key}": value for key, value in b.items()}
