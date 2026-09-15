# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from engineering_process import EngineeringCarburizingCase
from physics import (
    diffusivity_carbon_austenite,
    diffusivity_carbon_austenite_agren,
    effective_case_depth,
    equilibrium_surface_carbon,
    equilibrium_surface_carbon_from_potential,
)


def solve_engineering_depth(
    case: EngineeringCarburizingCase,
    depth_m: float,
    nx: int,
    nt: int,
    total_time_s: float,
    diffusivity_model: str = "legacy_parametric",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve 1-D near-surface carburizing with time-dependent ``h_m``.

    ``legacy_parametric`` preserves the original synthetic benchmark exactly.
    ``agren_binary`` uses the published composition- and temperature-dependent
    Ågren relation and is intended for the revised benchmark.
    """

    if diffusivity_model not in {"legacy_parametric", "agren_binary"}:
        raise ValueError(f"Unknown diffusivity_model={diffusivity_model!r}")

    def diffusivity(temp_c: float, carbon_wt: np.ndarray | float) -> np.ndarray:
        if diffusivity_model == "agren_binary":
            return diffusivity_carbon_austenite_agren(temp_c, carbon_wt)
        return diffusivity_carbon_austenite(
            temp_c,
            carbon_wt,
            case.d_ref,
            case.activation_j_mol,
        )

    depth = np.linspace(0.0, depth_m, nx, dtype=np.float64)
    time = np.linspace(0.0, total_time_s, nt, dtype=np.float64)
    process_time = np.linspace(0.0, total_time_s, len(case.cp), dtype=np.float64)
    dx = depth[1] - depth[0]

    carbon = np.full(nx, case.c0, dtype=np.float64)
    out = np.empty((nx, nt), dtype=np.float32)
    out[:, 0] = carbon.astype(np.float32)

    max_d = diffusivity(float(np.max(case.temperature_c)), float(np.max(case.cp)))
    stable_dt = 0.40 * dx * dx / max(float(max_d), 1e-20)

    for k in range(nt - 1):
        dt = float(time[k + 1] - time[k])
        n_sub = max(1, int(np.ceil(dt / stable_dt)))
        dt_sub = dt / n_sub
        for sub in range(n_sub):
            t_now = time[k] + (sub + 0.5) * dt_sub
            cp_now = float(np.interp(t_now, process_time, case.cp))
            temp_now = float(np.interp(t_now, process_time, case.temperature_c))
            h_now = float(np.interp(t_now, process_time, case.h_m))
            c_eq = float(
                equilibrium_surface_carbon_from_potential(cp_now)
                if diffusivity_model == "agren_binary"
                else equilibrium_surface_carbon(cp_now, temp_now)
            )

            d_nodes = diffusivity(temp_now, carbon)
            d_face = 0.5 * (d_nodes[:-1] + d_nodes[1:])
            flux = d_face * (carbon[1:] - carbon[:-1]) / dx

            rhs = np.zeros_like(carbon)
            rhs[1:-1] = (flux[1:] - flux[:-1]) / dx
            rhs[0] = 2.0 * flux[0] / dx + 2.0 * h_now * (c_eq - carbon[0]) / dx
            rhs[-1] = -2.0 * flux[-1] / dx
            carbon = np.clip(carbon + dt_sub * rhs, 0.02, 1.45)

        out[:, k + 1] = carbon.astype(np.float32)

    return out, depth.astype(np.float32), time.astype(np.float32)


def interp_profile_to_depth_map(
    carbon_depth_time: np.ndarray,
    depth_grid_m: np.ndarray,
    depth_map_mm: np.ndarray,
) -> np.ndarray:
    """Map C(depth,t) to a 2-D/3-D field using nearest-surface depth in mm."""

    depth_flat_m = np.nan_to_num(depth_map_mm, nan=-1.0).reshape(-1) * 1e-3
    fields = []
    for t in range(carbon_depth_time.shape[1]):
        values = np.interp(depth_flat_m, depth_grid_m, carbon_depth_time[:, t], left=np.nan, right=carbon_depth_time[-1, t])
        fields.append(values.reshape(depth_map_mm.shape))
    return np.stack(fields, axis=0).astype(np.float32)


def engineering_depth_metrics(pred: np.ndarray, truth: np.ndarray, depth_m: np.ndarray) -> dict[str, float]:
    diff = pred - truth
    rel = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1) / np.maximum(
        np.linalg.norm(truth.reshape(truth.shape[0], -1), axis=1), 1e-8
    )
    idx_05 = int(np.searchsorted(depth_m, 0.5e-3))
    idx_10 = int(np.searchsorted(depth_m, 1.0e-3))
    ecd = []
    for i in range(pred.shape[0]):
        ecd.append(
            abs(
                effective_case_depth(depth_m, pred[i, :, -1])
                - effective_case_depth(depth_m, truth[i, :, -1])
            )
        )
    return {
        "field_rel_l2": float(np.mean(rel)),
        "surface_c_mae_wt": float(np.mean(np.abs(diff[:, 0, :]))),
        "c_0p5mm_mae_wt": float(np.mean(np.abs(diff[:, idx_05, :]))),
        "c_1p0mm_mae_wt": float(np.mean(np.abs(diff[:, idx_10, :]))),
        "final_profile_mae_wt": float(np.mean(np.abs(diff[:, :, -1]))),
        "case_depth_mae_mm": float(np.mean(ecd) * 1e3),
    }
