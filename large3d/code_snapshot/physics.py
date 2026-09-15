# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Synthetic carburizing physics utilities used by the steel_carburizing example.
# The model is intentionally modest: 1-D diffusion in depth with a Robin surface
# boundary driven by carbon potential. It is a baseline digital-twin scaffold,
# not a substitute for calibrated thermodynamic/kinetic software.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


R_GAS = 8.31446261815324
MOLAR_MASS_C_G_MOL = 12.011
MOLAR_MASS_FE_G_MOL = 55.845


@dataclass(frozen=True)
class CarburizingCase:
    """One synthetic furnace run."""

    c0: float
    cp: np.ndarray
    temperature_c: np.ndarray
    h_m: float
    d_ref: float
    activation_j_mol: float


def diffusivity_carbon_austenite(
    temperature_c: np.ndarray,
    carbon_wt: np.ndarray | float,
    d_ref: float,
    activation_j_mol: float,
    reference_temperature_c: float = 930.0,
) -> np.ndarray:
    """Approximate carbon diffusivity in austenite.

    Parameters are chosen for synthetic studies. For publication work, replace
    them with a calibrated alloy-specific law or a thermodynamic database.
    """

    temp_k = np.asarray(temperature_c, dtype=np.float64) + 273.15
    ref_k = reference_temperature_c + 273.15
    arrhenius = np.exp((-activation_j_mol / R_GAS) * (1.0 / temp_k - 1.0 / ref_k))
    carbon_factor = 1.0 + 0.18 * np.asarray(carbon_wt, dtype=np.float64)
    return d_ref * arrhenius * carbon_factor


def carbon_weight_percent_to_mole_fraction(carbon_wt: np.ndarray | float) -> np.ndarray:
    """Convert carbon content expressed in wt% to Fe--C mole fraction.

    The conversion treats the matrix as binary Fe--C.  This is the composition
    convention required by the Ågren diffusivity relation below.
    """

    carbon_mass_fraction = np.asarray(carbon_wt, dtype=np.float64) / 100.0
    carbon_mass_fraction = np.clip(carbon_mass_fraction, 0.0, 0.20)
    n_c = carbon_mass_fraction / MOLAR_MASS_C_G_MOL
    n_fe = (1.0 - carbon_mass_fraction) / MOLAR_MASS_FE_G_MOL
    return n_c / np.maximum(n_c + n_fe, 1.0e-30)


def diffusivity_carbon_austenite_agren(
    temperature_c: np.ndarray | float,
    carbon_wt: np.ndarray | float,
) -> np.ndarray:
    """Carbon diffusivity in binary Fe--C austenite after Ågren (1986).

    The expression is reported in J. Ågren, *Scripta Metallurgica* 20
    (1986) 1507--1510, doi:10.1016/0036-9748(86)90384-4.  Temperature is
    supplied in degree Celsius, carbon in wt%, and the returned diffusivity is
    in m²/s.
    """

    temperature_k = np.asarray(temperature_c, dtype=np.float64) + 273.15
    x_c = carbon_weight_percent_to_mole_fraction(carbon_wt)
    y_c = x_c / np.maximum(1.0 - x_c, 1.0e-30)
    prefactor = 4.53e-7 * (1.0 + y_c * (1.0 - y_c) * 8339.9 / temperature_k)
    exponent = -(1.0 / temperature_k - 2.221e-4) * (17767.0 - 26436.0 * y_c)
    return prefactor * np.exp(exponent)


def equilibrium_surface_carbon(cp_wt: np.ndarray, temperature_c: np.ndarray) -> np.ndarray:
    """Map measured carbon potential to equilibrium surface carbon.

    This placeholder assumes the sensor carbon potential already expresses the
    atmosphere's equilibrium carbon content in wt%. A small temperature
    correction is included to keep the mapping explicit and easy to replace.
    """

    cp = np.asarray(cp_wt, dtype=np.float64)
    temp = np.asarray(temperature_c, dtype=np.float64)
    c_eq = cp * (1.0 - 1.5e-4 * (temp - 930.0))
    return np.clip(c_eq, 0.02, 1.35)


def equilibrium_surface_carbon_from_potential(cp_wt: np.ndarray | float) -> np.ndarray:
    """Use carbon potential directly as the gas/steel equilibrium carbon.

    Carbon potential is represented in wt% C and denotes the equilibrium
    carbon content imposed by the atmosphere.  The publication protocol uses
    this identity instead of an unsupported empirical temperature correction.
    """

    return np.clip(np.asarray(cp_wt, dtype=np.float64), 0.02, 1.35)


def sample_process(
    rng: np.random.Generator,
    nt: int,
    total_time_s: float,
) -> CarburizingCase:
    """Create one synthetic carburizing run with realistic-looking fluctuations."""

    time_h = np.linspace(0.0, total_time_s / 3600.0, nt)

    c0 = rng.uniform(0.16, 0.24)
    cp_mean = rng.uniform(0.78, 1.12)
    cp_amp = rng.uniform(0.0, 0.18)
    cp_period_h = rng.uniform(0.35, 3.0)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    cp = cp_mean + cp_amp * np.sin(2.0 * np.pi * time_h / cp_period_h + phase)

    # Add correlated atmosphere-generator fluctuation.
    ou = np.zeros(nt, dtype=np.float64)
    tau_h = rng.uniform(0.12, 0.8)
    dt_h = max(time_h[1] - time_h[0], 1e-6)
    decay = np.exp(-dt_h / tau_h)
    sigma = rng.uniform(0.01, 0.045)
    for i in range(1, nt):
        ou[i] = decay * ou[i - 1] + sigma * np.sqrt(1.0 - decay**2) * rng.normal()
    cp = cp + ou

    # Occasional generator step/pulse. This is critical for studying dynamics,
    # because equal-mean histories can still produce different case profiles.
    if rng.random() < 0.45:
        center = rng.integers(max(2, nt // 6), max(3, 5 * nt // 6))
        width = rng.integers(max(2, nt // 18), max(3, nt // 8))
        magnitude = rng.uniform(-0.09, 0.12)
        lo = max(0, center - width // 2)
        hi = min(nt, center + width // 2)
        cp[lo:hi] += magnitude
    cp = np.clip(cp, 0.55, 1.25)

    temp_base = rng.uniform(900.0, 950.0)
    temp_amp = rng.uniform(0.0, 5.0)
    temp_period_h = rng.uniform(1.0, 5.0)
    temperature_c = temp_base + temp_amp * np.sin(
        2.0 * np.pi * time_h / temp_period_h + rng.uniform(0.0, 2.0 * np.pi)
    )
    temperature_c += rng.normal(0.0, 0.65, size=nt)

    h_m = 10.0 ** rng.uniform(-8.4, -7.3)
    d_ref = 10.0 ** rng.uniform(-11.05, -10.65)
    activation = rng.uniform(132_000.0, 158_000.0)

    return CarburizingCase(
        c0=float(c0),
        cp=cp.astype(np.float32),
        temperature_c=temperature_c.astype(np.float32),
        h_m=float(h_m),
        d_ref=float(d_ref),
        activation_j_mol=float(activation),
    )


def solve_carburizing_1d(
    case: CarburizingCase,
    nx: int,
    nt: int,
    depth_m: float,
    total_time_s: float,
    internal_substeps: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Explicit finite-volume-like 1-D carburizing solver.

    Returns carbon field with shape ``[nx, nt]`` in wt%, depth coordinates in m,
    and time coordinates in s.
    """

    if nx < 8 or nt < 4:
        raise ValueError("nx >= 8 and nt >= 4 are required")

    x = np.linspace(0.0, depth_m, nx, dtype=np.float64)
    time = np.linspace(0.0, total_time_s, nt, dtype=np.float64)
    dx = x[1] - x[0]

    carbon = np.full(nx, case.c0, dtype=np.float64)
    out = np.empty((nx, nt), dtype=np.float32)
    out[:, 0] = carbon.astype(np.float32)

    max_d = diffusivity_carbon_austenite(
        np.max(case.temperature_c), np.max(case.cp), case.d_ref, case.activation_j_mol
    )
    stable_dt = 0.42 * dx * dx / max(float(max_d), 1e-20)

    for k in range(nt - 1):
        dt = time[k + 1] - time[k]
        n_sub = max(internal_substeps, int(np.ceil(dt / stable_dt)))
        dt_sub = dt / n_sub
        cp_left = float(case.cp[k])
        cp_right = float(case.cp[k + 1])
        t_left = float(case.temperature_c[k])
        t_right = float(case.temperature_c[k + 1])

        for s in range(n_sub):
            alpha = (s + 0.5) / n_sub
            cp_now = (1.0 - alpha) * cp_left + alpha * cp_right
            temp_now = (1.0 - alpha) * t_left + alpha * t_right

            d_nodes = diffusivity_carbon_austenite(
                temp_now, carbon, case.d_ref, case.activation_j_mol
            )
            c_eq = float(equilibrium_surface_carbon(cp_now, temp_now))

            rhs = np.zeros_like(carbon)
            d_face = 0.5 * (d_nodes[1:] + d_nodes[:-1])
            flux_internal = d_face * (carbon[1:] - carbon[:-1]) / dx

            rhs[1:-1] = (flux_internal[1:] - flux_internal[:-1]) / dx
            rhs[0] = 2.0 * flux_internal[0] / dx + 2.0 * case.h_m * (c_eq - carbon[0]) / dx
            rhs[-1] = -2.0 * flux_internal[-1] / dx

            carbon = carbon + dt_sub * rhs
            carbon = np.clip(carbon, 0.02, 1.45)

        out[:, k + 1] = carbon.astype(np.float32)

    return out, x.astype(np.float32), time.astype(np.float32)


def effective_case_depth(depth_m: np.ndarray, carbon_wt: np.ndarray, threshold: float = 0.40) -> float:
    """Depth where the final carbon profile first falls below a threshold."""

    above = carbon_wt >= threshold
    if not np.any(above):
        return 0.0
    if np.all(above):
        return float(depth_m[-1])
    idx = int(np.argmax(~above))
    x0, x1 = depth_m[idx - 1], depth_m[idx]
    c0, c1 = carbon_wt[idx - 1], carbon_wt[idx]
    if abs(float(c1 - c0)) < 1e-12:
        return float(x0)
    frac = (threshold - c0) / (c1 - c0)
    return float(x0 + frac * (x1 - x0))
