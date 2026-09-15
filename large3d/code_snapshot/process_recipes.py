# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics import CarburizingCase


@dataclass(frozen=True)
class ProductionRecipe:
    name: str
    cp_boost: float
    cp_diffuse: float
    temperature_c: float
    n_cycles: int
    total_time_s: float


def _piecewise_boost_diffuse(
    time_h: np.ndarray,
    cp_boost: float,
    cp_diffuse: float,
    n_cycles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total_h = float(time_h[-1] - time_h[0])
    cycle_h = total_h / n_cycles
    cp = np.empty_like(time_h, dtype=np.float64)

    for cycle in range(n_cycles):
        start = cycle * cycle_h
        end = (cycle + 1) * cycle_h
        boost_fraction = rng.uniform(0.28, 0.48)
        boost_end = start + boost_fraction * cycle_h
        cycle_mask = (time_h >= start) & (time_h <= end)
        boost_mask = cycle_mask & (time_h <= boost_end)
        diffuse_mask = cycle_mask & ~boost_mask
        cp[boost_mask] = cp_boost + rng.normal(0.0, 0.008)
        cp[diffuse_mask] = cp_diffuse + rng.normal(0.0, 0.006)

    # Smooth controller transitions instead of unphysical square steps.
    if len(cp) >= 7:
        kernel = np.asarray([1, 2, 3, 2, 1], dtype=np.float64)
        kernel = kernel / kernel.sum()
        cp = np.convolve(cp, kernel, mode="same")
        cp[:2] = cp[2]
        cp[-2:] = cp[-3]
    return cp


def _add_atmosphere_generator_disturbance(
    cp: np.ndarray,
    time_h: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add measured, not filtered-out, production-like Cp disturbances."""

    total_h = float(time_h[-1] - time_h[0])
    drift_amp = rng.uniform(0.008, 0.035)
    drift_period_h = rng.uniform(max(0.8, total_h / 3.0), max(1.0, total_h * 1.4))
    cp = cp + drift_amp * np.sin(2.0 * np.pi * time_h / drift_period_h + rng.uniform(0.0, 2.0 * np.pi))

    # Ornstein-Uhlenbeck controller hunting / generator instability.
    ou = np.zeros_like(cp)
    dt_h = max(float(time_h[1] - time_h[0]), 1e-6)
    tau_h = rng.uniform(0.08, 0.35)
    decay = np.exp(-dt_h / tau_h)
    sigma = rng.uniform(0.004, 0.018)
    for i in range(1, len(cp)):
        ou[i] = decay * ou[i - 1] + sigma * np.sqrt(1.0 - decay**2) * rng.normal()
    cp = cp + ou

    # Occasional enriching-gas overshoot or lean dip, visible to a correct sensor.
    if rng.random() < 0.65:
        width_h = rng.uniform(0.12, 0.45)
        start_h = rng.uniform(0.1, max(0.2, total_h - width_h - 0.1))
        magnitude = rng.choice([-1.0, 1.0]) * rng.uniform(0.035, 0.11)
        mask = (time_h >= start_h) & (time_h <= start_h + width_h)
        cp[mask] += magnitude
    return np.clip(cp, 0.55, 1.25)


def sample_gas_carburizing_recipe(
    rng: np.random.Generator,
    nt: int,
    total_time_s: float,
) -> tuple[CarburizingCase, ProductionRecipe]:
    """Sample a production-like gas carburizing boost-diffuse recipe.

    This represents the high-temperature carburizing hold. Heat-up,
    quenching, and tempering are intentionally excluded from the diffusion
    solve, because the present model predicts carbon concentration rather than
    phase transformation or hardness.
    """

    time_h = np.linspace(0.0, total_time_s / 3600.0, nt)
    temperature_c = float(rng.choice([920.0, 930.0, 940.0]) + rng.normal(0.0, 2.0))
    cp_boost = float(rng.uniform(1.02, 1.15))
    cp_diffuse = float(rng.uniform(0.76, 0.90))
    n_cycles = int(rng.choice([2, 3, 4]))

    cp = _piecewise_boost_diffuse(time_h, cp_boost, cp_diffuse, n_cycles, rng)
    cp = _add_atmosphere_generator_disturbance(cp, time_h, rng)

    # Furnace uniformity and load disturbance around a controlled setpoint.
    temp = np.full(nt, temperature_c, dtype=np.float64)
    temp += rng.normal(0.0, 0.6, size=nt)
    temp += rng.uniform(0.0, 2.5) * np.sin(
        2.0 * np.pi * time_h / rng.uniform(1.5, 5.0) + rng.uniform(0.0, 2.0 * np.pi)
    )
    temp = np.clip(temp, 900.0, 950.0)

    case = CarburizingCase(
        c0=float(rng.uniform(0.16, 0.23)),
        cp=cp.astype(np.float32),
        temperature_c=temp.astype(np.float32),
        h_m=float(10.0 ** rng.uniform(-8.1, -7.25)),
        d_ref=float(10.0 ** rng.uniform(-11.0, -10.55)),
        activation_j_mol=float(rng.uniform(135_000.0, 155_000.0)),
    )
    recipe = ProductionRecipe(
        name="gas_boost_diffuse",
        cp_boost=cp_boost,
        cp_diffuse=cp_diffuse,
        temperature_c=temperature_c,
        n_cycles=n_cycles,
        total_time_s=total_time_s,
    )
    return case, recipe


def sample_process_for_mode(
    rng: np.random.Generator,
    nt: int,
    total_time_s: float,
    mode: str,
) -> tuple[CarburizingCase, str]:
    if mode == "production":
        case, recipe = sample_gas_carburizing_recipe(rng, nt, total_time_s)
        return case, recipe.name
    if mode == "synthetic":
        from physics import sample_process

        return sample_process(rng, nt, total_time_s), "synthetic_random"
    raise ValueError("mode must be 'production' or 'synthetic'")
