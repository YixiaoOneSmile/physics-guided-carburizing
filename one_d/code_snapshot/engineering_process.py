# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics import (
    diffusivity_carbon_austenite,
    diffusivity_carbon_austenite_agren,
    equilibrium_surface_carbon,
    equilibrium_surface_carbon_from_potential,
)


@dataclass(frozen=True)
class EngineeringCarburizingCase:
    c0: float
    cp: np.ndarray
    temperature_c: np.ndarray
    h_m: np.ndarray
    d_ref: float
    activation_j_mol: float
    stage: np.ndarray


PROCESS_CHANNELS_ENGINEERING = (
    "carbon_potential_wt",
    "temperature_c",
    "mass_transfer_m_s",
)

PROCESS_FAMILIES_ENGINEERING = (
    "early_boost",
    "late_boost",
    "lean_dip",
    "overshoot",
    "multi_pulse",
    "high_frequency_small_amplitude",
    "low_frequency_large_amplitude",
)

PROCESS_DESCRIPTOR_CHANNELS_ENGINEERING = (
    "cp_mean_wt",
    "cp_std_wt",
    "cp_min_wt",
    "cp_max_wt",
    "boost_time_fraction",
    "diffuse_time_fraction",
    "ceq_integral_wt_h",
    "diffusivity_integral_m2",
    "surface_drive_integral_wt_m",
)

# Publication-v4 keeps the same nine numerical descriptors as v3, but fixes
# the name and unit of channel 7.  The implementation has always divided the
# Ceq time integral by the process duration, so this quantity is a time
# average (wt%), not an integral (wt% h).
PROCESS_DESCRIPTOR_CHANNELS_PUBLICATION_V4_STRICT = (
    "cp_sample_mean_wt",
    "cp_std_wt",
    "cp_min_wt",
    "cp_max_wt",
    "boost_time_fraction",
    "diffuse_time_fraction",
    "ceq_time_average_wt",
    "diffusivity_integral_m2",
    "surface_drive_integral_wt_m",
)

MATERIAL_CHANNELS_ENGINEERING = (
    "c0_wt",
    "d_ref_m2_s",
    "activation_j_mol",
    "boost_h_m_s",
    "diffuse_h_m_s",
)

MATERIAL_CHANNELS_PUBLICATION_V3 = (
    "c0_wt",
    "boost_h_m_s",
    "diffuse_h_m_s",
)


@dataclass(frozen=True)
class PairedEngineeringProcess:
    case: EngineeringCarburizingCase
    average_case: EngineeringCarburizingCase
    process_family: str
    pair_id: int
    variant_id: int


def _match_mean_clip(cp: np.ndarray, target_mean: float, lo: float = 0.55, hi: float = 1.25) -> np.ndarray:
    out = cp.astype(np.float64).copy()
    for _ in range(12):
        out = np.clip(out + (target_mean - float(out.mean())), lo, hi)
    return out.astype(np.float32)


def time_average(values: np.ndarray, time_h: np.ndarray) -> float:
    """Return the continuous-time average under linear interpolation."""

    values64 = np.asarray(values, dtype=np.float64)
    time64 = np.asarray(time_h, dtype=np.float64)
    duration_h = float(time64[-1] - time64[0])
    if values64.ndim != 1 or time64.ndim != 1 or values64.shape != time64.shape:
        raise ValueError("values and time_h must be one-dimensional arrays with equal shape")
    if duration_h <= 0.0 or np.any(np.diff(time64) <= 0.0):
        raise ValueError("time_h must be strictly increasing")
    return float(np.trapezoid(values64, time64) / duration_h)


def _match_time_average_clip(
    cp: np.ndarray,
    time_h: np.ndarray,
    target_mean: float,
    lo: float = 0.55,
    hi: float = 1.25,
) -> np.ndarray:
    """Match a linearly interpolated time average after enforcing bounds.

    A scalar offset has a monotone effect after clipping, so bisection is
    robust even when a process history touches either carbon-potential bound.
    The final float32 correction keeps the stored history within numerical
    tolerance of the requested physical time average.
    """

    source = np.asarray(cp, dtype=np.float64)
    time64 = np.asarray(time_h, dtype=np.float64)
    if not lo <= target_mean <= hi:
        raise ValueError("target_mean must lie within the clipping bounds")

    lower = lo - float(np.max(source))
    upper = hi - float(np.min(source))
    for _ in range(80):
        offset = 0.5 * (lower + upper)
        current = time_average(np.clip(source + offset, lo, hi), time64)
        if current < target_mean:
            lower = offset
        else:
            upper = offset

    out = np.clip(source + 0.5 * (lower + upper), lo, hi).astype(np.float32)
    # Usually no value is clipped in the publication design.  A few bounded
    # fixed-point steps also make the float32 representation deterministic.
    for _ in range(8):
        error = target_mean - time_average(out, time64)
        if abs(error) <= 2.0e-8:
            break
        out = np.clip(out.astype(np.float64) + error, lo, hi).astype(np.float32)
    return out


def _base_stage_schedule(
    rng: np.random.Generator,
    nt: int,
    total_time_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_h = np.linspace(0.0, total_time_s / 3600.0, nt, dtype=np.float64)
    total_h = max(float(time_h[-1]), 1e-6)
    stage = np.full(nt, 2, dtype=np.int32)

    boost_fraction = float(rng.uniform(0.38, 0.58))
    if rng.random() < 0.35:
        first_end = boost_fraction * total_h * rng.uniform(0.55, 0.82)
        second_start = total_h * rng.uniform(0.45, 0.66)
        second_end = min(total_h, second_start + boost_fraction * total_h * rng.uniform(0.22, 0.36))
        stage[(time_h <= first_end) | ((time_h >= second_start) & (time_h <= second_end))] = 1
    else:
        stage[time_h <= boost_fraction * total_h] = 1

    boost_h = float(10.0 ** rng.uniform(-7.70, -7.20))
    diffuse_h = float(10.0 ** rng.uniform(-9.35, -8.45))
    h_m = np.where(stage == 1, boost_h, diffuse_h).astype(np.float64)

    boost_temp = float(rng.uniform(900.0, 940.0))
    diffuse_temp = float(rng.uniform(830.0, min(925.0, boost_temp - 2.0)))
    temp = np.where(stage == 1, boost_temp, diffuse_temp).astype(np.float64)
    temp += rng.uniform(0.6, 2.5) * np.sin(
        2.0 * np.pi * time_h / rng.uniform(1.4, max(1.5, total_h))
        + rng.uniform(0.0, 2.0 * np.pi)
    )
    temp = np.clip(temp, 820.0, 950.0)
    return time_h, temp, h_m, stage


def _pulse(time_h: np.ndarray, center_h: float, width_h: float) -> np.ndarray:
    width = max(float(width_h), 1e-4)
    return np.exp(-0.5 * ((time_h - center_h) / width) ** 2)


def _family_cp(
    family: str,
    variant_id: int,
    time_h: np.ndarray,
    stage: np.ndarray,
    target_mean: float,
    rng: np.random.Generator,
) -> np.ndarray:
    total_h = max(float(time_h[-1]), 1e-6)
    cp = np.where(stage == 1, target_mean + 0.12, target_mean - 0.10).astype(np.float64)
    amp = float(rng.uniform(0.06, 0.14))

    if family == "early_boost":
        center = total_h * (0.16 if variant_id == 0 else 0.34)
        cp += amp * _pulse(time_h, center, total_h * 0.07)
    elif family == "late_boost":
        center = total_h * (0.66 if variant_id == 0 else 0.84)
        cp += amp * _pulse(time_h, center, total_h * 0.08)
    elif family == "lean_dip":
        center = total_h * (0.32 if variant_id == 0 else 0.68)
        cp -= amp * _pulse(time_h, center, total_h * 0.075)
    elif family == "overshoot":
        center = total_h * (0.28 if variant_id == 0 else 0.58)
        cp += 1.15 * amp * _pulse(time_h, center, total_h * 0.045)
        cp -= 0.45 * amp * _pulse(time_h, min(total_h, center + total_h * 0.16), total_h * 0.09)
    elif family == "multi_pulse":
        centers = (0.18, 0.42, 0.72) if variant_id == 0 else (0.28, 0.55, 0.84)
        signs = (1.0, -0.7, 0.9)
        for c, s in zip(centers, signs):
            cp += s * amp * _pulse(time_h, total_h * c, total_h * 0.045)
    elif family == "high_frequency_small_amplitude":
        phase = 0.0 if variant_id == 0 else np.pi / 2.0
        cp += 0.45 * amp * np.sin(2.0 * np.pi * time_h / max(total_h / 7.0, 1e-3) + phase)
    elif family == "low_frequency_large_amplitude":
        phase = 0.0 if variant_id == 0 else np.pi
        cp += 1.10 * amp * np.sin(2.0 * np.pi * time_h / max(total_h * 1.2, 1e-3) + phase)
    else:
        raise ValueError(f"Unknown process family {family}")

    # A small deterministic controller ripple keeps the histories realistic
    # while preserving the equal-mean pairing after the final normalization.
    cp += 0.006 * np.sin(2.0 * np.pi * time_h / max(total_h / 3.0, 1e-3) + 0.73 * variant_id)
    return _match_mean_clip(cp, target_mean)


def sample_paired_dynamic_processes(
    rng: np.random.Generator,
    nt: int,
    total_time_s: float,
    pair_id: int,
    family: str | None = None,
) -> tuple[PairedEngineeringProcess, PairedEngineeringProcess]:
    """Create two equal-mean dynamic histories with shared material and T/h_m.

    The two returned cases intentionally have the same geometry/material-facing
    state except for the timing of ``Cp(t)``. This makes average-Cp baselines
    vulnerable in the near-surface and late-profile metrics.
    """

    family = family or PROCESS_FAMILIES_ENGINEERING[pair_id % len(PROCESS_FAMILIES_ENGINEERING)]
    if family not in PROCESS_FAMILIES_ENGINEERING:
        raise ValueError(f"Unknown process family {family}")

    time_h, temp, h_m, stage = _base_stage_schedule(rng, nt, total_time_s)
    target_mean = float(rng.uniform(0.82, 0.94))
    c0 = float(rng.uniform(0.16, 0.23))
    d_ref = float(10.0 ** rng.uniform(-11.0, -10.65))
    activation = float(rng.uniform(135_000.0, 155_000.0))

    samples = []
    for variant_id in (0, 1):
        cp = _family_cp(family, variant_id, time_h, stage, target_mean, rng)
        case = EngineeringCarburizingCase(
            c0=c0,
            cp=cp,
            temperature_c=temp.astype(np.float32),
            h_m=h_m.astype(np.float32),
            d_ref=d_ref,
            activation_j_mol=activation,
            stage=stage.copy(),
        )
        samples.append(
            PairedEngineeringProcess(
                case=case,
                average_case=average_cp_case(case),
                process_family=family,
                pair_id=pair_id,
                variant_id=variant_id,
            )
        )
    return samples[0], samples[1]


def _base_stage_schedule_v3(
    rng: np.random.Generator,
    nt: int,
    total_time_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Physically bounded gas-carburizing schedule for the revised benchmark.

    Temperatures remain in the austenitic gas-carburizing range.  Carbon
    transfer coefficients are centred on literature-scale values of order
    10^-7 m/s instead of dropping by two orders of magnitude in the diffuse
    stage.
    """

    time_h = np.linspace(0.0, total_time_s / 3600.0, nt, dtype=np.float64)
    total_h = max(float(time_h[-1]), 1.0e-6)
    stage = np.full(nt, 2, dtype=np.int32)
    boost_fraction = float(rng.uniform(0.40, 0.58))
    if rng.random() < 0.30:
        first_end = total_h * boost_fraction * rng.uniform(0.58, 0.76)
        second_start = total_h * rng.uniform(0.52, 0.68)
        second_width = total_h * boost_fraction * rng.uniform(0.24, 0.34)
        stage[(time_h <= first_end) | ((time_h >= second_start) & (time_h <= second_start + second_width))] = 1
    else:
        stage[time_h <= total_h * boost_fraction] = 1

    diffuse_h = float(rng.uniform(0.8e-7, 1.5e-7))
    boost_h = float(min(2.6e-7, diffuse_h * rng.uniform(1.25, 1.75)))
    h_m = np.where(stage == 1, boost_h, diffuse_h).astype(np.float64)

    diffuse_temp = float(rng.uniform(890.0, 925.0))
    boost_temp = float(rng.uniform(max(905.0, diffuse_temp), 950.0))
    temperature = np.where(stage == 1, boost_temp, diffuse_temp).astype(np.float64)
    temperature += rng.uniform(0.5, 2.0) * np.sin(
        2.0 * np.pi * time_h / rng.uniform(1.5, max(1.6, total_h))
        + rng.uniform(0.0, 2.0 * np.pi)
    )
    return time_h, np.clip(temperature, 885.0, 955.0), h_m, stage


def _family_cp_v3(
    family: str,
    variant_id: int,
    time_h: np.ndarray,
    stage: np.ndarray,
    target_mean: float,
    amplitude: float,
) -> np.ndarray:
    """Generate equal-mean histories with one amplitude shared by each pair."""

    total_h = max(float(time_h[-1]), 1.0e-6)
    base = np.where(stage == 1, target_mean + 0.045, target_mean - 0.040).astype(np.float64)
    signal = np.zeros_like(time_h, dtype=np.float64)

    if family == "early_boost":
        center = total_h * (0.14 if variant_id == 0 else 0.38)
        signal = amplitude * _pulse(time_h, center, total_h * 0.075)
    elif family == "late_boost":
        center = total_h * (0.58 if variant_id == 0 else 0.84)
        signal = amplitude * _pulse(time_h, center, total_h * 0.075)
    elif family == "lean_dip":
        center = total_h * (0.25 if variant_id == 0 else 0.72)
        signal = -amplitude * _pulse(time_h, center, total_h * 0.085)
    elif family == "overshoot":
        center = total_h * (0.22 if variant_id == 0 else 0.66)
        signal += 1.15 * amplitude * _pulse(time_h, center, total_h * 0.050)
        signal -= 0.55 * amplitude * _pulse(time_h, center + total_h * 0.14, total_h * 0.085)
    elif family == "multi_pulse":
        centers = (0.14, 0.38, 0.68) if variant_id == 0 else (0.30, 0.58, 0.86)
        for center, sign in zip(centers, (1.0, -0.75, 0.90)):
            signal += sign * amplitude * _pulse(time_h, total_h * center, total_h * 0.050)
    elif family == "high_frequency_small_amplitude":
        phase = 0.0 if variant_id == 0 else np.pi / 2.0
        signal = amplitude * np.sin(2.0 * np.pi * 7.0 * time_h / total_h + phase)
    elif family == "low_frequency_large_amplitude":
        phase = 0.0 if variant_id == 0 else np.pi
        signal = amplitude * np.sin(2.0 * np.pi * time_h / (1.2 * total_h) + phase)
    else:
        raise ValueError(f"Unknown process family {family}")

    # Remove the discrete-time mean before combining with the common base.
    # The final correction accounts for the non-zero mean of the stage base.
    signal -= float(np.mean(signal))
    return _match_mean_clip(base + signal, target_mean, lo=0.55, hi=1.20)


def sample_paired_dynamic_processes_v3(
    rng: np.random.Generator,
    nt: int,
    total_time_s: float,
    pair_id: int,
    family: str | None = None,
) -> tuple[PairedEngineeringProcess, PairedEngineeringProcess]:
    """Create a revised equal-mean pair for the publication benchmark.

    Both variants share material, temperature, transfer coefficient, stage
    schedule, and perturbation amplitude.  Their only prescribed difference is
    the temporal placement or phase of the carbon-potential perturbation.
    """

    family = family or PROCESS_FAMILIES_ENGINEERING[pair_id % len(PROCESS_FAMILIES_ENGINEERING)]
    if family not in PROCESS_FAMILIES_ENGINEERING:
        raise ValueError(f"Unknown process family {family}")
    time_h, temperature, h_m, stage = _base_stage_schedule_v3(rng, nt, total_time_s)
    target_mean = float(rng.uniform(0.82, 0.96))
    if family == "high_frequency_small_amplitude":
        amplitude = float(rng.uniform(0.055, 0.090))
    elif family == "low_frequency_large_amplitude":
        amplitude = float(rng.uniform(0.085, 0.135))
    else:
        amplitude = float(rng.uniform(0.11, 0.18))

    c0 = float(rng.uniform(0.16, 0.24))
    # Legacy kinetic fields are retained for dataclass/API compatibility.  The
    # revised solver selects the published Ågren relation explicitly.
    d_ref = 2.0e-11
    activation = 145_000.0
    samples: list[PairedEngineeringProcess] = []
    for variant_id in (0, 1):
        cp = _family_cp_v3(family, variant_id, time_h, stage, target_mean, amplitude)
        case = EngineeringCarburizingCase(
            c0=c0,
            cp=cp,
            temperature_c=temperature.astype(np.float32),
            h_m=h_m.astype(np.float32),
            d_ref=d_ref,
            activation_j_mol=activation,
            stage=stage.copy(),
        )
        samples.append(
            PairedEngineeringProcess(
                case=case,
                average_case=average_cp_case(case),
                process_family=family,
                pair_id=pair_id,
                variant_id=variant_id,
            )
        )
    return samples[0], samples[1]


def sample_paired_dynamic_processes_v4_strict(
    rng: np.random.Generator,
    nt: int,
    total_time_s: float,
    pair_id: int,
    family: str | None = None,
) -> tuple[PairedEngineeringProcess, PairedEngineeringProcess]:
    """Create publication pairs with an equal continuous-time Cp average.

    This is a versioned correction of the v3 design.  It intentionally uses
    the same random draws, bounds, schedules, and perturbation amplitudes as
    v3; only the definition used for the final Cp offset and Average-Cp prior
    changes.  The legacy v3 sampler remains untouched and reproducible.
    """

    family = family or PROCESS_FAMILIES_ENGINEERING[pair_id % len(PROCESS_FAMILIES_ENGINEERING)]
    if family not in PROCESS_FAMILIES_ENGINEERING:
        raise ValueError(f"Unknown process family {family}")
    time_h, temperature, h_m, stage = _base_stage_schedule_v3(rng, nt, total_time_s)
    # Quantize the shared reference once because histories and HDF5 fields are
    # stored as float32.  Both variants then use the identical stored prior.
    target_mean = float(np.float32(rng.uniform(0.82, 0.96)))
    if family == "high_frequency_small_amplitude":
        amplitude = float(rng.uniform(0.055, 0.090))
    elif family == "low_frequency_large_amplitude":
        amplitude = float(rng.uniform(0.085, 0.135))
    else:
        amplitude = float(rng.uniform(0.11, 0.18))

    c0 = float(rng.uniform(0.16, 0.24))
    d_ref = 2.0e-11
    activation = 145_000.0
    samples: list[PairedEngineeringProcess] = []
    for variant_id in (0, 1):
        # Reuse the v3 waveform exactly, then replace its arithmetic-mean
        # normalization with a continuous-time normalization.
        cp_v3 = _family_cp_v3(family, variant_id, time_h, stage, target_mean, amplitude)
        cp = _match_time_average_clip(cp_v3, time_h, target_mean, lo=0.55, hi=1.20)
        case = EngineeringCarburizingCase(
            c0=c0,
            cp=cp,
            temperature_c=temperature.astype(np.float32),
            h_m=h_m.astype(np.float32),
            d_ref=d_ref,
            activation_j_mol=activation,
            stage=stage.copy(),
        )
        samples.append(
            PairedEngineeringProcess(
                case=case,
                average_case=average_cp_case_time_weighted(case, time_h, target_mean),
                process_family=family,
                pair_id=pair_id,
                variant_id=variant_id,
            )
        )
    return samples[0], samples[1]


def _add_controller_disturbance(cp: np.ndarray, time_h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    total_h = max(float(time_h[-1] - time_h[0]), 1e-6)
    drift = rng.uniform(0.006, 0.025) * np.sin(
        2.0 * np.pi * time_h / rng.uniform(total_h * 0.6, total_h * 1.6) + rng.uniform(0.0, 2.0 * np.pi)
    )
    cp = cp + drift

    if rng.random() < 0.75:
        width = rng.uniform(0.08, 0.35)
        start = rng.uniform(0.1, max(0.2, total_h - width - 0.1))
        amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.025, 0.08)
        cp[(time_h >= start) & (time_h <= start + width)] += amp

    ou = np.zeros_like(cp)
    dt_h = max(float(time_h[1] - time_h[0]), 1e-6)
    tau_h = rng.uniform(0.08, 0.30)
    decay = np.exp(-dt_h / tau_h)
    sigma = rng.uniform(0.002, 0.010)
    for i in range(1, len(cp)):
        ou[i] = decay * ou[i - 1] + sigma * np.sqrt(1.0 - decay**2) * rng.normal()
    return cp + ou


def sample_open_engineering_process(
    rng: np.random.Generator,
    nt: int,
    total_time_s: float,
) -> EngineeringCarburizingCase:
    """Open, COSMAP-independent boost/diffuse gas carburizing process."""

    time_h = np.linspace(0.0, total_time_s / 3600.0, nt, dtype=np.float64)
    total_h = max(float(time_h[-1]), 1e-6)
    cp = np.empty(nt, dtype=np.float64)
    temp = np.empty(nt, dtype=np.float64)
    h_m = np.empty(nt, dtype=np.float64)
    stage = np.zeros(nt, dtype=np.int32)

    n_cycles = int(rng.choice([1, 2, 3]))
    cycle_edges = np.linspace(0.0, total_h, n_cycles + 1)
    boost_h = float(10.0 ** rng.uniform(-7.75, -7.25))
    diffuse_h = float(10.0 ** rng.uniform(-9.3, -8.35))
    base_boost_temp = float(rng.uniform(890.0, 940.0))
    base_diffuse_temp = float(rng.uniform(830.0, min(930.0, base_boost_temp)))

    for cycle in range(n_cycles):
        c0 = cycle_edges[cycle]
        c1 = cycle_edges[cycle + 1]
        boost_fraction = float(rng.uniform(0.38, 0.68 if n_cycles == 1 else 0.55))
        boost_end = c0 + boost_fraction * (c1 - c0)
        boost_mask = (time_h >= c0) & (time_h <= boost_end)
        diffuse_mask = (time_h > boost_end) & (time_h <= c1)

        cp[boost_mask] = rng.uniform(0.95, 1.15)
        cp[diffuse_mask] = rng.uniform(0.70, 0.90)
        temp[boost_mask] = base_boost_temp + rng.normal(0.0, 1.2)
        temp[diffuse_mask] = base_diffuse_temp + rng.normal(0.0, 1.2)
        h_m[boost_mask] = boost_h
        h_m[diffuse_mask] = diffuse_h
        stage[boost_mask] = 1
        stage[diffuse_mask] = 2

    cp = _add_controller_disturbance(cp, time_h, rng)
    cp = np.where(stage == 1, np.clip(cp, 0.88, 1.22), np.clip(cp, 0.62, 0.98))
    temp += rng.uniform(0.0, 2.0) * np.sin(
        2.0 * np.pi * time_h / rng.uniform(1.2, max(1.3, total_h)) + rng.uniform(0.0, 2.0 * np.pi)
    )
    temp = np.clip(temp, 820.0, 950.0)

    return EngineeringCarburizingCase(
        c0=float(rng.uniform(0.16, 0.23)),
        cp=cp.astype(np.float32),
        temperature_c=temp.astype(np.float32),
        h_m=h_m.astype(np.float32),
        d_ref=float(10.0 ** rng.uniform(-11.0, -10.65)),
        activation_j_mol=float(rng.uniform(135_000.0, 155_000.0)),
        stage=stage,
    )


def process_matrix(case: EngineeringCarburizingCase) -> np.ndarray:
    return np.stack([case.cp, case.temperature_c, case.h_m], axis=0).astype(np.float32)


def process_descriptors(case: EngineeringCarburizingCase, total_time_s: float | None = None) -> np.ndarray:
    total_time_s = float(total_time_s or 1.0)
    time_s = np.linspace(0.0, total_time_s, len(case.cp), dtype=np.float64)
    time_h = time_s / 3600.0
    ceq = equilibrium_surface_carbon(case.cp, case.temperature_c)
    diffusivity = diffusivity_carbon_austenite(
        case.temperature_c,
        np.full_like(case.cp, case.c0, dtype=np.float32),
        case.d_ref,
        case.activation_j_mol,
    )
    surface_drive = case.h_m * (ceq - case.c0)
    total_h = max(float(time_h[-1] - time_h[0]), 1e-6)
    return np.asarray(
        [
            float(np.mean(case.cp)),
            float(np.std(case.cp)),
            float(np.min(case.cp)),
            float(np.max(case.cp)),
            float(np.mean(case.stage == 1)),
            float(np.mean(case.stage == 2)),
            float(np.trapezoid(ceq, time_h) / total_h),
            float(np.trapezoid(diffusivity, time_s)),
            float(np.trapezoid(surface_drive, time_s)),
        ],
        dtype=np.float32,
    )


def process_descriptors_v3(case: EngineeringCarburizingCase, total_time_s: float | None = None) -> np.ndarray:
    """Process descriptors consistent with the publication-v3 physics."""

    total_time_s = float(total_time_s or 1.0)
    time_s = np.linspace(0.0, total_time_s, len(case.cp), dtype=np.float64)
    time_h = time_s / 3600.0
    ceq = equilibrium_surface_carbon_from_potential(case.cp)
    diffusivity = diffusivity_carbon_austenite_agren(case.temperature_c, np.full_like(case.cp, case.c0))
    surface_drive = case.h_m * (ceq - case.c0)
    total_h = max(float(time_h[-1] - time_h[0]), 1e-6)
    return np.asarray(
        [
            float(np.mean(case.cp)),
            float(np.std(case.cp)),
            float(np.min(case.cp)),
            float(np.max(case.cp)),
            float(np.mean(case.stage == 1)),
            float(np.mean(case.stage == 2)),
            float(np.trapezoid(ceq, time_h) / total_h),
            float(np.trapezoid(diffusivity, time_s)),
            float(np.trapezoid(surface_drive, time_s)),
        ],
        dtype=np.float32,
    )


def average_cp_case(case: EngineeringCarburizingCase) -> EngineeringCarburizingCase:
    return EngineeringCarburizingCase(
        c0=case.c0,
        cp=np.full_like(case.cp, float(np.mean(case.cp))),
        temperature_c=case.temperature_c.copy(),
        h_m=case.h_m.copy(),
        d_ref=case.d_ref,
        activation_j_mol=case.activation_j_mol,
        stage=case.stage.copy(),
    )


def average_cp_case_time_weighted(
    case: EngineeringCarburizingCase,
    time_h: np.ndarray,
    reference_mean: float | None = None,
) -> EngineeringCarburizingCase:
    """Construct a constant-Cp prior from the linear-interpolation time mean."""

    mean_cp = time_average(case.cp, time_h) if reference_mean is None else float(reference_mean)
    return EngineeringCarburizingCase(
        c0=case.c0,
        cp=np.full_like(case.cp, mean_cp),
        temperature_c=case.temperature_c.copy(),
        h_m=case.h_m.copy(),
        d_ref=case.d_ref,
        activation_j_mol=case.activation_j_mol,
        stage=case.stage.copy(),
    )
