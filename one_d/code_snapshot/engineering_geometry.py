# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


ENGINEERING_SHAPES = (
    "slab",
    "cylinder",
    "ring",
    "stepped_shaft",
    "notched_block",
    "gear_ring",
)

GEOMETRY_PARAM_NAMES = (
    "domain_x_mm",
    "domain_y_mm",
    "domain_z_mm",
    "outer_radius_mm",
    "inner_radius_mm",
    "length_mm",
    "width_mm",
    "height_mm",
    "module_mm",
    "teeth",
    "root_radius_mm",
    "tip_radius_mm",
    "notch_width_mm",
    "notch_depth_mm",
    "large_radius_mm",
    "small_radius_mm",
    "pressure_angle_deg",
)


@dataclass(frozen=True)
class EngineeringGeometry:
    shape_id: int
    shape_name: str
    params: np.ndarray
    mask3d: np.ndarray
    sdf_mm: np.ndarray
    depth_to_surface_mm: np.ndarray
    local_thickness_mm: np.ndarray
    coords_mm: np.ndarray
    section_mask: np.ndarray
    section_depth_mm: np.ndarray
    section_x_mm: np.ndarray
    section_y_mm: np.ndarray


def _axis(span_mm: float, n: int) -> np.ndarray:
    return np.linspace(-0.5 * span_mm, 0.5 * span_mm, n, dtype=np.float32)


def _gear_outer_radius(theta: np.ndarray, root: float, tip: float, teeth: int) -> np.ndarray:
    pitch = 2.0 * np.pi / teeth
    phase = ((theta + 0.5 * pitch) % pitch) / pitch - 0.5
    a = np.abs(phase)
    top_half = 0.23
    flank_half = 0.40
    radius = np.full_like(theta, root, dtype=np.float64)
    radius[a <= top_half] = tip
    flank = (a > top_half) & (a <= flank_half)
    frac = (flank_half - a[flank]) / (flank_half - top_half)
    radius[flank] = root + (tip - root) * frac
    return radius


def _mask_from_xy(shape_name: str, x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    outer = params[3]
    inner = params[4]
    width = params[6]
    height = params[7]
    teeth = max(8, int(round(params[9])))
    root = params[10]
    tip = params[11]

    if shape_name == "slab":
        return (np.abs(x) <= 0.5 * width) & (np.abs(y) <= 0.5 * height)
    if shape_name == "cylinder":
        return r <= outer
    if shape_name == "ring":
        return (r <= outer) & (r >= inner)
    if shape_name == "stepped_shaft":
        return r <= params[14]
    if shape_name == "notched_block":
        mask = (np.abs(x) <= 0.5 * width) & (np.abs(y) <= 0.5 * height)
        notch = (x > 0.5 * width - params[13]) & (np.abs(y) <= 0.5 * params[12])
        return mask & ~notch

    gear_outer = _gear_outer_radius(theta, root, tip, teeth)
    return (r <= gear_outer) & (r >= inner)


def _mask_from_xyz(shape_name: str, x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    xy = _mask_from_xy(shape_name, x, y, params)
    length = params[5]
    if shape_name == "stepped_shaft":
        r = np.sqrt(x**2 + y**2)
        radius = np.where(z < 0.0, params[14], params[15])
        return (r <= radius) & (np.abs(z) <= 0.5 * length)
    return xy & (np.abs(z) <= 0.5 * length)


def _distances(mask: np.ndarray, spacing: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    inside = ndimage.distance_transform_edt(mask, sampling=spacing)
    outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
    sdf = inside - outside
    depth = np.where(mask, inside, np.nan)
    return sdf.astype(np.float32), depth.astype(np.float32)


def _params_for_shape(shape_name: str, rng: np.random.Generator) -> np.ndarray:
    p = np.zeros(len(GEOMETRY_PARAM_NAMES), dtype=np.float32)
    if shape_name == "slab":
        width = rng.uniform(30.0, 80.0)
        height = rng.uniform(15.0, 50.0)
        length = rng.uniform(4.0, 20.0)
        span_x, span_y, span_z = width * 1.25, height * 1.35, length * 1.35
        p[[5, 6, 7]] = [length, width, height]
    elif shape_name == "cylinder":
        outer = rng.uniform(15.0, 40.0)
        length = rng.uniform(50.0, 150.0)
        span_x = span_y = outer * 2.35
        span_z = length * 1.12
        p[[3, 5, 6, 7]] = [outer, length, outer * 2.0, outer * 2.0]
    elif shape_name == "ring":
        outer = rng.uniform(20.0, 60.0)
        inner = rng.uniform(max(8.0, outer * 0.35), outer * 0.75)
        length = rng.uniform(8.0, 30.0)
        span_x = span_y = outer * 2.25
        span_z = length * 1.35
        p[[3, 4, 5, 6, 7]] = [outer, inner, length, outer * 2.0, outer * 2.0]
    elif shape_name == "stepped_shaft":
        large = rng.uniform(18.0, 40.0)
        small = rng.uniform(10.0, large - 4.0)
        length = rng.uniform(80.0, 180.0)
        span_x = span_y = large * 2.35
        span_z = length * 1.10
        p[[3, 5, 14, 15]] = [large, length, large, small]
    elif shape_name == "notched_block":
        width = rng.uniform(35.0, 85.0)
        height = rng.uniform(25.0, 70.0)
        length = rng.uniform(8.0, 25.0)
        notch_width = rng.uniform(6.0, min(20.0, height * 0.45))
        notch_depth = rng.uniform(5.0, min(22.0, width * 0.35))
        span_x, span_y, span_z = width * 1.25, height * 1.25, length * 1.35
        p[[5, 6, 7, 12, 13]] = [length, width, height, notch_width, notch_depth]
    else:
        teeth = int(rng.integers(18, 37))
        module = float(rng.uniform(2.0, 4.0))
        pitch = 0.5 * module * teeth
        tip = pitch + module
        root = pitch - 1.25 * module
        inner = rng.uniform(max(5.0, root * 0.28), root * 0.62)
        length = rng.uniform(8.0, 30.0)
        span_x = span_y = tip * 2.25
        span_z = length * 1.35
        p[[3, 4, 5, 8, 9, 10, 11, 16]] = [tip, inner, length, module, teeth, root, tip, 20.0]

    p[0:3] = [span_x, span_y, span_z]
    return p


def make_engineering_geometry(
    n3d: int,
    n2d: int,
    shape_name: str,
    rng: np.random.Generator,
) -> EngineeringGeometry:
    if shape_name not in ENGINEERING_SHAPES:
        raise ValueError(f"Unknown engineering shape {shape_name}")
    params = _params_for_shape(shape_name, rng)

    x1 = _axis(float(params[0]), n3d)
    y1 = _axis(float(params[1]), n3d)
    z1 = _axis(float(params[2]), n3d)
    x, y, z = np.meshgrid(x1, y1, z1, indexing="ij")
    mask3d = _mask_from_xyz(shape_name, x, y, z, params)
    dx = float(x1[1] - x1[0]) if n3d > 1 else 1.0
    dy = float(y1[1] - y1[0]) if n3d > 1 else 1.0
    dz = float(z1[1] - z1[0]) if n3d > 1 else 1.0
    sdf, depth = _distances(mask3d, (dx, dy, dz))
    local_thickness = np.where(mask3d, np.maximum(2.0 * depth, min(params[0], params[1], params[2]) * 0.05), 0.0)
    coords = np.stack([x, y, z], axis=0).astype(np.float32)

    xs = _axis(float(params[0]), n2d)
    ys = _axis(float(params[1]), n2d)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    section_mask = _mask_from_xy(shape_name, xx, yy, params)
    sx = float(xs[1] - xs[0]) if n2d > 1 else 1.0
    sy = float(ys[1] - ys[0]) if n2d > 1 else 1.0
    _section_sdf, section_depth = _distances(section_mask, (sy, sx))

    return EngineeringGeometry(
        shape_id=ENGINEERING_SHAPES.index(shape_name),
        shape_name=shape_name,
        params=params,
        mask3d=mask3d.astype(np.float32),
        sdf_mm=sdf,
        depth_to_surface_mm=depth,
        local_thickness_mm=local_thickness.astype(np.float32),
        coords_mm=coords,
        section_mask=section_mask.astype(np.float32),
        section_depth_mm=section_depth,
        section_x_mm=xs.astype(np.float32),
        section_y_mm=ys.astype(np.float32),
    )


def balanced_shape_name(case_id: int) -> str:
    return ENGINEERING_SHAPES[case_id % len(ENGINEERING_SHAPES)]
