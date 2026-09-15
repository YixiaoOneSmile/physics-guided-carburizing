# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


SHAPE_NAMES = (
    "slab",
    "cylinder",
    "ring",
    "stepped_shaft",
    "notched_block",
    "gear_ring",
)


@dataclass(frozen=True)
class VoxelGeometry:
    mask: np.ndarray
    sdf: np.ndarray
    surface: np.ndarray
    coords: np.ndarray
    shape_id: int
    shape_name: str


def coordinate_grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    coords = np.stack([x, y, z], axis=0).astype(np.float32)
    return x, y, z, coords


def signed_distance_from_mask(mask: np.ndarray) -> np.ndarray:
    """Approximate signed distance on a unit cube grid; positive inside material."""

    spacing = 2.0 / max(mask.shape[0] - 1, 1)
    inside = ndimage.distance_transform_edt(mask, sampling=spacing)
    outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
    return (inside - outside).astype(np.float32)


def surface_faces(mask: np.ndarray) -> np.ndarray:
    """Return exposed material faces in order: -x,+x,-y,+y,-z,+z."""

    surface = np.zeros((6, *mask.shape), dtype=np.float32)
    directions = (
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 1),
        (2, -1),
        (2, 1),
    )
    for face_id, (axis, sign) in enumerate(directions):
        neighbor = np.zeros_like(mask, dtype=bool)
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        if sign < 0:
            src[axis] = slice(0, -1)
            dst[axis] = slice(1, None)
        else:
            src[axis] = slice(1, None)
            dst[axis] = slice(0, -1)
        neighbor[tuple(dst)] = mask[tuple(src)]
        surface[face_id] = (mask & ~neighbor).astype(np.float32)
    return surface


def make_geometry(
    n: int,
    shape_name: str,
    rng: np.random.Generator | None = None,
) -> VoxelGeometry:
    """Build one analytic voxel shape for 3-D carburizing studies."""

    if rng is None:
        rng = np.random.default_rng()
    if shape_name not in SHAPE_NAMES:
        raise ValueError(f"Unknown shape {shape_name}; choose from {SHAPE_NAMES}")

    x, y, z, coords = coordinate_grid(n)
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    if shape_name == "slab":
        hx = rng.uniform(0.62, 0.95)
        hy = rng.uniform(0.62, 0.95)
        hz = rng.uniform(0.55, 0.95)
        mask = (np.abs(x) <= hx) & (np.abs(y) <= hy) & (np.abs(z) <= hz)
    elif shape_name == "cylinder":
        radius = rng.uniform(0.48, 0.72)
        half_len = rng.uniform(0.75, 0.98)
        mask = (r <= radius) & (np.abs(z) <= half_len)
    elif shape_name == "ring":
        outer = rng.uniform(0.60, 0.82)
        inner = rng.uniform(0.20, min(0.42, outer - 0.12))
        mask = (r <= outer) & (r >= inner) & (np.abs(z) <= rng.uniform(0.72, 0.98))
    elif shape_name == "stepped_shaft":
        r1 = rng.uniform(0.50, 0.70)
        r2 = rng.uniform(0.30, min(0.48, r1 - 0.06))
        radius = np.where(z < 0.0, r1, r2)
        mask = (r <= radius) & (np.abs(z) <= rng.uniform(0.78, 0.98))
    elif shape_name == "notched_block":
        mask = (np.abs(x) <= 0.82) & (np.abs(y) <= 0.72) & (np.abs(z) <= 0.82)
        notch_width = rng.uniform(0.18, 0.34)
        notch_depth = rng.uniform(0.25, 0.48)
        notch = (np.abs(y) < notch_width) & (x > 0.82 - notch_depth) & (np.abs(z) < 0.55)
        mask = mask & ~notch
    else:
        teeth = int(rng.choice([8, 10, 12]))
        base_outer = rng.uniform(0.58, 0.70)
        tooth_amp = rng.uniform(0.08, 0.15)
        outer = base_outer + tooth_amp * (0.5 + 0.5 * np.cos(teeth * theta))
        inner = rng.uniform(0.18, 0.32)
        mask = (r <= outer) & (r >= inner) & (np.abs(z) <= rng.uniform(0.65, 0.92))

    mask = mask.astype(bool)
    sdf = signed_distance_from_mask(mask)
    surface = surface_faces(mask)
    return VoxelGeometry(
        mask=mask.astype(np.float32),
        sdf=sdf,
        surface=surface,
        coords=coords,
        shape_id=SHAPE_NAMES.index(shape_name),
        shape_name=shape_name,
    )


def random_geometry(n: int, rng: np.random.Generator) -> VoxelGeometry:
    return make_geometry(n, str(rng.choice(SHAPE_NAMES)), rng)
