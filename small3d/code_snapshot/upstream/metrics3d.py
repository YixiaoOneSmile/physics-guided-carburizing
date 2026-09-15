# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def compute_sequence_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    surface: np.ndarray,
    dx_m: float,
) -> dict[str, float]:
    """Compute paper-facing 3-D sequence metrics in physical wt% units."""

    active = mask > 0.5
    surface_active = (surface.sum(axis=0) > 0.5) & active
    diff = pred - truth
    truth_active = truth[:, active]
    diff_active = diff[:, active]
    rel_l2 = np.linalg.norm(diff_active.reshape(1, -1)) / max(
        np.linalg.norm(truth_active.reshape(1, -1)), 1e-12
    )
    active_mae = float(np.mean(np.abs(diff_active)))
    final_active_mae = float(np.mean(np.abs(diff[-1, active])))
    if np.any(surface_active):
        surface_mae = float(np.mean(np.abs(diff[:, surface_active])))
    else:
        surface_mae = active_mae

    volume = dx_m**3
    pred_mass = np.sum(pred * mask[None], axis=(1, 2, 3)) * volume
    truth_mass = np.sum(truth * mask[None], axis=(1, 2, 3)) * volume
    mass_rel = float(
        np.mean(np.abs(pred_mass - truth_mass) / np.maximum(np.abs(truth_mass), 1e-12))
    )
    return {
        "active_field_rel_l2": float(rel_l2),
        "active_mae_wt": active_mae,
        "surface_c_mae_wt": surface_mae,
        "final_active_mae_wt": final_active_mae,
        "mass_uptake_rel_error": mass_rel,
    }


def aggregate_metrics(rows: list[dict], prefix_filter: str | None = None) -> dict[str, float]:
    keys = [
        "active_field_rel_l2",
        "active_mae_wt",
        "surface_c_mae_wt",
        "final_active_mae_wt",
        "mass_uptake_rel_error",
    ]
    if prefix_filter is not None:
        rows = [row for row in rows if row["shape_name"] == prefix_filter]
    if not rows:
        return {key: float("nan") for key in keys}
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def write_metrics_outputs(
    outdir: Path,
    rows: list[dict],
    shape_names: list[str],
    extra_metrics: dict | None = None,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_metrics(rows)
    if extra_metrics:
        summary.update(extra_metrics)
    summary["num_samples"] = len(rows)

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if rows:
        with open(outdir / "case_table.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    shape_rows = []
    for name in shape_names:
        metrics = aggregate_metrics(rows, prefix_filter=name)
        metrics["shape_name"] = name
        metrics["num_samples"] = len([row for row in rows if row["shape_name"] == name])
        shape_rows.append(metrics)
    with open(outdir / "shape_breakdown.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["shape_name", "num_samples"] + [
            "active_field_rel_l2",
            "active_mae_wt",
            "surface_c_mae_wt",
            "final_active_mae_wt",
            "mass_uptake_rel_error",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(shape_rows)
    return summary
