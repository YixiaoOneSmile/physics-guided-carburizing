# SPDX-License-Identifier: Apache-2.0

"""Evaluate publication checkpoints on every stored depth--time grid point."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from datapipe_engineering import CarburizingProfileQueryDataset
from models_geosurface import GeoSurfaceDeepONet, GeoSurfaceResidualTCN
from train_publication_residual_tcn import file_sha256, forward_output, move_to_device, seed_everything


METRICS = (
    "field_mae_wt",
    "field_rmse_wt",
    "field_rel_l2",
    "surface_history_mae_wt",
    "near_surface_mae_wt",
    "final_profile_mae_wt",
    "final_surface_abs_error_wt",
    "uptake_integral_abs_error_wt_mm",
    "uptake_integral_rel_error",
    "effective_case_depth_abs_error_mm",
)


def decode(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def effective_case_depth(depth_mm: np.ndarray, profile: np.ndarray, threshold: float = 0.40) -> float:
    above = profile >= threshold
    if not np.any(above):
        return 0.0
    if np.all(above):
        return float(depth_mm[-1])
    index = int(np.argmax(~above))
    if index == 0:
        return 0.0
    d0, d1 = float(depth_mm[index - 1]), float(depth_mm[index])
    c0, c1 = float(profile[index - 1]), float(profile[index])
    if abs(c1 - c0) < 1.0e-12:
        return d0
    return d0 + (threshold - c0) * (d1 - d0) / (c1 - c0)


def metric_row(
    prediction: np.ndarray,
    truth: np.ndarray,
    depth_mm: np.ndarray,
    c0: float,
    metadata: dict,
) -> dict:
    difference = prediction - truth
    near_surface = depth_mm <= 0.65
    pred_uptake = float(np.trapezoid(prediction[-1] - c0, depth_mm))
    true_uptake = float(np.trapezoid(truth[-1] - c0, depth_mm))
    return {
        **metadata,
        "field_mae_wt": float(np.mean(np.abs(difference))),
        "field_rmse_wt": float(np.sqrt(np.mean(difference**2))),
        "field_rel_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(truth), 1.0e-12)),
        "surface_history_mae_wt": float(np.mean(np.abs(difference[:, 0]))),
        "near_surface_mae_wt": float(np.mean(np.abs(difference[:, near_surface]))),
        "final_profile_mae_wt": float(np.mean(np.abs(difference[-1]))),
        "final_surface_abs_error_wt": float(abs(difference[-1, 0])),
        "uptake_integral_abs_error_wt_mm": abs(pred_uptake - true_uptake),
        "uptake_integral_rel_error": abs(pred_uptake - true_uptake) / max(abs(true_uptake), 1.0e-8),
        "effective_case_depth_abs_error_mm": abs(
            effective_case_depth(depth_mm, prediction[-1])
            - effective_case_depth(depth_mm, truth[-1])
        ),
    }


def summarize(rows: list[dict], method: str) -> dict:
    result = {"method": method, "num_samples": len(rows)}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        result[f"{metric}_mean"] = float(np.mean(values))
        result[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        result[f"{metric}_median"] = float(np.median(values))
    return result


def grouped(rows: list[dict], method: str, key: str, names: list[str]) -> list[dict]:
    parts: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        parts[int(row[key])].append(row)
    output = []
    for group_id, group_rows in sorted(parts.items()):
        summary = summarize(group_rows, method)
        summary.update({"group_type": key, "group_id": group_id, "group": names[group_id]})
        output.append(summary)
    return output


def pair_summary(records: dict[int, dict[int, dict]], method: str) -> dict:
    rows = []
    for pair_id, variants in records.items():
        if set(variants) != {0, 1}:
            continue
        a, b = variants[0], variants[1]
        true_delta = a["truth"] - b["truth"]
        predicted_delta = a["prediction"] - b["prediction"]
        error = predicted_delta - true_delta
        rows.append(
            {
                "pair_id": pair_id,
                "field_delta_mae_wt": float(np.mean(np.abs(error))),
                "surface_delta_mae_wt": float(np.mean(np.abs(error[:, 0]))),
                "final_delta_mae_wt": float(np.mean(np.abs(error[-1]))),
            }
        )
    output = {"method": method, "num_pairs": len(rows)}
    for metric in ("field_delta_mae_wt", "surface_delta_mae_wt", "final_delta_mae_wt"):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        output[f"{metric}_mean"] = float(np.mean(values))
        output[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return output


def build_model(checkpoint: dict, sample: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.nn.Module, argparse.Namespace]:
    config = checkpoint["config"]
    args = argparse.Namespace(**config)
    geometry_channels = sample["geometry_params"].numel() if args.context_mode == "geometry_case" else 0
    shape_channels = sample["shape_onehot"].numel() if args.context_mode == "geometry_case" else 0
    descriptor_channels = sample["process_descriptors"].numel() if args.use_descriptors else 0
    common_model_args = dict(
        query_channels=3,
        temporal_channels=sample["temporal_features"].shape[0],
        geometry_channels=geometry_channels,
        material_channels=sample["material"].numel(),
        descriptor_channels=descriptor_channels,
        shape_channels=shape_channels,
        residual_scale=float(args.residual_scale),
        output_mode="absolute" if args.method == "absolute" else "residual",
        time_channel_index=0,
        depth_channel_index=1,
        residual_envelope=(
            "none" if args.method == "absolute" else getattr(args, "residual_envelope", "none")
        ),
        residual_depth_decay=float(getattr(args, "residual_depth_decay", 18.0)),
    )
    if getattr(args, "architecture", "tcn") == "deeponet":
        model = GeoSurfaceDeepONet(
            **common_model_args,
            temporal_steps=sample["temporal_features"].shape[1],
            basis=int(getattr(args, "deeponet_basis", 128)),
            hidden=int(getattr(args, "deeponet_hidden", 192)),
        ).to(device)
    else:
        model = GeoSurfaceResidualTCN(
            **common_model_args,
            latent=int(args.latent),
            temporal_mode=args.temporal_mode,
            temporal_steps=sample["temporal_features"].shape[1],
            global_history_mode=getattr(args, "global_history_mode", "pooled"),
            temporal_blend_decay=float(getattr(args, "temporal_blend_decay", 18.0)),
        ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, args


@torch.no_grad()
def evaluate_model(
    model: GeoSurfaceResidualTCN,
    model_args: argparse.Namespace,
    loader: DataLoader,
    device: torch.device,
    nt: int,
    nd: int,
) -> tuple[list[dict], dict[int, dict[int, dict]], float]:
    rows = []
    pair_records: dict[int, dict[int, dict]] = defaultdict(dict)
    elapsed = 0.0
    timed_cases = 0
    for batch_index, batch in enumerate(loader):
        batch_device = move_to_device(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _, prediction = forward_output(model, batch_device, model_args)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if batch_index >= 2:
            elapsed += time.perf_counter() - start
            timed_cases += int(prediction.shape[0])
        prediction_np = prediction.detach().cpu().numpy().reshape(-1, nt, nd)
        truth_np = batch["carbon"].numpy().reshape(-1, nt, nd)
        depth_np = batch["depth_grid_mm"].numpy()
        material_np = batch["material_raw"].numpy()
        for index in range(prediction_np.shape[0]):
            metadata = {
                "shape_id": int(batch["shape_id"][index]),
                "process_family_id": int(batch["process_family_id"][index]),
                "pair_id": int(batch["pair_id"][index]),
                "variant_id": int(batch["variant_id"][index]),
                "dynamic_sensitivity_score": float(batch["dynamic_sensitivity_score"][index]),
            }
            rows.append(metric_row(prediction_np[index], truth_np[index], depth_np[index], float(material_np[index, 0]), metadata))
            pair_records[metadata["pair_id"]][metadata["variant_id"]] = {
                "prediction": prediction_np[index],
                "truth": truth_np[index],
            }
    return rows, pair_records, elapsed / max(timed_cases, 1)


def evaluate_average_prior(dataset: CarburizingProfileQueryDataset) -> tuple[list[dict], dict[int, dict[int, dict]]]:
    rows = []
    pair_records: dict[int, dict[int, dict]] = defaultdict(dict)
    depth = np.asarray(dataset.cache["depth_grid_mm"], dtype=np.float32)
    for index in range(len(dataset)):
        prediction = np.asarray(dataset.cache["avg_carbon_profile"][index], dtype=np.float32)
        truth = np.asarray(dataset.cache["carbon_profile"][index], dtype=np.float32)
        metadata = {
            "shape_id": int(dataset.cache["shape_id"][index]),
            "process_family_id": int(dataset.cache["process_family_id"][index]),
            "pair_id": int(dataset.cache["pair_id"][index]),
            "variant_id": int(dataset.cache["variant_id"][index]),
            "dynamic_sensitivity_score": float(dataset.cache["dynamic_sensitivity_score"][index]),
        }
        rows.append(metric_row(prediction, truth, depth, float(dataset.cache["material"][index, 0]), metadata))
        pair_records[metadata["pair_id"]][metadata["variant_id"]] = {
            "prediction": prediction,
            "truth": truth,
        }
    return rows, pair_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9090)
    args = parser.parse_args()
    seed_everything(args.seed)
    test_path = args.dataset_root / "test.hdf5"
    dataset = CarburizingProfileQueryDataset(str(test_path), random_query=False, full_grid=True, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    with h5py.File(test_path, "r") as handle:
        shape_names = decode(np.asarray(handle["shape_names"]))
        family_names = decode(np.asarray(handle["process_family_names"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, model_args = build_model(checkpoint, dataset[0], device)
    architecture = getattr(model_args, "architecture", "tcn")
    method_name = f"{model_args.method}_{architecture}_seed{model_args.seed}"
    model_rows, model_pairs, forward_seconds = evaluate_model(
        model, model_args, loader, device, dataset.nt, dataset.nd
    )
    prior_rows, prior_pairs = evaluate_average_prior(dataset)
    summaries = [summarize(prior_rows, "average_cp_prior"), summarize(model_rows, method_name)]
    summaries[1].update(
        {
            "cached_prior_model_forward_seconds_per_case": forward_seconds,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "test_sha256": file_sha256(test_path),
        }
    )
    pair_summaries = [
        pair_summary(prior_pairs, "average_cp_prior"),
        pair_summary(model_pairs, method_name),
    ]
    grouped_rows = (
        grouped(prior_rows, "average_cp_prior", "process_family_id", family_names)
        + grouped(model_rows, method_name, "process_family_id", family_names)
        + grouped(prior_rows, "average_cp_prior", "shape_id", shape_names)
        + grouped(model_rows, method_name, "shape_id", shape_names)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "per_case_metrics.csv", model_rows)
    write_csv(args.output_dir / "grouped_metrics.csv", grouped_rows)
    write_csv(args.output_dir / "pair_metrics.csv", pair_summaries)
    prior_scope = (
        "initial carbon concentration available"
        if model_args.method == "initial_prior"
        else "average-Cp profile already available"
    )
    report = {
        "summary": summaries,
        "pair_summary": pair_summaries,
        "timing_scope": f"neural-network forward pass with {prior_scope}",
    }
    with open(args.output_dir / "evaluation.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
