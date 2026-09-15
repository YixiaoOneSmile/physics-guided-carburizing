#!/usr/bin/env python3
"""Evaluate one matrix cell on a complete split and all 21 time layers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def update_corr(acc: dict[str, float], pred: np.ndarray, target: np.ndarray) -> None:
    p = pred.astype(np.float64).reshape(-1); t = target.astype(np.float64).reshape(-1)
    acc["n"] += p.size; acc["sum_p"] += float(p.sum()); acc["sum_t"] += float(t.sum())
    acc["sum_pp"] += float(np.dot(p, p)); acc["sum_tt"] += float(np.dot(t, t)); acc["sum_pt"] += float(np.dot(p, t))


def corr_value(acc: dict[str, float]) -> float:
    n = max(acc["n"], 1.0)
    cov = acc["sum_pt"] - acc["sum_p"] * acc["sum_t"] / n
    vp = acc["sum_pp"] - acc["sum_p"] ** 2 / n
    vt = acc["sum_tt"] - acc["sum_t"] ** 2 / n
    return float(cov / max(np.sqrt(max(vp, 0.0) * max(vt, 0.0)), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backbone", choices=["mlp", "resnet", "fno"], required=True)
    parser.add_argument("--mode", choices=["direct", "prior_direct", "residual"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(cfg["experiment_root"]); source_root = Path(cfg["source_root"]); sidecar_root = Path(cfg["sidecar_root"])
    sys.path.insert(0, str(root / "code_snapshot")); sys.path.insert(0, str(root / "code_snapshot" / "upstream"))
    from matrix_datapipe3d import Matrix3DQueryDataset
    from matrix_models3d import MatrixPredictor3D
    from metrics3d import compute_sequence_metrics, write_metrics_outputs

    run_name = f"{args.backbone}_{args.mode}_seed_{args.seed}"
    checkpoint_path = root / "checkpoints" / run_name / "best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MatrixPredictor3D(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    stats = json.loads((sidecar_root / "residual_stats.json").read_text(encoding="utf-8"))
    dataset = Matrix3DQueryDataset(
        source_root / f"{args.split}.hdf5", sidecar_root / f"{args.split}_avgcp.hdf5",
        sidecar_root / "residual_stats.json", target_mode=args.mode,
        random_time=False, samples_per_case=21, preload=True,
    )
    output_root = root / "results" / run_name / args.split
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_rows = []; prior_rows = []; forward_times = []
    corr = {key: 0.0 for key in ("n", "sum_p", "sum_t", "sum_pp", "sum_tt", "sum_pt")}
    with h5py.File(source_root / f"{args.split}.hdf5", "r") as source, h5py.File(sidecar_root / f"{args.split}_avgcp.hdf5", "r") as sidecar:
        n_cases = int(source["carbon"].shape[0]); nt = int(source["carbon"].shape[1])
        shape_names = [item.decode("utf-8") for item in source["shape_names"][:]]
        dx_m = float(source.attrs["domain_mm"]) * 1e-3 / max(source["carbon"].shape[-1] - 1, 1)
        for case_id in range(n_cases):
            truth = np.asarray(source["carbon"][case_id], dtype=np.float32)
            prior = np.asarray(sidecar["avg_carbon"][case_id], dtype=np.float32)
            mask = np.asarray(source["mask"][case_id, 0], dtype=np.float32)
            surface = np.asarray(source["surface"][case_id], dtype=np.float32)
            shape_id = int(source["shape_id"][case_id]); parts = []
            for start in range(0, nt, int(cfg["time_batch_size"])):
                samples = [dataset.get_case_query(case_id, q) for q in range(start, min(start + int(cfg["time_batch_size"]), nt))]
                grid = torch.stack([s["grid"] for s in samples]).to(device)
                process = torch.stack([s["process"] for s in samples]).to(device)
                query = torch.stack([s["query_index"] for s in samples]).to(device)
                sync(device); started = time.perf_counter()
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bool(cfg["amp_bfloat16"] and device.type == "cuda")):
                    pred_norm = model(grid, process, query)
                sync(device); forward_times.append(time.perf_counter() - started)
                pred_norm = pred_norm.float().cpu().numpy()[:, 0]
                if args.mode == "residual":
                    physical = prior[start:start + len(samples)] + pred_norm * float(stats["residual_std"]) + float(stats["residual_mean"])
                else:
                    physical = pred_norm * float(stats["carbon_std"]) + float(stats["carbon_mean"])
                parts.append(physical)
            raw = np.concatenate(parts, axis=0); pred = np.clip(raw, 0.02, 1.45)
            active = mask > 0.5; update_corr(corr, pred[:, active], truth[:, active])
            meta = {"case_id": case_id, "shape_id": shape_id, "shape_name": shape_names[shape_id]}
            prediction_rows.append({**meta, **compute_sequence_metrics(pred, truth, mask, surface, dx_m)})
            prior_rows.append({**meta, **compute_sequence_metrics(prior, truth, mask, surface, dx_m)})
            print(f"{args.split} {case_id + 1}/{n_cases}", flush=True)
    pred_summary = write_metrics_outputs(output_root / "prediction", prediction_rows, shape_names, extra_metrics={
        "model_forward_time_per_case_sequence_s": float(np.sum(forward_times) / max(len(prediction_rows), 1)),
        "prediction_pearson_r": corr_value(corr), "parameter_count": int(checkpoint["parameter_count"]),
    })
    prior_summary = write_metrics_outputs(output_root / "strict_avgcp", prior_rows, shape_names)
    comparison = {
        "run_name": run_name, "split": args.split, "checkpoint_epoch": int(checkpoint["epoch"]),
        "backbone": args.backbone, "target_mode": args.mode, "seed": args.seed,
        "parameter_count": int(checkpoint["parameter_count"]), "prediction": pred_summary, "strict_avgcp": prior_summary,
    }
    (output_root / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
