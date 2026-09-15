#!/usr/bin/env python3
"""Train one matched backbone/target-mode cell of the 3-D matrix."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def loss_terms(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, surface: torch.Tensor):
    active_surface = (surface.sum(dim=1, keepdim=True) > 0.5).to(mask.dtype) * mask
    data = masked_mean((pred - target).square(), mask)
    surface_l1 = masked_mean((pred - target).abs(), active_surface)
    pred_mean = (pred * mask).sum(dim=(2, 3, 4)) / mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
    target_mean = (target * mask).sum(dim=(2, 3, 4)) / mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
    mass_l1 = (pred_mean - target_mean).abs().mean()
    return data, surface_l1, mass_l1


@torch.no_grad()
def validate(model, loader, device, amp_enabled: bool) -> dict[str, float]:
    model.eval()
    totals = {"mse": 0.0, "surface_l1": 0.0, "mass_l1": 0.0, "batches": 0}
    for batch in loader:
        batch = move(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            pred = model(batch["grid"], batch["process"], batch["query_index"])
            data, surface_l1, mass_l1 = loss_terms(pred, batch["target"], batch["mask"], batch["surface"])
        totals["mse"] += float(data)
        totals["surface_l1"] += float(surface_l1)
        totals["mass_l1"] += float(mass_l1)
        totals["batches"] += 1
    batches = max(int(totals.pop("batches")), 1)
    return {key: value / batches for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backbone", choices=["mlp", "resnet", "fno"], required=True)
    parser.add_argument("--mode", choices=["direct", "prior_direct", "residual"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(cfg["experiment_root"])
    source_root = Path(cfg["source_root"])
    sidecar_root = Path(cfg["sidecar_root"])
    sys.path.insert(0, str(root / "code_snapshot"))
    from matrix_datapipe3d import GRID_CHANNELS, Matrix3DQueryDataset
    from matrix_models3d import MatrixPredictor3D

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg["amp_bfloat16"] and device.type == "cuda")
    stats_path = sidecar_root / "residual_stats.json"
    epochs = 2 if args.smoke else int(cfg["epochs"])
    steps_per_epoch = 4 if args.smoke else int(cfg["steps_per_epoch"])
    samples_per_case = max(4, int(np.ceil(steps_per_epoch * int(cfg["batch_size"]) / 66)))
    train_dataset = Matrix3DQueryDataset(
        source_root / "train.hdf5", sidecar_root / "train_avgcp.hdf5", stats_path,
        target_mode=args.mode, random_time=True, samples_per_case=samples_per_case, preload=True,
    )
    val_dataset = Matrix3DQueryDataset(
        source_root / "val.hdf5", sidecar_root / "val_avgcp.hdf5", stats_path,
        target_mode=args.mode, random_time=False, samples_per_case=21, preload=True,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=int(cfg["batch_size"]), shuffle=True,
        num_workers=int(cfg["num_workers"]), pin_memory=device.type == "cuda", generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(cfg["batch_size"]), shuffle=False,
        num_workers=int(cfg["num_workers"]), pin_memory=device.type == "cuda",
    )
    model_kwargs = {
        "backbone": args.backbone,
        "grid_channels": len(GRID_CHANNELS),
        "process_hidden_channels": int(cfg["process_hidden_channels"]),
        "process_latent_channels": int(cfg["process_latent_channels"]),
        "depth_gate_lambda": float(cfg["depth_gate_lambda"]),
        "mlp_hidden_channels": int(cfg["mlp_hidden_channels"]),
        "mlp_layers": int(cfg["mlp_layers"]),
        "resnet_hidden_channels": int(cfg["resnet_hidden_channels"]),
        "resnet_blocks": int(cfg["resnet_blocks"]),
        "fno_latent_channels": int(cfg["fno_latent_channels"]),
        "fno_layers": int(cfg["fno_layers"]),
        "fno_modes": list(cfg["fno_modes"]),
        "decoder_layer_size": int(cfg["decoder_layer_size"]),
        "padding": int(cfg["padding"]),
    }
    model = MatrixPredictor3D(**model_kwargs).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    run_name = f"{args.backbone}_{args.mode}_seed_{args.seed}" + ("_smoke" if args.smoke else "")
    checkpoint_dir = root / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = root / "logs" / f"training_{run_name}.jsonl"
    if log_path.exists():
        log_path.unlink()
    best_val = float("inf")
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        sums = {"total": 0.0, "mse": 0.0, "surface_l1": 0.0, "mass_l1": 0.0}
        batches = 0
        for batch in train_loader:
            if batches >= steps_per_epoch:
                break
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                pred = model(batch["grid"], batch["process"], batch["query_index"])
                data, surface_l1, mass_l1 = loss_terms(pred, batch["target"], batch["mask"], batch["surface"])
                loss = data + float(cfg["surface_loss_weight"]) * surface_l1 + float(cfg["mass_loss_weight"]) * mass_l1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sums["total"] += float(loss.detach())
            sums["mse"] += float(data.detach())
            sums["surface_l1"] += float(surface_l1.detach())
            sums["mass_l1"] += float(mass_l1.detach())
            batches += 1
        scheduler.step()
        record = {
            "epoch": epoch, "elapsed_s": time.perf_counter() - started,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value / max(batches, 1) for key, value in sums.items()},
        }
        if epoch == 1 or epoch % int(cfg["validation_interval"]) == 0 or epoch == epochs:
            validation = validate(model, val_loader, device, amp_enabled)
            record.update({f"val_{key}": value for key, value in validation.items()})
            selection = validation["mse"] + float(cfg["surface_loss_weight"]) * validation["surface_l1"]
            if selection < best_val:
                best_val = selection
                torch.save({
                    "epoch": epoch, "model": model.state_dict(), "config": cfg,
                    "model_kwargs": model_kwargs, "backbone": args.backbone, "target_mode": args.mode,
                    "seed": args.seed, "best_val": best_val, "parameter_count": parameter_count,
                    "stats_path": str(stats_path),
                }, checkpoint_dir / "best.pt")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    print(json.dumps({
        "status": "complete", "run_name": run_name, "device": str(device), "pid": os.getpid(),
        "epochs": epochs, "best_val": best_val, "elapsed_s": time.perf_counter() - started,
        "parameter_count": parameter_count, "checkpoint": str(checkpoint_dir / "best.pt"),
    }), flush=True)


if __name__ == "__main__":
    main()
