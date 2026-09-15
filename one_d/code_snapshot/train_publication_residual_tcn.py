# SPDX-License-Identifier: Apache-2.0

"""Train the frozen publication-v3 depth--time surrogate protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from datapipe_engineering import CarburizingPairedProfileDataset, CarburizingProfileQueryDataset
from models_geosurface import GeoSurfaceDeepONet, GeoSurfaceResidualTCN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def strip_prefix(batch: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key.removeprefix(prefix): value for key, value in batch.items() if key.startswith(prefix)}


def model_inputs(
    batch: dict[str, torch.Tensor], args: argparse.Namespace, query: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model_query = batch["residual_query"] if query is None else query
    if args.method == "absolute":
        model_query = model_query.clone()
        model_query[..., 2] = 0.0
    elif args.method == "initial_prior":
        model_query = model_query.clone()
        c0 = batch["material_raw"][:, 0].to(model_query.dtype)
        model_query[..., 2] = c0[:, None]
    temporal = batch["temporal_features"]
    if args.history_mode == "descriptors_only":
        temporal = torch.zeros_like(temporal)
    if args.context_mode == "geometry_case":
        geometry = batch["geometry_params"]
        shape = batch["shape_onehot"]
    else:
        geometry = batch["geometry_params"][:, :0]
        shape = batch["shape_onehot"][:, :0]
    descriptors = batch["process_descriptors"]
    if not args.use_descriptors:
        descriptors = descriptors[:, :0]
    return model_query, temporal, geometry, batch["material"], descriptors, shape


def forward_output(
    model: GeoSurfaceResidualTCN,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    query: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(*model_inputs(batch, args, query=query))
    if args.method == "absolute":
        return output, output
    if args.method == "initial_prior":
        c0 = batch["material_raw"][:, 0].to(output.dtype)
        prior = c0[:, None, None].expand_as(output)
        return output, prior + output
    prior = (batch["residual_query"] if query is None else query)[..., 2:3]
    return output, prior + output


def weighted_l1(pred: torch.Tensor, truth: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(pred.dtype)
    return (torch.abs(pred - truth) * weight).sum() / weight.sum().clamp_min(1.0)


def interpolate_prior(
    avg_profile: torch.Tensor, time_norm: torch.Tensor, depth_norm: torch.Tensor
) -> torch.Tensor:
    grid = torch.stack([2.0 * depth_norm - 1.0, 2.0 * time_norm - 1.0], dim=-1)
    grid = grid.unsqueeze(2)
    values = F.grid_sample(
        avg_profile.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return values[:, 0, :, 0:1]


def diffusivity_agren_torch(temperature_c: torch.Tensor, carbon_wt: torch.Tensor) -> torch.Tensor:
    mass_fraction = torch.clamp(carbon_wt / 100.0, 0.0, 0.20)
    n_c = mass_fraction / 12.011
    n_fe = (1.0 - mass_fraction) / 55.845
    x_c = n_c / (n_c + n_fe).clamp_min(1.0e-30)
    y_c = x_c / (1.0 - x_c).clamp_min(1.0e-30)
    temperature_k = temperature_c + 273.15
    prefactor = 4.53e-7 * (1.0 + y_c * (1.0 - y_c) * 8339.9 / temperature_k)
    exponent = -(1.0 / temperature_k - 2.221e-4) * (17767.0 - 26436.0 * y_c)
    return prefactor * torch.exp(exponent)


def physics_consistency_loss(
    model: GeoSurfaceResidualTCN,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-difference PDE and Robin-boundary residuals in physical units."""

    count = min(args.physics_queries, batch["residual_query"].shape[1])
    query = batch["residual_query"][:, :count].clone()
    nt = batch["temporal_features"].shape[-1]
    nd = batch["avg_profile_full"].shape[-1]
    dt_norm = 1.0 / max(nt - 1, 1)
    dz_norm = 1.0 / max(nd - 1, 1)
    total_time_s = batch["time"][:, -1:].clamp_min(1.0)
    depth_max_m = batch["depth_grid_mm"][:, -1:].clamp_min(1.0e-6) * 1.0e-3
    dt_s = total_time_s / max(nt - 1, 1)
    dz_m = depth_max_m / max(nd - 1, 1)

    def shifted(delta_t: float, delta_z: float) -> torch.Tensor:
        shifted_query = query.clone()
        shifted_query[..., 0] = torch.clamp(shifted_query[..., 0] + delta_t, 0.0, 1.0)
        shifted_query[..., 1] = torch.clamp(shifted_query[..., 1] + delta_z, 0.0, 1.0)
        shifted_query[..., 2:3] = interpolate_prior(
            batch["avg_profile_full"], shifted_query[..., 0], shifted_query[..., 1]
        )
        return shifted_query

    q_tp = shifted(dt_norm, 0.0)
    q_tm = shifted(-dt_norm, 0.0)
    q_dp = shifted(0.0, dz_norm)
    q_dm = shifted(0.0, -dz_norm)
    _, center = forward_output(model, batch, args, query=query)
    _, c_tp = forward_output(model, batch, args, query=q_tp)
    _, c_tm = forward_output(model, batch, args, query=q_tm)
    _, c_dp = forward_output(model, batch, args, query=q_dp)
    _, c_dm = forward_output(model, batch, args, query=q_dm)

    process_at_query = GeoSurfaceResidualTCN.gather_query_time(batch["process_raw"], query[..., 0])
    temperature = process_at_query[..., 1:2]
    d_center = diffusivity_agren_torch(temperature, center)
    d_plus = diffusivity_agren_torch(temperature, c_dp)
    d_minus = diffusivity_agren_torch(temperature, c_dm)
    time_derivative = (c_tp - c_tm) / (2.0 * dt_s[:, None, :])
    flux_plus = 0.5 * (d_center + d_plus) * (c_dp - center) / dz_m[:, None, :]
    flux_minus = 0.5 * (d_center + d_minus) * (center - c_dm) / dz_m[:, None, :]
    divergence = (flux_plus - flux_minus) / dz_m[:, None, :]
    interior = (
        (query[..., 0:1] >= dt_norm)
        & (query[..., 0:1] <= 1.0 - dt_norm)
        & (query[..., 1:2] >= dz_norm)
        & (query[..., 1:2] <= 1.0 - dz_norm)
    ).to(center.dtype)
    pde = weighted_l1(time_derivative, divergence, interior) / args.pde_scale_wt_s

    q_surface = shifted(0.0, -2.0)
    q_inside = q_surface.clone()
    q_inside[..., 1] = dz_norm
    q_inside[..., 2:3] = interpolate_prior(
        batch["avg_profile_full"], q_inside[..., 0], q_inside[..., 1]
    )
    _, c_surface = forward_output(model, batch, args, query=q_surface)
    _, c_inside = forward_output(model, batch, args, query=q_inside)
    process_surface = GeoSurfaceResidualTCN.gather_query_time(batch["process_raw"], q_surface[..., 0])
    cp = process_surface[..., 0:1]
    temp_surface = process_surface[..., 1:2]
    h_m = process_surface[..., 2:3]
    d_surface = diffusivity_agren_torch(temp_surface, c_surface)
    d_inside = diffusivity_agren_torch(temp_surface, c_inside)
    inward_flux = -0.5 * (d_surface + d_inside) * (c_inside - c_surface) / dz_m[:, None, :]
    robin_flux = h_m * (cp - c_surface)
    robin = torch.mean(torch.abs(inward_flux - robin_flux)) / args.robin_scale_wt_m_s
    return pde, robin


def structured_predictions(
    model: GeoSurfaceResidualTCN,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict the complete surface history and final depth profile."""

    time_norm = batch["time"] / batch["time"][:, -1:].clamp_min(1.0)
    surface_query = torch.stack(
        [
            time_norm,
            torch.zeros_like(time_norm),
            batch["avg_profile_full"][:, :, 0],
        ],
        dim=-1,
    )
    depth_norm = batch["depth_grid_mm"] / batch["depth_grid_mm"][:, -1:].clamp_min(1.0e-6)
    final_query = torch.stack(
        [
            torch.ones_like(depth_norm),
            depth_norm,
            batch["avg_profile_full"][:, -1, :],
        ],
        dim=-1,
    )
    _, predicted_surface = forward_output(model, batch, args, query=surface_query)
    _, predicted_final = forward_output(model, batch, args, query=final_query)
    return predicted_surface, predicted_final


def structured_profile_losses(
    model: GeoSurfaceResidualTCN,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Losses on complete surface histories and final depth profiles."""

    predicted_surface, predicted_final = structured_predictions(model, batch, args)
    true_surface = batch["carbon_profile_full"][:, :, 0:1]
    true_final = batch["carbon_profile_full"][:, -1, :].unsqueeze(-1)
    surface_history = torch.mean(torch.abs(predicted_surface - true_surface))
    final_profile = torch.mean(torch.abs(predicted_final - true_final))

    depth = batch["depth_grid_mm"]
    c0 = batch["material_raw"][:, 0:1]
    predicted_uptake = torch.trapezoid(predicted_final.squeeze(-1) - c0, depth, dim=1)
    true_uptake = torch.trapezoid(true_final.squeeze(-1) - c0, depth, dim=1)
    depth_max = depth[:, -1].clamp_min(1.0e-6)
    uptake = torch.mean(torch.abs(predicted_uptake - true_uptake) / depth_max)

    predicted_indicator = torch.sigmoid((predicted_final.squeeze(-1) - 0.40) / 0.02)
    true_indicator = torch.sigmoid((true_final.squeeze(-1) - 0.40) / 0.02)
    predicted_ecd = torch.trapezoid(predicted_indicator, depth, dim=1)
    true_ecd = torch.trapezoid(true_indicator, depth, dim=1)
    ecd = torch.mean(torch.abs(predicted_ecd - true_ecd) / depth_max)
    return surface_history, final_profile, uptake, ecd


def supervised_terms(
    model: GeoSurfaceResidualTCN,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    include_physics: bool,
) -> dict[str, torch.Tensor]:
    raw_output, prediction = forward_output(model, batch, args)
    if args.method == "absolute":
        target_output = batch["carbon"]
    elif args.method == "initial_prior":
        c0 = batch["material_raw"][:, 0].to(batch["carbon"].dtype)
        target_output = batch["carbon"] - c0[:, None, None]
    else:
        target_output = batch["residual"]
    data_mse = F.mse_loss(raw_output, target_output)
    surface_l1, final_l1, uptake_l1, ecd_l1 = structured_profile_losses(model, batch, args)
    initial_l1 = weighted_l1(prediction, batch["carbon"], batch["initial_weight"])
    pde = prediction.new_tensor(0.0)
    robin = prediction.new_tensor(0.0)
    if include_physics and args.method == "residual_physics":
        pde, robin = physics_consistency_loss(model, batch, args)
    loss = (
        args.data_weight * data_mse
        + args.surface_weight * surface_l1
        + args.final_weight * final_l1
        + args.uptake_weight * uptake_l1
        + args.ecd_weight * ecd_l1
        + args.initial_weight * initial_l1
        + args.pde_weight * pde
        + args.robin_weight * robin
    )
    return {
        "loss": loss,
        "data_mse": data_mse.detach(),
        "surface_l1": surface_l1.detach(),
        "final_l1": final_l1.detach(),
        "uptake_l1": uptake_l1.detach(),
        "ecd_l1": ecd_l1.detach(),
        "initial_l1": initial_l1.detach(),
        "pde_residual": pde.detach(),
        "robin_residual": robin.detach(),
    }


def pair_loss(
    model: GeoSurfaceResidualTCN, pair_batch: dict[str, torch.Tensor], args: argparse.Namespace
) -> torch.Tensor:
    sample_a = strip_prefix(pair_batch, "a_")
    sample_b = strip_prefix(pair_batch, "b_")
    surface_a, final_a = structured_predictions(model, sample_a, args)
    surface_b, final_b = structured_predictions(model, sample_b, args)
    true_surface_delta = (
        sample_a["carbon_profile_full"][:, :, 0:1]
        - sample_b["carbon_profile_full"][:, :, 0:1]
    )
    true_final_delta = (
        sample_a["carbon_profile_full"][:, -1, :].unsqueeze(-1)
        - sample_b["carbon_profile_full"][:, -1, :].unsqueeze(-1)
    )
    surface_delta = torch.mean(torch.abs((surface_a - surface_b) - true_surface_delta))
    final_delta = torch.mean(torch.abs((final_a - final_b) - true_final_delta))
    return 0.70 * surface_delta + 0.30 * final_delta


@torch.no_grad()
def validate(
    model: GeoSurfaceResidualTCN,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        terms = supervised_terms(model, batch, args, include_physics=False)
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + float(value.cpu())
        count += 1
    result = {key: value / max(count, 1) for key, value in totals.items()}
    result["selection_score"] = (
        result["surface_l1"]
        + 0.5 * result["final_l1"]
        + 0.25 * result["uptake_l1"]
        + 0.10 * result["ecd_l1"]
        + 0.25 * np.sqrt(result["data_mse"])
    )
    return result


def append_csv(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=["absolute", "initial_prior", "residual", "residual_pair", "residual_physics"],
        required=True,
    )
    parser.add_argument("--architecture", choices=["tcn", "deeponet"], default="tcn")
    parser.add_argument(
        "--temporal-mode", choices=["query", "global", "depth_blend"], default="query"
    )
    parser.add_argument(
        "--global-history-mode", choices=["pooled", "position_mlp"], default="pooled"
    )
    parser.add_argument("--history-mode", choices=["full", "descriptors_only"], default="full")
    parser.add_argument("--context-mode", choices=["profile", "geometry_case"], default="profile")
    parser.add_argument("--use-descriptors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--queries-per-case", type=int, default=2048)
    parser.add_argument("--val-queries-per-case", type=int, default=4096)
    parser.add_argument("--latent", type=int, default=192)
    parser.add_argument("--deeponet-basis", type=int, default=128)
    parser.add_argument("--deeponet-hidden", type=int, default=192)
    parser.add_argument("--residual-scale", type=float, default=0.22)
    parser.add_argument("--residual-envelope", choices=["none", "diffusion"], default="diffusion")
    parser.add_argument("--residual-depth-decay", type=float, default=18.0)
    parser.add_argument("--temporal-blend-decay", type=float, default=18.0)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--lr-patience", type=int, default=6)
    parser.add_argument("--early-stopping-patience", type=int, default=18)
    parser.add_argument("--data-weight", type=float, default=1.0)
    parser.add_argument("--surface-weight", type=float, default=0.35)
    parser.add_argument("--final-weight", type=float, default=0.15)
    parser.add_argument("--uptake-weight", type=float, default=0.10)
    parser.add_argument("--ecd-weight", type=float, default=0.05)
    parser.add_argument("--initial-weight", type=float, default=0.05)
    parser.add_argument("--pair-weight", type=float, default=0.05)
    parser.add_argument("--pde-weight", type=float, default=2.0e-4)
    parser.add_argument("--robin-weight", type=float, default=2.0e-4)
    parser.add_argument("--physics-queries", type=int, default=256)
    parser.add_argument("--physics-warmup-epochs", type=int, default=15)
    parser.add_argument("--pde-scale-wt-s", type=float, default=1.0e-5)
    parser.add_argument("--robin-scale-wt-m-s", type=float, default=1.0e-8)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    seed_everything(args.seed)
    train_path = args.dataset_root / "train.hdf5"
    val_path = args.dataset_root / "val.hdf5"
    train_dataset = CarburizingProfileQueryDataset(
        str(train_path), args.queries_per_case, random_query=True, seed=args.seed
    )
    val_dataset = CarburizingProfileQueryDataset(
        str(val_path), args.val_queries_per_case, random_query=False, seed=args.seed + 1000
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pair_loader = None
    if args.method == "residual_pair":
        pair_dataset = CarburizingPairedProfileDataset(
            str(train_path), queries_per_case=args.queries_per_case, seed=args.seed + 2000
        )
        pair_loader = DataLoader(
            pair_dataset,
            batch_size=max(1, args.batch_size // 2),
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(args.seed + 1),
        )

    sample = train_dataset[0]
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
        residual_scale=args.residual_scale,
        output_mode="absolute" if args.method == "absolute" else "residual",
        time_channel_index=0,
        depth_channel_index=1,
        residual_envelope="none" if args.method == "absolute" else args.residual_envelope,
        residual_depth_decay=args.residual_depth_decay,
    )
    if args.architecture == "deeponet":
        model = GeoSurfaceDeepONet(
            **common_model_args,
            temporal_steps=sample["temporal_features"].shape[1],
            basis=args.deeponet_basis,
            hidden=args.deeponet_hidden,
        )
    else:
        model = GeoSurfaceResidualTCN(
            **common_model_args,
            latent=args.latent,
            temporal_mode=args.temporal_mode,
            temporal_steps=sample["temporal_features"].shape[1],
            global_history_mode=args.global_history_mode,
            temporal_blend_decay=args.temporal_blend_decay,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=1.0e-6,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "train_sha256": file_sha256(train_path),
            "validation_sha256": file_sha256(val_path),
            "device": str(device),
            "model_type": type(model).__name__,
            "prediction_task": "C(depth,time)",
            "query_channels": ["time_norm", "depth_norm", "average_cp_prior_wt"],
        }
    )
    with open(args.output_dir / "training_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)

    best_score = float("inf")
    epochs_without_improvement = 0
    last_checkpoint = None
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        totals: dict[str, float] = {}
        pair_iterator = iter(pair_loader) if pair_loader is not None else None
        for batch in train_loader:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            terms = supervised_terms(
                model,
                batch,
                args,
                include_physics=epoch > args.physics_warmup_epochs,
            )
            loss = terms["loss"]
            current_pair_loss = loss.new_tensor(0.0)
            if pair_iterator is not None:
                try:
                    pair_batch = next(pair_iterator)
                except StopIteration:
                    pair_iterator = iter(pair_loader)
                    pair_batch = next(pair_iterator)
                pair_batch = move_to_device(pair_batch, device)
                current_pair_loss = pair_loss(model, pair_batch, args)
                loss = loss + args.pair_weight * current_pair_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            totals["pair_l1"] = totals.get("pair_l1", 0.0) + float(current_pair_loss.detach().cpu())
            totals["optimized_loss"] = totals.get("optimized_loss", 0.0) + float(loss.detach().cpu())

        train_metrics = {f"train_{key}": value / len(train_loader) for key, value in totals.items()}
        validation = validate(model, val_loader, device, args)
        scheduler.step(validation["selection_score"])
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **{f"validation_{k}": v for k, v in validation.items()},
        }
        append_csv(args.output_dir / "history.csv", row)
        print(json.dumps(row))
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": validation,
            "config": config,
        }
        last_checkpoint = checkpoint
        if validation["selection_score"] < best_score:
            best_score = validation["selection_score"]
            epochs_without_improvement = 0
            torch.save(checkpoint, args.output_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.early_stopping_patience:
            print(json.dumps({"early_stopping_epoch": epoch, "best_selection_score": best_score}))
            break
    if last_checkpoint is not None:
        torch.save(last_checkpoint, args.output_dir / "last.pt")


if __name__ == "__main__":
    main()
