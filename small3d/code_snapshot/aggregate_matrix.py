#!/usr/bin/env python3
"""Aggregate the matched matrix and apply the pre-registered decision gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BACKBONES = ("mlp", "resnet", "fno")
MODES = ("direct", "prior_direct", "residual")
METRICS = (
    "active_field_rel_l2", "active_mae_wt", "surface_c_mae_wt",
    "final_active_mae_wt", "mass_uptake_rel_error",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_cases(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def hierarchical_ci(differences: np.ndarray, seed: int, repeats: int) -> list[float]:
    """differences shape: [num_seeds, num_cases]."""
    rng = np.random.default_rng(seed); ns, nc = differences.shape
    values = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 500):
        stop = min(start + 500, repeats)
        seed_idx = rng.integers(0, ns, size=(stop - start, ns))
        batch = np.empty(stop - start)
        for i, selected in enumerate(seed_idx):
            per_seed = []
            for s in selected:
                case_idx = rng.integers(0, nc, size=nc)
                per_seed.append(differences[s, case_idx].mean())
            batch[i] = np.mean(per_seed)
        values[start:stop] = batch
    return np.quantile(values, [0.025, 0.975]).astype(float).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args(); root = args.experiment_root
    cells = {}; case_tables = {}; rows = []
    for backbone in BACKBONES:
        for mode in MODES:
            summaries = []; train_summaries = []; tables = []
            for seed in args.seeds:
                run = f"{backbone}_{mode}_seed_{seed}"
                test = read_json(root / "results" / run / "test" / "comparison.json")
                train = read_json(root / "results" / run / "train" / "comparison.json")
                summaries.append(test["prediction"]); train_summaries.append(train["prediction"])
                tables.append(read_cases(root / "results" / run / "test" / "prediction" / "case_table.csv"))
            summary = {
                metric: {
                    "mean": float(np.mean([s[metric] for s in summaries])),
                    "std_across_seeds": float(
                        np.std([s[metric] for s in summaries], ddof=1)
                        if len(summaries) > 1 else 0.0
                    ),
                    "per_seed": [float(s[metric]) for s in summaries],
                } for metric in METRICS
            }
            summary["parameter_count"] = int(summaries[0]["parameter_count"])
            summary["forward_time_mean_s"] = float(np.mean([s["model_forward_time_per_case_sequence_s"] for s in summaries]))
            summary["train_active_rel_l2_mean"] = float(np.mean([s["active_field_rel_l2"] for s in train_summaries]))
            summary["test_train_active_rel_l2_ratio"] = summary["active_field_rel_l2"]["mean"] / max(summary["train_active_rel_l2_mean"], 1e-12)
            cells[(backbone, mode)] = summary; case_tables[(backbone, mode)] = tables
            rows.append({
                "backbone": backbone, "mode": mode, "seeds": ";".join(map(str, args.seeds)),
                "parameter_count": summary["parameter_count"], "forward_time_s": summary["forward_time_mean_s"],
                "train_active_rel_l2": summary["train_active_rel_l2_mean"],
                **{metric: summary[metric]["mean"] for metric in METRICS},
            })

    comparisons = {}; backbone_gates = {}
    for bidx, backbone in enumerate(BACKBONES):
        comparisons[backbone] = {}; res = cells[(backbone, "residual")]
        for cidx, reference in enumerate(("direct", "prior_direct")):
            ref = cells[(backbone, reference)]; metric_results = {}
            for midx, metric in enumerate(METRICS):
                res_arrays = []; ref_arrays = []
                for sidx, _seed in enumerate(args.seeds):
                    res_rows = case_tables[(backbone, "residual")][sidx]
                    ref_rows = case_tables[(backbone, reference)][sidx]
                    res_arrays.append([float(r[metric]) for r in res_rows])
                    ref_arrays.append([float(r[metric]) for r in ref_rows])
                diff = np.asarray(res_arrays) - np.asarray(ref_arrays)
                metric_results[metric] = {
                    "reference_mean": ref[metric]["mean"], "residual_mean": res[metric]["mean"],
                    "relative_change": res[metric]["mean"] / ref[metric]["mean"] - 1.0,
                    "paired_difference": float(diff.mean()),
                    "paired_difference_95pct_ci": hierarchical_ci(diff, 7000 + 100*bidx + 10*cidx + midx, args.bootstrap),
                }
            comparisons[backbone][f"residual_vs_{reference}"] = metric_results
        vs_direct = comparisons[backbone]["residual_vs_direct"]
        vs_prior = comparisons[backbone]["residual_vs_prior_direct"]
        backbone_gates[backbone] = {
            "residual_beats_direct_active_rel_l2": vs_direct["active_field_rel_l2"]["relative_change"] < 0.0,
            "residual_beats_prior_direct_by_5pct": vs_prior["active_field_rel_l2"]["relative_change"] <= -0.05,
            "active_rel_l2_ci_excludes_zero_vs_prior_direct": (
                vs_prior["active_field_rel_l2"]["paired_difference_95pct_ci"][1] < 0.0
            ),
            "other_metrics_not_worse_than_2pct_vs_prior_direct": all(
                vs_prior[m]["relative_change"] <= 0.02 for m in METRICS[1:]
            ),
        }
    direct_wins = sum(g["residual_beats_direct_active_rel_l2"] for g in backbone_gates.values())
    screening_prior_wins = sum(
        g["residual_beats_prior_direct_by_5pct"] and g["other_metrics_not_worse_than_2pct_vs_prior_direct"]
        for g in backbone_gates.values()
    )
    formal_prior_wins = sum(
        g["residual_beats_prior_direct_by_5pct"]
        and g["active_rel_l2_ci_excludes_zero_vs_prior_direct"]
        and g["other_metrics_not_worse_than_2pct_vs_prior_direct"]
        for g in backbone_gates.values()
    )
    if len(args.seeds) == 1:
        decision_stage = "single_seed_screening"
        strong_prior_wins = screening_prior_wins
    else:
        decision_stage = "multi_seed_formal"
        strong_prior_wins = formal_prior_wins
    decision = "GO" if direct_wins == 3 and strong_prior_wins >= 2 else "STOP"
    output = {
        "seeds": args.seeds, "num_backbones": 3, "num_modes": 3, "num_test_cases": 22,
        "cells": {f"{b}_{m}": v for (b, m), v in cells.items()},
        "comparisons": comparisons, "backbone_gates": backbone_gates,
        "gate_counts": {
            "residual_beats_direct": direct_wins,
            "screening_residual_vs_prior_direct": screening_prior_wins,
            "formal_ci_supported_residual_vs_prior_direct": formal_prior_wins,
            "strong_residual_vs_prior_direct_for_current_stage": strong_prior_wins,
        },
        "decision_stage": decision_stage,
        "decision": decision,
    }
    suffix = "_".join(map(str, args.seeds)); outdir = root / "results"; outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"matrix_summary_seeds_{suffix}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (outdir / f"matrix_table_seeds_{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
