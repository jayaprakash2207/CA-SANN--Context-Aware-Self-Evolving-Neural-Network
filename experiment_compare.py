from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import (
    IndexedImageDataset,
    TrainingConfig,
    ensure_directory,
    make_datasets,
    plot_experiment_metrics,
    run_experiment,
    set_seed,
    setup_logging,
)


def _efficiency_metrics(row: dict[str, Any]) -> dict[str, float]:
    params = max(1.0, float(row["model_size"]))
    test_acc = float(row["test_accuracy"])
    return {
        "accuracy_per_100k_params": test_acc / (params / 100000.0),
    }


def _build_loaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = make_datasets(config)
    pin_memory = config.device.startswith("cuda") and torch.cuda.is_available()

    train_loader = DataLoader(
        IndexedImageDataset(train_ds),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        IndexedImageDataset(val_ds),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        IndexedImageDataset(test_ds),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def _format_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Model",
        "Val Acc",
        "Test Acc",
        "Val Loss",
        "Test Loss",
        "Params",
        "Candidate",
        "Explore",
        "Growth Events",
        "Rejected",
        "Train Time (s)",
    ]

    rendered_rows = []
    for row in rows:
        rendered_rows.append(
            [
                row["model"],
                f"{row['val_accuracy']:.4f}",
                f"{row['test_accuracy']:.4f}",
                f"{row['val_loss']:.6f}",
                f"{row['test_loss']:.6f}",
                str(row["model_size"]),
                str(row.get("candidate_growth_events", 0)),
                str(row.get("exploration_growth_events", 0)),
                str(row["growth_events"]),
                str(row.get("rejected_growth_events", 0)),
                f"{row['training_time_sec']:.2f}",
            ]
        )

    widths = [len(h) for h in headers]
    for values in rendered_rows:
        for idx, value in enumerate(values):
            widths[idx] = max(widths[idx], len(value))

    def line(values: list[str]) -> str:
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    table = [line(headers), separator]
    table.extend(line(values) for values in rendered_rows)
    return "\n".join(table)


def _verdict(static_row: dict[str, Any], sann_row: dict[str, Any]) -> tuple[str, str]:
    acc_delta = sann_row["test_accuracy"] - static_row["test_accuracy"]
    eff_delta = float(sann_row["accuracy_per_100k_params"]) - float(static_row["accuracy_per_100k_params"])
    loss_delta = sann_row["test_loss"] - static_row["test_loss"]

    acc_eps = 0.005
    eff_eps = 1e-6
    loss_eps = 0.01

    if acc_delta > acc_eps and eff_delta > eff_eps:
        return (
            "better",
            f"SANN is better: test accuracy +{acc_delta:.4f}, efficiency +{eff_delta:.6f}, test loss delta {loss_delta:+.6f}",
        )
    if abs(acc_delta) <= acc_eps and eff_delta > eff_eps:
        return (
            "better",
            f"SANN is better on efficiency: test accuracy delta {acc_delta:+.4f}, efficiency +{eff_delta:.6f}, test loss delta {loss_delta:+.6f}",
        )
    if acc_delta > acc_eps and eff_delta < -eff_eps:
        return (
            "trade-off",
            f"SANN is a trade-off: test accuracy +{acc_delta:.4f}, efficiency {eff_delta:+.6f}, test loss delta {loss_delta:+.6f}",
        )
    if abs(acc_delta) <= acc_eps and abs(eff_delta) <= eff_eps and abs(loss_delta) <= loss_eps:
        return (
            "similar",
            f"SANN is similar: test accuracy delta {acc_delta:+.4f}, efficiency delta {eff_delta:+.6f}, test loss delta {loss_delta:+.6f}",
        )
    return (
        "worse",
        f"SANN is worse: test accuracy {acc_delta:+.4f}, efficiency {eff_delta:+.6f}, test loss delta {loss_delta:+.6f}",
    )


def _tradeoff_summary(static_row: dict[str, Any], sann_row: dict[str, Any]) -> dict[str, float]:
    acc_delta = float(sann_row["test_accuracy"]) - float(static_row["test_accuracy"])
    size_delta = float(sann_row["model_size"]) - float(static_row["model_size"])
    size_ratio = float(sann_row["model_size"]) / max(1.0, float(static_row["model_size"]))
    acc_gain_per_1k_params = acc_delta / max(1.0, size_delta / 1000.0)
    return {
        "test_accuracy_delta": acc_delta,
        "model_size_delta": size_delta,
        "model_size_ratio": size_ratio,
        "accuracy_gain_per_1k_params": acc_gain_per_1k_params,
    }


def _efficiency_summary(static_row: dict[str, Any], sann_row: dict[str, Any]) -> dict[str, Any]:
    static_eff = float(static_row["accuracy_per_100k_params"])
    sann_eff = float(sann_row["accuracy_per_100k_params"])
    eff_delta = sann_eff - static_eff
    if eff_delta > 1e-9:
        winner = "SANN"
    elif eff_delta < -1e-9:
        winner = "Static"
    else:
        winner = "Tie"
    return {
        "static_accuracy_per_100k_params": static_eff,
        "sann_accuracy_per_100k_params": sann_eff,
        "efficiency_delta": eff_delta,
        "efficiency_ratio": sann_eff / max(1e-12, static_eff),
        "winner": winner,
    }


def _plot_accuracy_vs_model_size(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    models = [str(row["model"]) for row in rows]
    sizes = [float(row["model_size"]) for row in rows]
    accs = [float(row["test_accuracy"]) for row in rows]
    effs = [float(row["accuracy_per_100k_params"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"Static": "#1f77b4", "SANN": "#d62728"}

    for model, x, y, eff in zip(models, sizes, accs, effs):
        ax.scatter(x, y, s=140, color=colors.get(model, "#2ca02c"), alpha=0.9, edgecolors="white", linewidths=1.0)
        ax.annotate(
            f"{model}\nacc/100k={eff:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

    ax.set_title("Accuracy vs Model Size")
    ax.set_xlabel("Model Parameters")
    ax.set_ylabel("Test Accuracy")
    ax.grid(True, linestyle="--", alpha=0.35)

    x_pad = max(1000.0, 0.05 * max(sizes))
    y_min = max(0.0, min(accs) - 0.03)
    y_max = min(1.0, max(accs) + 0.03)
    ax.set_xlim(min(sizes) - x_pad, max(sizes) + x_pad)
    ax.set_ylim(y_min, y_max)

    plot_path = output_dir / "accuracy_vs_model_size.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def run_comparison(config: TrainingConfig) -> dict[str, Any]:
    setup_logging()
    set_seed(config.seed)

    train_loader, val_loader, test_loader = _build_loaders(config)

    t0 = time.perf_counter()
    sann_config = replace(config, context_gating=False)
    sann_result = run_experiment(
        experiment_name="sann",
        config=sann_config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dynamic=True,
    )
    sann_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    static_result = run_experiment(
        experiment_name="static",
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dynamic=False,
    )
    static_time = time.perf_counter() - t1

    rows = [
        {
            "model": "Static",
            "val_accuracy": static_result.final_val_accuracy,
            "test_accuracy": static_result.test_accuracy,
            "val_loss": static_result.final_val_loss,
            "test_loss": static_result.test_loss,
            "model_size": static_result.final_model_size,
            "growth_events": static_result.growth_event_count,
            "candidate_growth_events": static_result.candidate_growth_event_count,
            "exploration_growth_events": static_result.exploration_growth_event_count,
            "rejected_growth_events": static_result.rejected_growth_event_count,
            "val_efficiency": static_result.final_efficiency,
            "training_time_sec": static_time,
            "checkpoint": static_result.final_checkpoint,
        },
        {
            "model": "SANN",
            "val_accuracy": sann_result.final_val_accuracy,
            "test_accuracy": sann_result.test_accuracy,
            "val_loss": sann_result.final_val_loss,
            "test_loss": sann_result.test_loss,
            "model_size": sann_result.final_model_size,
            "growth_events": sann_result.growth_event_count,
            "candidate_growth_events": sann_result.candidate_growth_event_count,
            "exploration_growth_events": sann_result.exploration_growth_event_count,
            "rejected_growth_events": sann_result.rejected_growth_event_count,
            "val_efficiency": sann_result.final_efficiency,
            "training_time_sec": sann_time,
            "checkpoint": sann_result.final_checkpoint,
        },
    ]

    for row in rows:
        row.update(_efficiency_metrics(row))

    verdict_label, verdict_text = _verdict(rows[0], rows[1])
    tradeoff = _tradeoff_summary(rows[0], rows[1])
    efficiency = _efficiency_summary(rows[0], rows[1])

    output_dir = Path(config.output_dir)
    ensure_directory(output_dir)
    efficiency_plot_path = _plot_accuracy_vs_model_size(rows, output_dir)
    metrics_plot_path = plot_experiment_metrics(
        output_dir,
        {"SANN": sann_result.metrics, "Static": static_result.metrics},
        plot_filename="metrics.png",
    )

    payload = {
        "config": {
            "seed": config.seed,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "complexity_lambda": config.complexity_lambda,
            "train_samples": config.train_samples,
            "val_samples": config.val_samples,
            "test_samples": config.test_samples,
            "dataset_root": config.dataset_root,
            "train_label_noise": config.train_label_noise,
            "growth_budget_neurons": config.growth_budget_neurons,
            "growth_min_val_acc_gain": config.growth_min_val_acc_gain,
            "growth_min_val_loss_drop": config.growth_min_val_loss_drop,
            "growth_accept_min_val_acc_gain": config.growth_accept_min_val_acc_gain,
            "growth_accept_min_efficiency_gain": config.growth_accept_min_efficiency_gain,
            "exploration_budget_events": config.exploration_budget_events,
            "exploration_phase_epochs": config.exploration_phase_epochs,
            "exploration_eval_delay_epochs": config.exploration_eval_delay_epochs,
            "architecture_eval_batches": config.architecture_eval_batches,
            "architecture_eval_train_batches": config.architecture_eval_train_batches,
            "enable_candidate_growth": config.enable_candidate_growth,
            "candidate_min_growth_priority": config.candidate_min_growth_priority,
            "candidate_min_underperformance_to_grow": config.candidate_min_underperformance_to_grow,
            "candidate_max_layer_importance_to_grow": config.candidate_max_layer_importance_to_grow,
            "candidate_weak_neuron_fraction_for_growth": config.candidate_weak_neuron_fraction_for_growth,
            "candidate_growth_neurons": config.candidate_growth_neurons,
            "efficiency_drop_tolerance_early": config.efficiency_drop_tolerance_early,
            "efficiency_drop_tolerance_late": config.efficiency_drop_tolerance_late,
            "efficiency_decline_patience": config.efficiency_decline_patience,
        },
        "results": rows,
        "tracking": {
            "sann": {
                "epoch": sann_result.metrics.epoch,
                "efficiency": sann_result.metrics.efficiency,
                "peak_efficiency": sann_result.metrics.peak_efficiency,
                "growth_enabled": sann_result.metrics.growth_enabled,
                "capacity_status": sann_result.metrics.capacity_status,
                "difficulty_score": sann_result.metrics.difficulty_score,
                "growth_allowed": sann_result.metrics.growth_allowed,
                "candidate_growth_events": sann_result.metrics.candidate_growth_events,
                "exploration_growth_events": sann_result.metrics.exploration_growth_events,
                "accepted_growth_events": sann_result.metrics.accepted_growth_events,
                "rejected_growth_events": sann_result.metrics.rejected_growth_events,
                "learning_policy": sann_result.policy_summary,
            },
            "static": {
                "epoch": static_result.metrics.epoch,
                "efficiency": static_result.metrics.efficiency,
                "peak_efficiency": static_result.metrics.peak_efficiency,
                "growth_enabled": static_result.metrics.growth_enabled,
                "capacity_status": static_result.metrics.capacity_status,
                "difficulty_score": static_result.metrics.difficulty_score,
                "growth_allowed": static_result.metrics.growth_allowed,
                "candidate_growth_events": static_result.metrics.candidate_growth_events,
                "exploration_growth_events": static_result.metrics.exploration_growth_events,
                "accepted_growth_events": static_result.metrics.accepted_growth_events,
                "rejected_growth_events": static_result.metrics.rejected_growth_events,
                "learning_policy": static_result.policy_summary,
            },
        },
        "verdict": {
            "label": verdict_label,
            "message": verdict_text,
        },
        "tradeoff": tradeoff,
        "efficiency": efficiency,
        "artifacts": {
            "accuracy_vs_model_size_plot": str(efficiency_plot_path),
            "metrics_plot": str(metrics_plot_path),
        },
    }

    json_path = output_dir / "comparison_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = output_dir / "comparison_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nModel Comparison")
    print(_format_table(rows))
    print("\nAccuracy vs Model Size Trade-off")
    print(
        "SANN-static: "
        f"test_acc_delta={tradeoff['test_accuracy_delta']:+.4f}, "
        f"size_delta={tradeoff['model_size_delta']:+.0f}, "
        f"size_ratio={tradeoff['model_size_ratio']:.3f}x, "
        f"acc_gain_per_1k_params={tradeoff['accuracy_gain_per_1k_params']:+.6f}"
    )
    print("\nEfficiency Comparison")
    print(
        "Accuracy per parameter (per 100k): "
        f"Static={efficiency['static_accuracy_per_100k_params']:.6f}, "
        f"SANN={efficiency['sann_accuracy_per_100k_params']:.6f}, "
        f"delta={efficiency['efficiency_delta']:+.6f}, "
        f"ratio={efficiency['efficiency_ratio']:.3f}x, "
        f"winner={efficiency['winner']}"
    )
    print(f"\nVerdict: {verdict_text}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Plot (trade-off): {efficiency_plot_path}")
    print(f"Saved Plot (timeline): {metrics_plot_path}")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reusable SANN vs static comparison experiment")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--hidden-dims", type=str, default="128,64")
    parser.add_argument("--difficulty-threshold", type=float, default=0.55)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="runs/mnist_sann_comparison_proper")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-label-noise", type=float, default=0.15)
    parser.add_argument("--complexity-lambda", type=float, default=1e-6)
    parser.add_argument("--complexity-growth-power", type=float, default=1.35)
    parser.add_argument("--force-growth-debug", action="store_true")
    parser.add_argument("--force-growth-interval", type=int, default=2)
    parser.add_argument("--force-growth-neurons", type=int, default=2)
    parser.add_argument("--growth-budget-neurons", type=int, default=48)
    parser.add_argument("--growth-min-val-acc-gain", type=float, default=0.0025)
    parser.add_argument("--growth-min-val-loss-drop", type=float, default=0.004)
    parser.add_argument("--growth-accept-min-val-acc-gain", type=float, default=0.0)
    parser.add_argument("--growth-accept-min-efficiency-gain", type=float, default=0.0)
    parser.add_argument("--exploration-budget-events", type=int, default=2)
    parser.add_argument("--exploration-phase-epochs", type=int, default=6)
    parser.add_argument("--exploration-eval-delay-epochs", type=int, default=2)
    parser.add_argument("--architecture-eval-batches", type=int, default=10)
    parser.add_argument("--architecture-eval-train-batches", type=int, default=8)
    parser.add_argument("--enable-candidate-growth", action="store_true")
    parser.add_argument("--disable-candidate-growth", action="store_true")
    parser.add_argument("--candidate-min-growth-priority", type=float, default=0.45)
    parser.add_argument("--candidate-min-underperformance", type=float, default=0.05)
    parser.add_argument("--candidate-max-layer-importance", type=float, default=0.90)
    parser.add_argument("--candidate-weak-neuron-fraction", type=float, default=0.10)
    parser.add_argument("--candidate-growth-neurons", type=int, default=2)
    parser.add_argument("--efficiency-drop-tol-early", type=float, default=2.0e-7)
    parser.add_argument("--efficiency-drop-tol-late", type=float, default=5.0e-8)
    parser.add_argument("--efficiency-decline-patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _parse_hidden_dims(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    dims: list[int] = []
    for p in parts:
        dims.append(int(p))
    if not dims:
        raise ValueError("--hidden-dims must contain at least one integer, e.g. 128,64")
    return tuple(dims)


def main() -> None:
    args = parse_args()
    dataset_name = str(args.dataset).lower().strip()
    inferred_input_dim = 32 * 32 * 3 if dataset_name in {"cifar10", "cifar-10"} else 28 * 28
    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    config = replace(
        TrainingConfig(),
        dataset_name=args.dataset,
        input_dim=inferred_input_dim,
        hidden_dims=hidden_dims,
        difficulty_threshold=float(args.difficulty_threshold),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        seed=args.seed,
        train_label_noise=args.train_label_noise,
        complexity_lambda=args.complexity_lambda,
        complexity_growth_power=args.complexity_growth_power,
        force_growth_debug=args.force_growth_debug,
        force_growth_interval_epochs=args.force_growth_interval,
        force_growth_neurons=args.force_growth_neurons,
        growth_budget_neurons=args.growth_budget_neurons,
        growth_min_val_acc_gain=args.growth_min_val_acc_gain,
        growth_min_val_loss_drop=args.growth_min_val_loss_drop,
        growth_accept_min_val_acc_gain=args.growth_accept_min_val_acc_gain,
        growth_accept_min_efficiency_gain=args.growth_accept_min_efficiency_gain,
        exploration_budget_events=args.exploration_budget_events,
        exploration_phase_epochs=args.exploration_phase_epochs,
        exploration_eval_delay_epochs=args.exploration_eval_delay_epochs,
        architecture_eval_batches=args.architecture_eval_batches,
        architecture_eval_train_batches=args.architecture_eval_train_batches,
        enable_candidate_growth=(False if args.disable_candidate_growth else (True if args.enable_candidate_growth else True)),
        candidate_min_growth_priority=args.candidate_min_growth_priority,
        candidate_min_underperformance_to_grow=args.candidate_min_underperformance,
        candidate_max_layer_importance_to_grow=args.candidate_max_layer_importance,
        candidate_weak_neuron_fraction_for_growth=args.candidate_weak_neuron_fraction,
        candidate_growth_neurons=args.candidate_growth_neurons,
        efficiency_drop_tolerance_early=args.efficiency_drop_tol_early,
        efficiency_drop_tolerance_late=args.efficiency_drop_tol_late,
        efficiency_decline_patience=args.efficiency_decline_patience,
        dataloader_num_workers=args.num_workers,
    )
    run_comparison(config)


if __name__ == "__main__":
    main()
