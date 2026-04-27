from __future__ import annotations

import argparse
import gc
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
from utils import format_table, summarize_numeric, write_csv, write_json


MODEL_ORDER = ["Static", "SANN", "CA-SANN"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full experimental pipeline: Static vs SANN vs CA-SANN")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["mnist", "fashion_mnist", "cifar10"],
        help="Datasets: mnist | fashion_mnist | cifar10",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 23, 33, 43, 53])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=5000)
    parser.add_argument("--difficulty-threshold", type=float, default=0.55)
    parser.add_argument("--output-dir", type=str, default="runs/ca_sann_full_pipeline")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--num-workers", type=int, default=0)

    # Growth knobs (kept minimal; advanced tuning stays in train.py defaults)
    parser.add_argument("--complexity-lambda", type=float, default=1e-6)
    parser.add_argument("--complexity-growth-power", type=float, default=1.35)
    parser.add_argument("--growth-budget-neurons", type=int, default=48)

    # Reuse existing results
    parser.add_argument("--reuse", action="store_true", help="Reuse per-run JSON if present")

    return parser.parse_args()


def _accuracy_per_100k_params(test_accuracy: float, params: int) -> float:
    denom = max(1.0, float(params)) / 100000.0
    return float(test_accuracy) / denom


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


def _save_run_payload(run_dir: Path, payload: dict[str, Any]) -> Path:
    ensure_directory(run_dir)
    path = run_dir / "result.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_run_payload(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _result_to_payload(result: Any, *, model_label: str, dataset: str, seed: int, training_time_sec: float) -> dict[str, Any]:
    params = int(result.final_model_size)
    test_acc = float(result.test_accuracy)
    return {
        "dataset": str(dataset),
        "seed": int(seed),
        "model": str(model_label),
        "val_accuracy": float(result.final_val_accuracy),
        "test_accuracy": test_acc,
        "val_loss": float(result.final_val_loss),
        "test_loss": float(result.test_loss),
        "model_size": params,
        "efficiency": float(result.final_efficiency),
        "accuracy_per_100k_params": _accuracy_per_100k_params(test_acc, params),
        "growth_events": int(result.growth_event_count),
        "candidate_growth_events": int(result.candidate_growth_event_count),
        "exploration_growth_events": int(result.exploration_growth_event_count),
        "rejected_growth_events": int(result.rejected_growth_event_count),
        "safe_growth_accepted": int(result.safe_growth_accepted_count),
        "safe_growth_rejected_peak": int(result.safe_growth_rejected_peak_count),
        "training_time_sec": float(training_time_sec),
        "final_checkpoint": str(result.final_checkpoint),
        "policy_summary": result.policy_summary,
        "time_series": {
            "epoch": result.metrics.epoch,
            "val_accuracy": result.metrics.val_accuracy,
            "val_loss": result.metrics.val_loss,
            "model_size": result.metrics.model_size,
            "efficiency": result.metrics.efficiency,
            "peak_efficiency": result.metrics.peak_efficiency,
            "growth_events": result.metrics.growth_events,
            "candidate_growth_events": result.metrics.candidate_growth_events,
            "exploration_growth_events": result.metrics.exploration_growth_events,
            "accepted_growth_events": result.metrics.accepted_growth_events,
            "rejected_growth_events": result.metrics.rejected_growth_events,
            "capacity_status": result.metrics.capacity_status,
            "difficulty_score": result.metrics.difficulty_score,
            "growth_allowed": result.metrics.growth_allowed,
        },
    }


def _aggregate(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault((str(row["dataset"]), str(row["model"])), []).append(row)

    metrics = [
        "val_accuracy",
        "test_accuracy",
        "val_loss",
        "test_loss",
        "model_size",
        "efficiency",
        "accuracy_per_100k_params",
        "growth_events",
        "candidate_growth_events",
        "exploration_growth_events",
        "rejected_growth_events",
        "training_time_sec",
    ]

    out: list[dict[str, Any]] = []
    for (dataset, model), rows in grouped.items():
        agg: dict[str, Any] = {"dataset": dataset, "model": model, "n_seeds": len(rows)}
        for metric in metrics:
            values = [float(r.get(metric, 0.0)) for r in rows]
            m, s = summarize_numeric(values)
            agg[f"{metric}_mean"] = float(m)
            agg[f"{metric}_std"] = float(s)
        out.append(agg)

    order = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    out.sort(key=lambda r: (str(r["dataset"]), order.get(str(r["model"]), 999)))
    return out


def _plot_dataset_summary(output_dir: Path, *, dataset: str, aggregate_rows: list[dict[str, Any]]) -> None:
    rows = [r for r in aggregate_rows if str(r.get("dataset", "")).lower().strip() == dataset]
    if not rows:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy vs model size
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=160)
    colors = {"Static": "#1f77b4", "SANN": "#d62728", "CA-SANN": "#2ca02c"}
    for row in rows:
        model = str(row["model"])
        x = float(row.get("model_size_mean", 0.0))
        y = float(row.get("test_accuracy_mean", 0.0))
        ax.scatter([x], [y], s=140, color=colors.get(model, "#555555"), alpha=0.9, edgecolors="white", linewidths=1.0)
        ax.annotate(model, (x, y), textcoords="offset points", xytext=(8, 8), fontsize=9)

    ax.set_title(f"{dataset}: Test Accuracy vs Model Size")
    ax.set_xlabel("Trainable Parameters (mean)")
    ax.set_ylabel("Test Accuracy (mean)")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_model_size.png")
    plt.close(fig)

    # Efficiency (accuracy per 100k params)
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    rows_sorted = sorted(rows, key=lambda r: MODEL_ORDER.index(str(r["model"])) if str(r["model"]) in MODEL_ORDER else 999)
    labels = [str(r["model"]) for r in rows_sorted]
    values = [float(r.get("accuracy_per_100k_params_mean", 0.0)) for r in rows_sorted]
    ax2.bar(labels, values, color=[colors.get(l, "#555555") for l in labels])
    ax2.set_title(f"{dataset}: Efficiency (Acc / 100k params)")
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Accuracy per 100k parameters (mean)")
    ax2.grid(True, axis="y", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(output_dir / "efficiency_per_100k_params.png")
    plt.close(fig2)

    # Growth events
    fig3, ax3 = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    growth_means = [float(r.get("growth_events_mean", 0.0)) for r in rows_sorted]
    ax3.bar(labels, growth_means, color=[colors.get(l, "#555555") for l in labels])
    ax3.set_title(f"{dataset}: Growth Events (mean)")
    ax3.set_xlabel("Model")
    ax3.set_ylabel("Growth events (mean)")
    ax3.grid(True, axis="y", alpha=0.25)
    fig3.tight_layout()
    fig3.savefig(output_dir / "growth_events_mean.png")
    plt.close(fig3)


def _analyze_growth(seed_rows: list[dict[str, Any]], *, dataset: str) -> dict[str, Any]:
    # Analyze per-seed runs to infer when growth helped vs was unnecessary.
    rows = [r for r in seed_rows if str(r.get("dataset", "")).lower().strip() == dataset]
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[str(row["model"])] = row

    acc_eps = 0.005

    growth_helpful = 0
    growth_unnecessary = 0
    growth_wasted = 0

    helpful_episodes_total = 0
    accepted_events_total = 0
    blocked_ratio_values: list[float] = []

    for seed, models in by_seed.items():
        static_row = models.get("Static")
        sann_row = models.get("SANN")
        ca_row = models.get("CA-SANN")
        if static_row is None or sann_row is None or ca_row is None:
            continue

        ca_growth = int(ca_row.get("growth_events", 0))
        ca_rejected = int(ca_row.get("rejected_growth_events", 0))
        ca_test = float(ca_row.get("test_accuracy", 0.0))
        best = max(float(static_row.get("test_accuracy", 0.0)), float(sann_row.get("test_accuracy", 0.0)), ca_test)

        if ca_growth > 0 and (ca_test - float(static_row.get("test_accuracy", 0.0))) > acc_eps:
            growth_helpful += 1

        if ca_growth == 0 and (best - ca_test) <= acc_eps:
            growth_unnecessary += 1

        if ca_rejected > 0 and ca_growth == 0:
            growth_wasted += 1

        ts = ca_row.get("time_series", {})
        accepted = [int(v) for v in ts.get("accepted_growth_events", [])]
        val_acc = [float(v) for v in ts.get("val_accuracy", [])]
        eff = [float(v) for v in ts.get("efficiency", [])]
        growth_allowed = [int(v) for v in ts.get("growth_allowed", [])]

        if accepted:
            accepted_events_total += int(accepted[-1])
        if growth_allowed:
            blocked_ratio_values.append(1.0 - (sum(1 for v in growth_allowed if int(v) == 1) / float(len(growth_allowed))))

        # Helpful episodes: accepted growth at epoch t followed by acc/eff increase relative to t-1.
        for idx in range(1, min(len(accepted), len(val_acc), len(eff))):
            accepted_delta = int(accepted[idx]) - int(accepted[idx - 1])
            if accepted_delta <= 0:
                continue
            acc_gain = float(val_acc[idx]) - float(val_acc[idx - 1])
            eff_gain = float(eff[idx]) - float(eff[idx - 1])
            if acc_gain > 0.0 or eff_gain > 0.0:
                helpful_episodes_total += 1

    blocked_ratio_mean = sum(blocked_ratio_values) / float(len(blocked_ratio_values)) if blocked_ratio_values else 0.0

    return {
        "dataset": dataset,
        "seeds_total": len(by_seed),
        "growth_helpful_seeds": growth_helpful,
        "growth_unnecessary_seeds": growth_unnecessary,
        "growth_wasted_seeds": growth_wasted,
        "ca_accepted_events_total": accepted_events_total,
        "ca_helpful_episodes_total": helpful_episodes_total,
        "ca_blocked_ratio_mean": blocked_ratio_mean,
        "notes": {
            "definition_growth_helpful": "CA-SANN had >0 growth events and beat Static test accuracy by >0.5%.",
            "definition_growth_unnecessary": "CA-SANN had 0 growth events and was within 0.5% of the best model for that seed.",
            "definition_growth_wasted": "CA-SANN had rejected growth events but ended with 0 growth events.",
        },
    }


def _write_report(output_dir: Path, *, aggregate_rows: list[dict[str, Any]], growth_analysis: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("CA-SANN Full Experimental Pipeline Report")
    lines.append("=======================================")
    lines.append("")
    lines.append("Models compared: Static, SANN (ungated growth), CA-SANN (context-gated growth)")
    lines.append("Metrics: accuracy, model size, efficiency, growth events")
    lines.append("")

    datasets = sorted({str(r["dataset"]).lower().strip() for r in aggregate_rows})
    for dataset in datasets:
        lines.append(f"Dataset: {dataset}")
        lines.append("-" * (9 + len(dataset)))
        rows = [r for r in aggregate_rows if str(r["dataset"]).lower().strip() == dataset]
        order = {name: idx for idx, name in enumerate(MODEL_ORDER)}
        rows.sort(key=lambda r: order.get(str(r["model"]), 999))

        headers = ["Model", "Test Acc (mean±std)", "Params (mean)", "Eff/100k (mean)", "Growth (mean)"]
        table_rows: list[list[str]] = []
        for r in rows:
            table_rows.append(
                [
                    str(r["model"]),
                    f"{float(r['test_accuracy_mean']):.4f} ± {float(r['test_accuracy_std']):.4f}",
                    f"{float(r['model_size_mean']):.0f}",
                    f"{float(r['accuracy_per_100k_params_mean']):.4f}",
                    f"{float(r['growth_events_mean']):.2f}",
                ]
            )
        lines.append(format_table(headers, table_rows))
        lines.append("")

        analysis = next((a for a in growth_analysis if str(a.get("dataset", "")).lower().strip() == dataset), None)
        if analysis is not None:
            lines.append("Automatic growth analysis")
            lines.append(f"- Seeds analyzed: {int(analysis.get('seeds_total', 0))}")
            lines.append(
                "- Growth helpful (seed-level): "
                f"{int(analysis.get('growth_helpful_seeds', 0))}"
            )
            lines.append(
                "- Growth unnecessary (seed-level): "
                f"{int(analysis.get('growth_unnecessary_seeds', 0))}"
            )
            lines.append(
                "- Growth wasted (rejected then rolled back): "
                f"{int(analysis.get('growth_wasted_seeds', 0))}"
            )
            lines.append(
                "- Helpful growth episodes (epoch-level): "
                f"{int(analysis.get('ca_helpful_episodes_total', 0))}"
            )
            lines.append(
                "- Mean blocked ratio (fraction of epochs growth_allowed==0): "
                f"{float(analysis.get('ca_blocked_ratio_mean', 0.0)):.3f}"
            )
            lines.append("")

    report_path = output_dir / "analysis.txt"
    ensure_directory(report_path.parent)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_logging()

    root = Path(args.output_dir)
    ensure_directory(root)

    datasets = [str(d).lower().strip() for d in args.datasets]

    seed_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        for seed in args.seeds:
            set_seed(int(seed))

            seed_root = root / dataset / f"seed_{seed}"
            ensure_directory(seed_root)

            # Base config is shared across the three variants.
            inferred_input_dim = 32 * 32 * 3 if dataset in {"cifar10", "cifar-10"} else 28 * 28
            base_config = replace(
                TrainingConfig(),
                dataset_name=dataset,
                input_dim=int(inferred_input_dim),
                difficulty_threshold=float(args.difficulty_threshold),
                seed=int(seed),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                learning_rate=float(args.learning_rate),
                train_samples=int(args.train_samples),
                val_samples=int(args.val_samples),
                test_samples=int(args.test_samples),
                dataset_root=str(args.dataset_root),
                complexity_lambda=float(args.complexity_lambda),
                complexity_growth_power=float(args.complexity_growth_power),
                growth_budget_neurons=int(args.growth_budget_neurons),
                dataloader_num_workers=int(args.num_workers),
            )

            train_loader, val_loader, test_loader = _build_loaders(base_config)

            runs = [
                ("Static", False, True),
                ("SANN", True, False),
                ("CA-SANN", True, True),
            ]

            metrics_for_plot: dict[str, Any] = {}

            for model_label, dynamic, context_gating in runs:
                run_dir = seed_root / model_label.replace("/", "_").replace(" ", "_")
                cfg = replace(base_config, output_dir=str(run_dir), context_gating=bool(context_gating))

                cached = _load_run_payload(run_dir) if args.reuse else None
                if cached is not None:
                    seed_rows.append(cached)
                    # Keep minimal metrics for the combined plot only if present.
                    # (Plot will be skipped if cached doesn't include series.)
                    metrics_for_plot[model_label] = cached.get("time_series")
                    continue

                gc.collect()
                t0 = time.perf_counter()
                result = run_experiment(
                    experiment_name=model_label.lower().replace("-", "_").replace(" ", "_"),
                    config=cfg,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    dynamic=bool(dynamic),
                )
                dt = time.perf_counter() - t0

                payload = _result_to_payload(
                    result,
                    model_label=model_label,
                    dataset=dataset,
                    seed=int(seed),
                    training_time_sec=float(dt),
                )
                _save_run_payload(run_dir, payload)
                seed_rows.append(payload)

                metrics_for_plot[model_label] = result.metrics

            # Per-seed combined plot (only if we actually have metrics objects)
            metrics_objects: dict[str, Any] = {}
            for label, value in metrics_for_plot.items():
                if value is None:
                    continue
                # If cached, value may be dict; skip plot.
                if isinstance(value, dict):
                    continue
                metrics_objects[label] = value

            if metrics_objects:
                plot_experiment_metrics(
                    output_dir=seed_root,
                    metrics_by_name=metrics_objects,
                    plot_filename="metrics_all_models.png",
                )

    # Write seed-level tables
    seed_rows_sorted = sorted(
        seed_rows,
        key=lambda r: (
            str(r["dataset"]),
            int(r["seed"]),
            MODEL_ORDER.index(str(r["model"])) if str(r["model"]) in MODEL_ORDER else 999,
        ),
    )
    write_json(root / "seed_results.json", {"rows": seed_rows_sorted})

    seed_table_rows: list[dict[str, Any]] = []
    for r in seed_rows_sorted:
        seed_table_rows.append(
            {
                "dataset": r.get("dataset"),
                "seed": r.get("seed"),
                "model": r.get("model"),
                "val_accuracy": r.get("val_accuracy"),
                "test_accuracy": r.get("test_accuracy"),
                "val_loss": r.get("val_loss"),
                "test_loss": r.get("test_loss"),
                "model_size": r.get("model_size"),
                "efficiency": r.get("efficiency"),
                "accuracy_per_100k_params": r.get("accuracy_per_100k_params"),
                "growth_events": r.get("growth_events"),
                "candidate_growth_events": r.get("candidate_growth_events"),
                "exploration_growth_events": r.get("exploration_growth_events"),
                "rejected_growth_events": r.get("rejected_growth_events"),
                "training_time_sec": r.get("training_time_sec"),
            }
        )
    if seed_table_rows:
        write_csv(root / "seed_results.csv", seed_table_rows, fieldnames=list(seed_table_rows[0].keys()))

    # Aggregate tables + plots
    aggregate_rows = _aggregate(seed_rows_sorted)
    write_json(root / "aggregate_results.json", {"rows": aggregate_rows})
    write_csv(root / "aggregate_results.csv", aggregate_rows, fieldnames=list(aggregate_rows[0].keys()) if aggregate_rows else ["dataset", "model", "n_seeds"])

    for dataset in sorted({str(r["dataset"]).lower().strip() for r in aggregate_rows}):
        _plot_dataset_summary(root / dataset / "aggregate_plots", dataset=dataset, aggregate_rows=aggregate_rows)

    # Automatic growth analysis + report
    growth_analysis = [_analyze_growth(seed_rows_sorted, dataset=d) for d in sorted({str(r["dataset"]).lower().strip() for r in seed_rows_sorted})]
    write_json(root / "growth_analysis.json", {"rows": growth_analysis})
    _write_report(root, aggregate_rows=aggregate_rows, growth_analysis=growth_analysis)

    # Console summary table
    headers = ["Dataset", "Model", "Test Acc", "Params", "Eff/100k", "Growth"]
    table_rows: list[list[str]] = []
    for r in aggregate_rows:
        table_rows.append(
            [
                str(r.get("dataset", "")),
                str(r.get("model", "")),
                f"{float(r.get('test_accuracy_mean', 0.0)):.4f}±{float(r.get('test_accuracy_std', 0.0)):.4f}",
                f"{float(r.get('model_size_mean', 0.0)):.0f}",
                f"{float(r.get('accuracy_per_100k_params_mean', 0.0)):.4f}",
                f"{float(r.get('growth_events_mean', 0.0)):.2f}",
            ]
        )
    print("\nAggregate Summary")
    print(format_table(headers, table_rows))
    print(f"\nSaved outputs under: {root}")


if __name__ == "__main__":
    main()
