from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_compare import run_comparison
from train import TrainingConfig
from utils import format_table, render_research_analysis, summarize_numeric, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-seed research-grade SANN vs static experiment")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["mnist", "fashion_mnist", "cifar10"],
        help="Datasets to benchmark: mnist | fashion_mnist | cifar10",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--hidden-dims", type=str, default="128,64")
    parser.add_argument("--difficulty-threshold", type=float, default=0.55)
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="runs/ca_sann_benchmark")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 23, 33, 43, 53])
    parser.add_argument("--train-label-noise", type=float, default=0.15)
    parser.add_argument("--complexity-lambda", type=float, default=1e-6)
    parser.add_argument("--complexity-growth-power", type=float, default=1.35)
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
    parser.add_argument("--force-growth-debug", action="store_true")
    parser.add_argument("--force-growth-interval", type=int, default=2)
    parser.add_argument("--force-growth-neurons", type=int, default=2)
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


def _collect_seed_rows(payload: dict[str, Any], seed: int, dataset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload["results"]:
        rows.append(
            {
                "seed": seed,
                "dataset": str(dataset),
                "model": row["model"],
                "val_accuracy": float(row["val_accuracy"]),
                "test_accuracy": float(row["test_accuracy"]),
                "val_loss": float(row["val_loss"]),
                "test_loss": float(row["test_loss"]),
                "model_size": float(row["model_size"]),
                "growth_events": float(row["growth_events"]),
                "candidate_growth_events": float(row.get("candidate_growth_events", 0.0)),
                "rejected_growth_events": float(row.get("rejected_growth_events", 0.0)),
                "training_time_sec": float(row["training_time_sec"]),
                "accuracy_per_100k_params": float(row.get("accuracy_per_100k_params", 0.0)),
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), str(row["model"])), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    metric_names = [
        "val_accuracy",
        "test_accuracy",
        "val_loss",
        "test_loss",
        "model_size",
        "growth_events",
        "candidate_growth_events",
        "rejected_growth_events",
        "training_time_sec",
        "accuracy_per_100k_params",
    ]

    for (dataset, model_name), model_rows in grouped.items():
        out: dict[str, Any] = {"dataset": dataset, "model": model_name, "n_seeds": len(model_rows)}
        for metric in metric_names:
            values = [float(row[metric]) for row in model_rows]
            metric_mean, metric_std = summarize_numeric(values)
            out[f"{metric}_mean"] = metric_mean
            out[f"{metric}_std"] = metric_std
        aggregate_rows.append(out)

    aggregate_rows.sort(key=lambda r: (str(r.get("dataset", "")), str(r.get("model", ""))))
    return aggregate_rows


def _load_existing_seed_payload(seed_output_dir: Path) -> dict[str, Any] | None:
    payload_path = seed_output_dir / "comparison_results.json"
    if not payload_path.exists():
        return None
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _run_seed_with_retries(base_config: TrainingConfig, initial_batch_size: int) -> dict[str, Any]:
    batch_candidates = [
        max(8, int(initial_batch_size)),
        max(8, int(initial_batch_size // 2)),
        max(8, int(initial_batch_size // 4)),
    ]
    deduped_candidates: list[int] = []
    for candidate in batch_candidates:
        if candidate not in deduped_candidates:
            deduped_candidates.append(candidate)

    last_error: Exception | None = None
    for batch_size in deduped_candidates:
        config = replace(base_config, batch_size=batch_size)
        try:
            print(f"Running seed={config.seed} with batch_size={batch_size}")
            return run_comparison(config)
        except RuntimeError as exc:
            error_text = str(exc).lower()
            if "not enough memory" not in error_text and "out of memory" not in error_text:
                raise
            last_error = exc
            gc.collect()
            print(
                f"OOM on seed={config.seed}, batch_size={batch_size}. "
                "Retrying with a smaller batch size..."
            )

    if last_error is not None:
        raise last_error
    raise RuntimeError("Seed run failed before execution started.")


def _plot_aggregate_accuracy_vs_size(
    output_dir: Path,
    *,
    datasets: list[str],
    aggregate_rows: list[dict[str, Any]],
) -> None:
    if not aggregate_rows:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, max(1, len(datasets)), figsize=(6.5 * max(1, len(datasets)), 5.5), dpi=150)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        rows = [r for r in aggregate_rows if str(r.get("dataset", "")).lower().strip() == dataset]
        if not rows:
            ax.set_axis_off()
            continue

        for row in rows:
            model = str(row.get("model", ""))
            x = float(row.get("model_size_mean", 0.0))
            y = float(row.get("test_accuracy_mean", 0.0))
            marker = "o" if model.lower() == "static" else "s"
            ax.scatter([x], [y], s=70, marker=marker, label=model)
            ax.annotate(model, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

        ax.set_title(f"{dataset}: Test Accuracy vs Model Size")
        ax.set_xlabel("Trainable Parameters (mean)")
        ax.set_ylabel("Test Accuracy (mean)")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_model_size.png")
    plt.close(fig)

    # Efficiency summary plot (accuracy per 100k params) for each dataset/model.
    fig2, axes2 = plt.subplots(1, max(1, len(datasets)), figsize=(6.5 * max(1, len(datasets)), 5.0), dpi=150)
    if len(datasets) == 1:
        axes2 = [axes2]

    for ax, dataset in zip(axes2, datasets):
        rows = [r for r in aggregate_rows if str(r.get("dataset", "")).lower().strip() == dataset]
        if not rows:
            ax.set_axis_off()
            continue
        labels: list[str] = []
        values: list[float] = []
        for row in rows:
            labels.append(str(row.get("model", "")))
            values.append(float(row.get("accuracy_per_100k_params_mean", 0.0)))
        ax.bar(labels, values, color=["#4C78A8" if l.lower() == "static" else "#F58518" for l in labels])
        ax.set_title(f"{dataset}: Efficiency (Acc / 100k params)")
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy per 100k parameters (mean)")
        ax.grid(True, axis="y", alpha=0.25)

    fig2.tight_layout()
    fig2.savefig(output_dir / "efficiency_per_100k_params.png")
    plt.close(fig2)


def main() -> None:
    args = parse_args()
    root_output_dir = Path(args.output_dir)

    all_seed_rows: list[dict[str, Any]] = []
    verdict_counts = {"better": 0, "similar": 0, "trade-off": 0, "worse": 0}
    verdict_counts_by_dataset: dict[str, dict[str, int]] = {}

    datasets = [str(d).lower().strip() for d in args.datasets]
    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    for dataset in datasets:
        dataset_verdict_counts = {"better": 0, "similar": 0, "trade-off": 0, "worse": 0}
        verdict_counts_by_dataset[dataset] = dataset_verdict_counts

        inferred_input_dim = 32 * 32 * 3 if dataset in {"cifar10", "cifar-10"} else 28 * 28

        for seed in args.seeds:
            seed_output_dir = root_output_dir / dataset / f"seed_{seed}"
            existing_payload = _load_existing_seed_payload(seed_output_dir)
            if existing_payload is not None:
                print(f"Reusing existing results for dataset={dataset}, seed={seed} from {seed_output_dir}")
                all_seed_rows.extend(_collect_seed_rows(existing_payload, seed, dataset))
                verdict_label_existing = str(existing_payload["verdict"]["label"])
                if verdict_label_existing in verdict_counts:
                    verdict_counts[verdict_label_existing] += 1
                if verdict_label_existing in dataset_verdict_counts:
                    dataset_verdict_counts[verdict_label_existing] += 1
                continue

            base_config = replace(
                TrainingConfig(),
                dataset_name=dataset,
                input_dim=int(inferred_input_dim),
                hidden_dims=hidden_dims,
                difficulty_threshold=float(args.difficulty_threshold),
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                train_samples=args.train_samples,
                val_samples=args.val_samples,
                test_samples=args.test_samples,
                output_dir=str(seed_output_dir),
                dataset_root=args.dataset_root,
                train_label_noise=args.train_label_noise,
                complexity_lambda=args.complexity_lambda,
                complexity_growth_power=args.complexity_growth_power,
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
                force_growth_debug=args.force_growth_debug,
                force_growth_interval_epochs=args.force_growth_interval,
                force_growth_neurons=args.force_growth_neurons,
                dataloader_num_workers=args.num_workers,
            )

            payload = _run_seed_with_retries(base_config, args.batch_size)
            all_seed_rows.extend(_collect_seed_rows(payload, seed, dataset))

            verdict_label = str(payload["verdict"]["label"])
            if verdict_label in verdict_counts:
                verdict_counts[verdict_label] += 1
            if verdict_label in dataset_verdict_counts:
                dataset_verdict_counts[verdict_label] += 1

    aggregate_rows = _aggregate(all_seed_rows)
    _plot_aggregate_accuracy_vs_size(root_output_dir, datasets=datasets, aggregate_rows=aggregate_rows)

    aggregate_payload = {
        "config": {
            "datasets": datasets,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_dims": list(hidden_dims),
            "difficulty_threshold": float(args.difficulty_threshold),
            "train_samples": args.train_samples,
            "val_samples": args.val_samples,
            "test_samples": args.test_samples,
            "dataset_root": args.dataset_root,
            "train_label_noise": args.train_label_noise,
            "complexity_lambda": args.complexity_lambda,
            "complexity_growth_power": args.complexity_growth_power,
            "growth_budget_neurons": args.growth_budget_neurons,
            "growth_min_val_acc_gain": args.growth_min_val_acc_gain,
            "growth_min_val_loss_drop": args.growth_min_val_loss_drop,
            "growth_accept_min_val_acc_gain": args.growth_accept_min_val_acc_gain,
            "growth_accept_min_efficiency_gain": args.growth_accept_min_efficiency_gain,
            "exploration_budget_events": args.exploration_budget_events,
            "exploration_phase_epochs": args.exploration_phase_epochs,
            "exploration_eval_delay_epochs": args.exploration_eval_delay_epochs,
            "architecture_eval_batches": args.architecture_eval_batches,
            "architecture_eval_train_batches": args.architecture_eval_train_batches,
            "enable_candidate_growth": False if args.disable_candidate_growth else True,
            "candidate_min_growth_priority": args.candidate_min_growth_priority,
            "candidate_min_underperformance_to_grow": args.candidate_min_underperformance,
            "candidate_max_layer_importance_to_grow": args.candidate_max_layer_importance,
            "candidate_weak_neuron_fraction_for_growth": args.candidate_weak_neuron_fraction,
            "candidate_growth_neurons": args.candidate_growth_neurons,
            "efficiency_drop_tolerance_early": args.efficiency_drop_tol_early,
            "efficiency_drop_tolerance_late": args.efficiency_drop_tol_late,
            "efficiency_decline_patience": args.efficiency_decline_patience,
            "seeds": args.seeds,
            "force_growth_debug": args.force_growth_debug,
        },
        "verdict_counts": verdict_counts,
        "verdict_counts_by_dataset": verdict_counts_by_dataset,
        "seed_level_results": all_seed_rows,
        "aggregate_results": aggregate_rows,
    }

    write_json(root_output_dir / "aggregate_results.json", aggregate_payload)
    write_csv(
        root_output_dir / "aggregate_results.csv",
        aggregate_rows,
        fieldnames=list(aggregate_rows[0].keys()) if aggregate_rows else ["model", "n_seeds"],
    )

    headers = [
        "Dataset",
        "Model",
        "Seeds",
        "Test Acc Mean+/-Std",
        "Test Loss Mean+/-Std",
        "Params Mean+/-Std",
        "Growth Mean+/-Std",
        "Candidate Mean+/-Std",
        "Rejected Mean+/-Std",
    ]
    table_rows: list[list[str]] = []
    for row in aggregate_rows:
        table_rows.append(
            [
                str(row.get("dataset", "")),
                str(row["model"]),
                str(row["n_seeds"]),
                f"{row['test_accuracy_mean']:.4f} +/- {row['test_accuracy_std']:.4f}",
                f"{row['test_loss_mean']:.6f} +/- {row['test_loss_std']:.6f}",
                f"{row['model_size_mean']:.0f} +/- {row['model_size_std']:.1f}",
                f"{row['growth_events_mean']:.2f} +/- {row['growth_events_std']:.2f}",
                f"{row['candidate_growth_events_mean']:.2f} +/- {row['candidate_growth_events_std']:.2f}",
                f"{row['rejected_growth_events_mean']:.2f} +/- {row['rejected_growth_events_std']:.2f}",
            ]
        )

    print("\nMulti-Dataset, Multi-Seed Aggregate Summary")
    print(format_table(headers, table_rows))
    print(
        "\nVerdict counts: "
        f"better={verdict_counts['better']} | "
        f"similar={verdict_counts['similar']} | "
        f"trade-off={verdict_counts['trade-off']} | "
        f"worse={verdict_counts['worse']}"
    )

    analysis_sections: list[str] = []
    analysis_sections.append("CA-SANN Multi-Dataset Research Report")
    analysis_sections.append("===================================")
    analysis_sections.append("")
    analysis_sections.append(
        "Overall verdict counts: "
        f"better={verdict_counts['better']} | "
        f"similar={verdict_counts['similar']} | "
        f"trade-off={verdict_counts['trade-off']} | "
        f"worse={verdict_counts['worse']}"
    )
    analysis_sections.append("")

    for dataset in datasets:
        dataset_rows = [row for row in aggregate_rows if str(row.get("dataset", "")).lower().strip() == dataset]
        analysis_sections.append(f"\nDataset: {dataset}")
        analysis_sections.append("-" * (9 + len(dataset)))
        analysis_sections.append(
            render_research_analysis(dataset_rows, verdict_counts_by_dataset.get(dataset, {}))
        )
        analysis_sections.append("")

    analysis_text = "\n".join(analysis_sections)
    (root_output_dir / "analysis.txt").parent.mkdir(parents=True, exist_ok=True)
    (root_output_dir / "analysis.txt").write_text(analysis_text, encoding="utf-8")
    print("\n" + analysis_text)
    print(f"\nSaved aggregate JSON: {root_output_dir / 'aggregate_results.json'}")
    print(f"Saved aggregate CSV: {root_output_dir / 'aggregate_results.csv'}")
    print(f"Saved analysis report: {root_output_dir / 'analysis.txt'}")


if __name__ == "__main__":
    main()
