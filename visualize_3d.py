from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _collect_seed_level_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_dir in sorted(root.glob("seed_*")):
        csv_path = seed_dir / "comparison_results.csv"
        for row in _read_csv_rows(csv_path):
            model_size = _safe_float(row.get("model_size"))
            test_accuracy = _safe_float(row.get("test_accuracy"))
            growth_events = _safe_float(row.get("growth_events"))
            if "accuracy_per_100k_params" in row:
                acc_per_100k = _safe_float(row.get("accuracy_per_100k_params"))
            else:
                acc_per_100k = test_accuracy / max(1.0, model_size / 100000.0)
            rows.append(
                {
                    "seed": seed_dir.name,
                    "model": str(row.get("model", "Unknown")),
                    "model_size": model_size,
                    "test_accuracy": test_accuracy,
                    "growth_events": growth_events,
                    "accuracy_per_100k_params": acc_per_100k,
                }
            )
    return rows


def _plot_metric_3d(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    z_key: str,
    title: str,
    x_label: str,
    y_label: str,
    z_label: str,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    model_colors = {
        "Static": "#1f77b4",
        "SANN": "#d62728",
        "CA-SANN": "#2ca02c",
    }

    for model in sorted(set(str(row["model"]) for row in rows)):
        subset = [row for row in rows if str(row["model"]) == model]
        xs = [_safe_float(row.get(x_key)) for row in subset]
        ys = [_safe_float(row.get(y_key)) for row in subset]
        zs = [_safe_float(row.get(z_key)) for row in subset]

        ax.scatter(
            xs,
            ys,
            zs,
            s=60,
            alpha=0.9,
            label=model,
            color=model_colors.get(model, "#9467bd"),
            edgecolors="white",
            linewidths=0.7,
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.legend(loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


@dataclass
class LayerSnapshot:
    layer_index: int
    layer_type: str
    width: int
    active_count: int
    active_flags: list[bool]
    importance: list[float]
    edge_strengths: list[list[float]] | None = None


@dataclass
class CheckpointSnapshot:
    path: Path
    epoch: int
    architecture_type: str
    layers: list[LayerSnapshot]
    output_width: int
    parameter_count: int


def _sorted_state_keys(state_dict: dict[str, Any], suffix: str) -> list[str]:
    def sort_key(key: str) -> int:
        match = re.search(r"hidden_layers\.(\d+)\.", key)
        return int(match.group(1)) if match else 0

    return sorted((key for key in state_dict if key.endswith(suffix)), key=sort_key)


def _infer_architecture_type(state_dict: dict[str, Any]) -> str:
    if any(".linear.weight" in key for key in state_dict):
        return "mlp"
    if any(".conv.weight" in key for key in state_dict):
        return "cnn"
    raise ValueError("Unsupported checkpoint format: could not find hidden layer weights")


def _tensor_to_float_list(values: torch.Tensor) -> list[float]:
    return [float(v) for v in values.detach().cpu().tolist()]


def load_checkpoint_snapshot(checkpoint_path: str | Path) -> CheckpointSnapshot:
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    architecture_type = _infer_architecture_type(state_dict)

    layers: list[LayerSnapshot] = []
    if architecture_type == "mlp":
        weight_keys = _sorted_state_keys(state_dict, ".linear.weight")
        for layer_index, weight_key in enumerate(weight_keys):
            weight = state_dict[weight_key].detach().cpu()
            mask = state_dict.get(weight_key.replace(".linear.weight", ".mask"))
            if mask is None:
                mask_tensor = torch.ones(weight.shape[0], dtype=torch.float32)
            else:
                mask_tensor = mask.detach().cpu().float()
            importance = torch.norm(weight, p=2, dim=1)
            edge_strengths = torch.abs(weight)
            layers.append(
                LayerSnapshot(
                    layer_index=layer_index,
                    layer_type="dense",
                    width=int(weight.shape[0]),
                    active_count=int((mask_tensor > 0).sum().item()),
                    active_flags=[bool(v) for v in (mask_tensor > 0).tolist()],
                    importance=_tensor_to_float_list(importance),
                    edge_strengths=edge_strengths.tolist(),
                )
            )
    else:
        weight_keys = _sorted_state_keys(state_dict, ".conv.weight")
        for layer_index, weight_key in enumerate(weight_keys):
            weight = state_dict[weight_key].detach().cpu()
            mask = state_dict.get(weight_key.replace(".conv.weight", ".mask"))
            if mask is None:
                mask_tensor = torch.ones(weight.shape[0], dtype=torch.float32)
            else:
                mask_tensor = mask.detach().cpu().float()
            flat_weight = weight.view(weight.shape[0], -1)
            importance = torch.norm(flat_weight, p=2, dim=1)
            edge_strengths: list[list[float]] | None = None
            if weight.shape[1] > 0:
                aggregated = torch.norm(weight, p=2, dim=(2, 3))
                edge_strengths = aggregated.tolist()
            layers.append(
                LayerSnapshot(
                    layer_index=layer_index,
                    layer_type="conv",
                    width=int(weight.shape[0]),
                    active_count=int((mask_tensor > 0).sum().item()),
                    active_flags=[bool(v) for v in (mask_tensor > 0).tolist()],
                    importance=_tensor_to_float_list(importance),
                    edge_strengths=edge_strengths,
                )
            )

    output_width = int(state_dict["output_layer.weight"].shape[0])
    parameter_count = int(sum(value.numel() for value in state_dict.values()))
    epoch = int(checkpoint.get("epoch", 0))
    return CheckpointSnapshot(
        path=path,
        epoch=epoch,
        architecture_type=architecture_type,
        layers=layers,
        output_width=output_width,
        parameter_count=parameter_count,
    )


def _sample_indices(width: int, max_units: int) -> list[int]:
    if width <= max_units:
        return list(range(width))
    if max_units <= 1:
        return [0]
    return sorted({round(i * (width - 1) / (max_units - 1)) for i in range(max_units)})


def _layer_positions(width: int, layer_x: float, max_units: int, *, z_level: float = 0.0) -> tuple[list[int], list[tuple[float, float, float]]]:
    sampled = _sample_indices(width, max_units)
    count = len(sampled)
    if count == 1:
        y_positions = [0.0]
    else:
        y_positions = [i - (count - 1) / 2 for i in range(count)]
    positions = [(layer_x, y, z_level) for y in y_positions]
    return sampled, positions


def _plot_network_architecture_3d(
    snapshot: CheckpointSnapshot,
    output_path: Path,
    *,
    max_units_per_layer: int = 48,
    max_edges: int = 500,
) -> None:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    node_positions: dict[tuple[int, int], tuple[float, float, float]] = {}
    sampled_indices_per_layer: dict[int, list[int]] = {}
    layer_colors = {
        "dense": "#1f77b4",
        "conv": "#ff7f0e",
        "output": "#2ca02c",
    }

    for layer in snapshot.layers:
        sampled_indices, positions = _layer_positions(layer.width, float(layer.layer_index), max_units_per_layer)
        sampled_indices_per_layer[layer.layer_index] = sampled_indices
        sampled_importance = [layer.importance[idx] for idx in sampled_indices]
        max_importance = max(sampled_importance) if sampled_importance else 1.0
        sizes = [70 + 180 * (value / max_importance if max_importance > 0 else 0.0) for value in sampled_importance]
        colors = ["#d62728" if not layer.active_flags[idx] else layer_colors[layer.layer_type] for idx in sampled_indices]

        xs = [pos[0] for pos in positions]
        ys = [pos[1] for pos in positions]
        zs = [pos[2] for pos in positions]
        ax.scatter(xs, ys, zs, s=sizes, c=colors, alpha=0.95, edgecolors="white", linewidths=0.6)

        for idx, pos in zip(sampled_indices, positions):
            node_positions[(layer.layer_index, idx)] = pos

        ax.text(float(layer.layer_index), max(ys) + 2.0 if ys else 0.0, 0.6, f"L{layer.layer_index}\n{layer.width} units", ha="center")

    output_layer_index = len(snapshot.layers)
    output_sampled, output_positions = _layer_positions(snapshot.output_width, float(output_layer_index), min(snapshot.output_width, 24), z_level=0.0)
    for idx, pos in zip(output_sampled, output_positions):
        node_positions[(output_layer_index, idx)] = pos
    ax.scatter(
        [pos[0] for pos in output_positions],
        [pos[1] for pos in output_positions],
        [pos[2] for pos in output_positions],
        s=110,
        c=layer_colors["output"],
        alpha=0.95,
        edgecolors="white",
        linewidths=0.6,
    )
    ax.text(float(output_layer_index), max(pos[1] for pos in output_positions) + 2.0 if output_positions else 0.0, 0.6, f"Output\n{snapshot.output_width}", ha="center")

    edge_candidates: list[tuple[float, tuple[float, float, float], tuple[float, float, float]]] = []

    if snapshot.layers:
        last_hidden = snapshot.layers[-1]
        last_hidden_indices = sampled_indices_per_layer[last_hidden.layer_index]
        output_weight = torch.load(snapshot.path, map_location="cpu", weights_only=False)["model_state_dict"]["output_layer.weight"].detach().cpu()
        for out_idx in output_sampled:
            end = node_positions.get((output_layer_index, out_idx))
            if end is None:
                continue
            for hidden_idx in last_hidden_indices:
                start = node_positions.get((last_hidden.layer_index, hidden_idx))
                if start is None:
                    continue
                strength = float(abs(output_weight[out_idx, hidden_idx].item()))
                edge_candidates.append((strength, start, end))

        for layer_idx in range(len(snapshot.layers) - 1):
            dst_layer = snapshot.layers[layer_idx + 1]
            src_indices = sampled_indices_per_layer[layer_idx]
            dst_indices = sampled_indices_per_layer[layer_idx + 1]
            for dst_idx in dst_indices:
                end = node_positions.get((layer_idx + 1, dst_idx))
                if end is None:
                    continue
                for src_idx in src_indices:
                    start = node_positions.get((layer_idx, src_idx))
                    if start is None:
                        continue
                    strength = float(dst_layer.edge_strengths[dst_idx][src_idx]) if dst_layer.edge_strengths is not None else 0.0
                    edge_candidates.append((strength, start, end))

    edge_candidates.sort(key=lambda item: item[0], reverse=True)
    top_edges = edge_candidates[:max_edges]
    max_strength = top_edges[0][0] if top_edges else 1.0
    for strength, start, end in top_edges:
        alpha = 0.12 + 0.48 * (strength / max_strength if max_strength > 0 else 0.0)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color="#7f7f7f",
            alpha=alpha,
            linewidth=0.8,
        )

    ax.set_title(
        f"3D Network View: {snapshot.path.name}\n"
        f"type={snapshot.architecture_type.upper()} | epoch={snapshot.epoch} | params={snapshot.parameter_count:,}"
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Sampled Unit Position")
    ax.set_zlabel("Depth")
    ax.view_init(elev=18, azim=-68)
    ax.set_zticks([])
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _extract_epoch_from_name(path: Path) -> int:
    match = re.search(r"epoch(\d+)", path.name)
    if match:
        return int(match.group(1))
    return 0


def _plot_growth_sequence_3d(checkpoint_dir: Path, output_path: Path) -> None:
    checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=_extract_epoch_from_name)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {checkpoint_dir}")

    snapshots = [load_checkpoint_snapshot(path) for path in checkpoints]
    max_layers = max(len(snapshot.layers) for snapshot in snapshots)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for layer_idx in range(max_layers):
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        active_zs: list[float] = []
        for snapshot in snapshots:
            if layer_idx >= len(snapshot.layers):
                continue
            layer = snapshot.layers[layer_idx]
            xs.append(float(snapshot.epoch))
            ys.append(float(layer_idx))
            zs.append(float(layer.width))
            active_zs.append(float(layer.active_count))

        if not xs:
            continue

        ax.plot(xs, ys, zs, linewidth=2.2, marker="o", label=f"Layer {layer_idx} total")
        ax.plot(xs, ys, active_zs, linewidth=1.6, linestyle="--", marker="x", alpha=0.8, label=f"Layer {layer_idx} active")

    total_params_x = [float(snapshot.epoch) for snapshot in snapshots]
    total_params_y = [float(max_layers + 0.4)] * len(snapshots)
    total_params_z = [float(snapshot.parameter_count) / 1000.0 for snapshot in snapshots]
    ax.plot(total_params_x, total_params_y, total_params_z, linewidth=2.6, color="#d62728", marker="s", label="Params (thousands)")

    ax.set_title(f"3D Growth Timeline: {checkpoint_dir.name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Layer Index")
    ax.set_zlabel("Units / Channels")
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3D visualizations for SANN experiment outputs and network checkpoints")
    parser.add_argument(
        "--mode",
        type=str,
        default="metrics",
        choices=["metrics", "network", "sequence"],
        help="metrics: experiment efficiency plots, network: 3D architecture from one checkpoint, sequence: 3D growth over checkpoints",
    )
    parser.add_argument("--input-root", type=str, default="runs/mnist_sann_research")
    parser.add_argument("--output-dir", type=str, default="runs/mnist_sann_research")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--max-units-per-layer", type=int, default=48)
    parser.add_argument("--max-edges", type=int, default=500)
    return parser.parse_args()


def _run_metrics_mode(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)

    rows = _collect_seed_level_rows(input_root)
    if not rows:
        raise SystemExit(f"No seed-level comparison_results.csv found under {input_root}")

    plot1 = output_dir / "efficiency_3d_size_acc_growth.png"
    _plot_metric_3d(
        rows,
        x_key="model_size",
        y_key="test_accuracy",
        z_key="growth_events",
        title="3D: Model Size vs Accuracy vs Growth Events",
        x_label="Model Parameters",
        y_label="Test Accuracy",
        z_label="Growth Events",
        output_path=plot1,
    )

    plot2 = output_dir / "efficiency_3d_size_acc_accperparam.png"
    _plot_metric_3d(
        rows,
        x_key="model_size",
        y_key="test_accuracy",
        z_key="accuracy_per_100k_params",
        title="3D: Model Size vs Accuracy vs Accuracy/100k Params",
        x_label="Model Parameters",
        y_label="Test Accuracy",
        z_label="Accuracy per 100k Params",
        output_path=plot2,
    )

    print(f"Saved: {plot1}")
    print(f"Saved: {plot2}")


def _run_network_mode(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required when --mode network")

    snapshot = load_checkpoint_snapshot(checkpoint)
    output_path = Path(args.output_dir) / f"{checkpoint.stem}_network_3d.png"
    _plot_network_architecture_3d(
        snapshot,
        output_path,
        max_units_per_layer=int(args.max_units_per_layer),
        max_edges=int(args.max_edges),
    )
    print(f"Saved: {output_path}")


def _run_sequence_mode(args: argparse.Namespace) -> None:
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(args.input_root) / "checkpoints"
    output_path = Path(args.output_dir) / f"{checkpoint_dir.name}_growth_sequence_3d.png"
    _plot_growth_sequence_3d(checkpoint_dir, output_path)
    print(f"Saved: {output_path}")


def main() -> None:
    args = parse_args()
    if args.mode == "metrics":
        _run_metrics_mode(args)
    elif args.mode == "network":
        _run_network_mode(args)
    else:
        _run_sequence_mode(args)


if __name__ == "__main__":
    main()
