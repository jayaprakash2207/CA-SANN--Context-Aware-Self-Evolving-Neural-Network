from __future__ import annotations

import json
import logging
import random
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyzer import AnalyzerConfig, ErrorAnalyzer
from controller import ArchitectureAction, ControllerConfig, GrowthConfig, GrowthController, GrowthOutcome, PruneConfig
from context_estimators import (
    CapacityEstimator,
    CapacityEstimatorConfig,
    DifficultyEstimator,
    DifficultyEstimatorConfig,
    prediction_diagnostics,
)
from meta_controller import MetaController, MetaControllerConfig
from model import DynamicCNN, DynamicCNNConfig, DynamicMLP, DynamicMLPConfig


@dataclass
class TrainingConfig:
    seed: int = 13
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 8e-4
    weight_decay: float = 1e-5
    complexity_lambda: float = 1e-6
    complexity_growth_power: float = 1.35
    train_samples: int = 10000
    val_samples: int = 2000
    test_samples: int = 5000
    input_dim: int = 28 * 28
    num_classes: int = 10
    hidden_dims: tuple[int, ...] = (128, 64)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    architecture_eval_batches: int = 10
    architecture_eval_train_batches: int = 8
    output_dir: str = "runs/mnist_sann_experiment"
    checkpoint_dirname: str = "checkpoints"
    plot_filename: str = "metrics.png"
    summary_filename: str = "summary.json"
    max_val_batches: int | None = None
    dataset_root: str = "data"
    dataset_name: str = "mnist"
    model_arch: str = "mlp"  # mlp | cnn (cifar10 defaults to cnn)
    download_mnist: bool = True
    train_label_noise: float = 0.15
    cnn_in_channels: int = 3
    cnn_channels: tuple[int, ...] = (32, 64, 128)
    cnn_kernel_size: int = 3
    cnn_use_batchnorm: bool = True
    debug_log_details: bool = True
    force_growth_debug: bool = False
    force_growth_interval_epochs: int = 2
    force_growth_neurons: int = 2
    growth_budget_neurons: int = 48
    growth_min_val_acc_gain: float = 0.0025
    growth_min_val_loss_drop: float = 0.004
    growth_accept_min_val_acc_gain: float = 0.0
    growth_accept_min_efficiency_gain: float = 0.0
    enable_candidate_growth: bool = True
    candidate_min_growth_priority: float = 0.45
    candidate_min_underperformance_to_grow: float = 0.05
    candidate_max_layer_importance_to_grow: float = 0.90
    candidate_weak_neuron_fraction_for_growth: float = 0.10
    candidate_growth_neurons: int = 2
    efficiency_drop_tolerance_early: float = 2.0e-7
    efficiency_drop_tolerance_late: float = 5.0e-8
    efficiency_decline_patience: int = 3
    exploration_budget_events: int = 2
    exploration_phase_epochs: int = 6
    exploration_eval_delay_epochs: int = 2
    underfit_delayed_eval_epochs: int = 2
    difficulty_threshold: float = 0.55
    # If true, growth is gated by capacity_status=='underfit' and difficulty_score>=difficulty_threshold.
    # If false, growth is allowed whenever the controller proposes it (classic SANN behavior).
    context_gating: bool = True
    max_entropy_val_batches: int = 8
    allow_candidate_growth_when_meta_noop: bool = True
    stagnation_window: int = 3
    stagnation_min_delta: float = 3e-3
    high_val_loss_threshold: float = 0.35
    high_val_loss_window: int = 3
    no_growth_force_after_epochs: int = 4
    fallback_growth_neurons: int = 2
    max_model_params: int = 500000
    grad_clip_norm: float = 1.0
    post_growth_lr_scale: float = 0.5
    post_growth_recovery_epochs: int = 2
    dataloader_num_workers: int = 0


@dataclass
class ExperimentMetrics:
    epoch: list[int]
    train_loss: list[float]
    val_loss: list[float]
    val_accuracy: list[float]
    model_size: list[int]
    growth_events: list[int]
    complexity_penalty: list[float]
    meta_action: list[str]
    meta_confidence: list[float]
    complexity_penalty_value: list[float]
    efficiency: list[float]
    peak_efficiency: list[float]
    candidate_growth_events: list[int]
    exploration_growth_events: list[int]
    accepted_growth_events: list[int]
    rejected_growth_events: list[int]
    growth_enabled: list[int]
    safe_growth_accepted: list[int]
    safe_growth_rejected_peak: list[int]
    exploration_growth_accepted: list[int]
    exploration_growth_rejected: list[int]
    capacity_status: list[str]
    difficulty_score: list[float]
    growth_allowed: list[int]


@dataclass
class ExperimentResult:
    name: str
    metrics: ExperimentMetrics
    final_val_loss: float
    final_val_accuracy: float
    final_model_size: int
    growth_event_count: int
    candidate_growth_event_count: int
    exploration_growth_event_count: int
    rejected_growth_event_count: int
    safe_growth_accepted_count: int
    safe_growth_rejected_peak_count: int
    exploration_growth_accepted_count: int
    exploration_growth_rejected_count: int
    final_efficiency: float
    final_checkpoint: str
    test_loss: float
    test_accuracy: float
    policy_summary: dict[str, Any] = field(default_factory=dict)


class IndexedTensorDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.x[idx], self.y[idx], torch.tensor(idx, dtype=torch.long)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IndexedImageDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, dataset: Dataset[tuple[Tensor, Tensor]]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        image, label = self.dataset[idx]
        return image, label, torch.tensor(idx, dtype=torch.long)


class NoisyLabelDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, dataset: Dataset[tuple[Tensor, Tensor]], noise_ratio: float, num_classes: int, seed: int) -> None:
        self.dataset = dataset
        self.noise_ratio = max(0.0, min(1.0, noise_ratio))
        self.num_classes = num_classes
        self.generator = torch.Generator().manual_seed(seed)
        noisy_count = int(len(dataset) * self.noise_ratio)
        noisy_indices = torch.randperm(len(dataset), generator=self.generator)[:noisy_count].tolist()
        self.noisy_label_map: dict[int, int] = {
            idx: int(torch.randint(low=0, high=self.num_classes, size=(1,), generator=self.generator).item())
            for idx in noisy_indices
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        image, label = self.dataset[idx]
        return image, torch.tensor(self.noisy_label_map.get(idx, int(label)), dtype=torch.long)


def make_datasets(
    config: TrainingConfig,
) -> tuple[Dataset[tuple[Tensor, Tensor]], Dataset[tuple[Tensor, Tensor]], Dataset[tuple[Tensor, Tensor]]]:
    name = str(config.dataset_name).lower().strip()

    if name in {"mnist", "fashion_mnist", "fashion-mnist"}:
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        is_fashion = name in {"fashion_mnist", "fashion-mnist"}
        ds_cls = datasets.FashionMNIST if is_fashion else datasets.MNIST
        train_full = ds_cls(
            root=config.dataset_root,
            train=True,
            download=config.download_mnist,
            transform=transform,
        )
        test_dataset = ds_cls(
            root=config.dataset_root,
            train=False,
            download=config.download_mnist,
            transform=transform,
        )
    elif name in {"cifar10", "cifar-10"}:
        # Standard CIFAR-10 normalization.
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        train_full = datasets.CIFAR10(
            root=config.dataset_root,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.CIFAR10(
            root=config.dataset_root,
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset_name: {config.dataset_name}")

    train_indices = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(config.seed))[: config.train_samples + config.val_samples]
    train_subset = torch.utils.data.Subset(train_full, train_indices.tolist())
    train_subset, val_subset = random_split(
        train_subset,
        lengths=[config.train_samples, config.val_samples],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_subset = NoisyLabelDataset(train_subset, noise_ratio=config.train_label_noise, num_classes=config.num_classes, seed=config.seed)

    test_subset = torch.utils.data.Subset(test_dataset, list(range(min(config.test_samples, len(test_dataset)))))
    return train_subset, val_subset, test_subset


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def model_size_penalty(model: nn.Module, config: TrainingConfig, reference_params: int) -> float:
    params = float(count_parameters(model))
    ref = float(max(1, reference_params))
    scaled = params * ((params / ref) ** max(0.0, config.complexity_growth_power))
    return config.complexity_lambda * scaled


def efficiency_score(accuracy: float, model_size: int) -> float:
    return float(accuracy) / float(max(1, model_size))


def adaptive_efficiency_drop_tolerance(config: TrainingConfig, epoch: int) -> float:
    progress = min(1.0, max(0.0, float(epoch) / float(max(1, config.epochs))))
    return (
        config.efficiency_drop_tolerance_early
        + (config.efficiency_drop_tolerance_late - config.efficiency_drop_tolerance_early) * progress
    )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    *,
    output_dir: Path,
    checkpoint_dirname: str,
    experiment_name: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: ExperimentMetrics,
    tag: str,
) -> Path:
    checkpoint_dir = output_dir / checkpoint_dirname
    ensure_directory(checkpoint_dir)
    checkpoint_path = checkpoint_dir / f"{experiment_name}_epoch{epoch:03d}_{tag}.pt"
    torch.save(
        {
            "epoch": epoch,
            "experiment_name": experiment_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_size": count_parameters(model),
            "metrics": {
                "epoch": metrics.epoch,
                "train_loss": metrics.train_loss,
                "val_loss": metrics.val_loss,
                "val_accuracy": metrics.val_accuracy,
                "model_size": metrics.model_size,
                "growth_events": metrics.growth_events,
                "complexity_penalty": metrics.complexity_penalty,
                "meta_action": metrics.meta_action,
                "meta_confidence": metrics.meta_confidence,
                "efficiency": metrics.efficiency,
                "peak_efficiency": metrics.peak_efficiency,
                "candidate_growth_events": metrics.candidate_growth_events,
                "exploration_growth_events": metrics.exploration_growth_events,
                "accepted_growth_events": metrics.accepted_growth_events,
                "rejected_growth_events": metrics.rejected_growth_events,
                "growth_enabled": metrics.growth_enabled,
                "safe_growth_accepted": metrics.safe_growth_accepted,
                "safe_growth_rejected_peak": metrics.safe_growth_rejected_peak,
                "exploration_growth_accepted": metrics.exploration_growth_accepted,
                "exploration_growth_rejected": metrics.exploration_growth_rejected,
                "capacity_status": metrics.capacity_status,
                "difficulty_score": metrics.difficulty_score,
                "growth_allowed": metrics.growth_allowed,
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def evaluate(
    model: Any,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (x, y, _) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            if bool(getattr(model, "expects_flatten_input", False)):
                x = x.view(x.size(0), -1)
            logits = model(x)
            per_sample_loss = criterion(logits, y)
            total_loss += float(per_sample_loss.sum().item())
            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == y).sum().item())
            total_samples += y.shape[0]

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def plot_experiment_metrics(output_dir: Path, metrics_by_name: dict[str, ExperimentMetrics], plot_filename: str) -> Path:
    ensure_directory(output_dir)
    plot_path = output_dir / plot_filename

    def _capacity_to_num(value: str) -> int:
        v = str(value).lower().strip()
        if v == "underfit":
            return 0
        if v == "optimal":
            return 1
        if v == "overfit":
            return 2
        return -1

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), dpi=150)
    for name, metrics in metrics_by_name.items():
        axes[0, 0].plot(metrics.epoch, metrics.train_loss, label=f"{name} train")
        axes[0, 0].plot(metrics.epoch, metrics.val_loss, linestyle="--", label=f"{name} val")
        axes[0, 1].plot(metrics.epoch, metrics.val_accuracy, label=f"{name} val")
        axes[1, 0].plot(metrics.epoch, metrics.model_size, label=name)
        axes[1, 1].plot(metrics.epoch, metrics.efficiency, label=f"{name} efficiency")
        axes[1, 1].plot(metrics.epoch, metrics.peak_efficiency, linestyle=":", label=f"{name} peak efficiency")
        axes[1, 1].step(metrics.epoch, metrics.exploration_growth_events, where="post", linestyle="-.", label=f"{name} exploration growth")
        axes[1, 1].step(metrics.epoch, metrics.accepted_growth_events, where="post", linestyle="--", label=f"{name} accepted growth")
        axes[1, 1].step(metrics.epoch, metrics.rejected_growth_events, where="post", linestyle=":", label=f"{name} rejected growth")

        if getattr(metrics, "difficulty_score", None):
            axes[2, 0].plot(metrics.epoch, metrics.difficulty_score, label=f"{name} difficulty")
        if getattr(metrics, "growth_allowed", None):
            axes[2, 1].step(metrics.epoch, metrics.growth_allowed, where="post", label=f"{name} growth_allowed")
        if getattr(metrics, "capacity_status", None):
            capacity_nums = [_capacity_to_num(v) for v in metrics.capacity_status]
            axes[2, 1].plot(metrics.epoch, capacity_nums, linestyle="--", alpha=0.8, label=f"{name} capacity_status")

    axes[0, 0].set_title("Loss vs Epoch")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].set_title("Validation Accuracy vs Epoch")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].set_title("Model Size vs Epoch")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Trainable Parameters")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].set_title("Efficiency and Growth Decisions")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score / Cumulative Events")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend()

    axes[2, 0].set_title("Difficulty Score vs Epoch")
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("Difficulty (0-1)")
    axes[2, 0].set_ylim(-0.05, 1.05)
    axes[2, 0].grid(True, alpha=0.25)
    axes[2, 0].legend()

    axes[2, 1].set_title("Growth Gating and Capacity Status")
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylabel("growth_allowed / capacity_status")
    axes[2, 1].set_yticks([-1, 0, 1, 2])
    axes[2, 1].set_yticklabels(["unknown", "underfit", "optimal", "overfit"])
    axes[2, 1].grid(True, alpha=0.25)
    axes[2, 1].legend()

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def serialize_summary(output_dir: Path, summary_filename: str, payload: dict) -> Path:
    ensure_directory(output_dir)
    summary_path = output_dir / summary_filename
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def make_model(config: TrainingConfig) -> Any:
    dataset_name = str(getattr(config, "dataset_name", "mnist")).lower().strip()
    model_arch = str(getattr(config, "model_arch", "mlp")).lower().strip()

    if dataset_name in {"cifar10", "cifar-10"}:
        # Default to CNN for CIFAR-10 unless explicitly overridden.
        if model_arch == "mlp":
            model_arch = "cnn"

    if model_arch == "cnn":
        return DynamicCNN(
            DynamicCNNConfig(
                in_channels=int(config.cnn_in_channels),
                channels=list(config.cnn_channels),
                num_classes=int(config.num_classes),
                activation="relu",
                kernel_size=int(config.cnn_kernel_size),
                use_batchnorm=bool(config.cnn_use_batchnorm),
            )
        )

    if dataset_name in {"cifar10", "cifar-10"}:
        inferred_input_dim = 32 * 32 * 3
    else:
        inferred_input_dim = 28 * 28
    return DynamicMLP(
        DynamicMLPConfig(
            input_dim=int(inferred_input_dim),
            hidden_dims=list(config.hidden_dims),
            output_dim=int(config.num_classes),
            activation="relu",
        )
    )


def _format_layer_scalar_stats(values: dict[int, tuple[float, float]]) -> str:
    if not values:
        return "none"
    parts: list[str] = []
    for layer_idx in sorted(values.keys()):
        mean, variance = values[layer_idx]
        parts.append(f"L{layer_idx}(mean={mean:.6f},var={variance:.6f})")
    return " | ".join(parts)


def _format_grad_stats(values: dict[int, tuple[float, float]]) -> str:
    if not values:
        return "none"
    parts: list[str] = []
    for layer_idx in sorted(values.keys()):
        mean_grad, max_grad = values[layer_idx]
        parts.append(f"L{layer_idx}(mean={mean_grad:.6f},max={max_grad:.6f})")
    return " | ".join(parts)


def _model_summary(model: Any) -> str:
    summary_fn = getattr(model, "architecture_summary", None)
    if callable(summary_fn):
        return str(summary_fn())
    hidden = ",".join(str(dim) for dim in getattr(model, "hidden_dims", []))
    out_layer = getattr(model, "output_layer", None)
    if out_layer is not None:
        return (
            f"hidden_dims=[{hidden}] | output_layer(in={out_layer.in_features},out={out_layer.out_features}) "
            f"| params={count_parameters(model)}"
        )
    return f"hidden_dims=[{hidden}] | params={count_parameters(model)}"


def run_experiment(
    *,
    experiment_name: str,
    config: TrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    dynamic: bool,
) -> ExperimentResult:
    device = torch.device(config.device)
    model = make_model(config).to(device)
    analyzer = ErrorAnalyzer(
        AnalyzerConfig(
            loss_plateau_window=config.stagnation_window,
            loss_plateau_min_delta=config.stagnation_min_delta,
            high_val_loss_threshold=config.high_val_loss_threshold,
            high_val_loss_window=config.high_val_loss_window,
        )
    )
    analyzer.on_architecture_change(model)
    controller = (
        GrowthController(
            ControllerConfig(
                growth=GrowthConfig(
                    warmup_epochs=1,
                    decision_interval=1,
                    min_growth_loss=0.10,
                    gradient_vanishing_threshold=5e-4,
                    activation_dead_ratio_threshold=0.25,
                    hard_sample_trigger=16,
                    high_val_loss_trigger=True,
                    force_growth_after_epochs=config.no_growth_force_after_epochs,
                    force_growth_neurons=config.fallback_growth_neurons,
                    max_model_params=config.max_model_params,
                    base_growth=4,
                    max_growth_per_event=10,
                    max_growth_per_layer=8,
                    cooldown_epochs=3,
                    max_total_growth_ratio=0.25,
                    min_layer_growth=2,
                    min_growth_priority=0.75,
                    require_growth_stagnation=True,
                    min_hard_sample_pressure=0.15,
                    total_growth_budget_neurons=config.growth_budget_neurons,
                    allow_candidate_growth=config.enable_candidate_growth,
                    candidate_min_growth_priority=config.candidate_min_growth_priority,
                    candidate_min_underperformance_to_grow=config.candidate_min_underperformance_to_grow,
                    candidate_max_layer_importance_to_grow=config.candidate_max_layer_importance_to_grow,
                    candidate_weak_neuron_fraction_for_growth=config.candidate_weak_neuron_fraction_for_growth,
                    candidate_growth_neurons=config.candidate_growth_neurons,
                ),
                prune=PruneConfig(enabled=True, prune_interval=3, max_neurons_per_event=4),
            )
        )
        if dynamic
        else None
    )
    optimizer = build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss(reduction="none")

    output_dir = Path(config.output_dir)
    ensure_directory(output_dir)

    metrics = ExperimentMetrics(
        epoch=[],
        train_loss=[],
        val_loss=[],
        val_accuracy=[],
        model_size=[],
        growth_events=[],
        complexity_penalty=[],
        meta_action=[],
        meta_confidence=[],
        complexity_penalty_value=[],
        efficiency=[],
        peak_efficiency=[],
        candidate_growth_events=[],
        exploration_growth_events=[],
        accepted_growth_events=[],
        rejected_growth_events=[],
        growth_enabled=[],
        safe_growth_accepted=[],
        safe_growth_rejected_peak=[],
        exploration_growth_accepted=[],
        exploration_growth_rejected=[],
        capacity_status=[],
        difficulty_score=[],
        growth_allowed=[],
    )
    growth_event_count = 0
    candidate_growth_event_count = 0
    exploration_growth_event_count = 0
    total_growth_added = 0
    rejected_growth_event_count = 0
    growth_recovery_remaining = 0
    growth_recovery_total = max(0, config.post_growth_recovery_epochs)
    freeze_grad_masks: dict[str, torch.Tensor] = {}
    reference_params = count_parameters(model)
    best_efficiency_so_far = 0.0
    efficiency_decline_streak = 0
    no_growth_phase = False
    pending_exploration_evals: list[dict[str, Any]] = []
    safe_growth_accepted_count = 0
    safe_growth_rejected_peak_count = 0
    exploration_growth_accepted_count = 0
    exploration_growth_rejected_count = 0
    capacity_estimator = CapacityEstimator(CapacityEstimatorConfig())
    difficulty_estimator = DifficultyEstimator(DifficultyEstimatorConfig(hard_sample_trigger=16))
    last_capacity_status = "optimal"
    last_difficulty_score = 0.0
    growth_allowed = True

    def _build_post_growth_freeze_masks(
        model_ref: Any,
        growth_specs: list[tuple[int, int]],
    ) -> dict[str, torch.Tensor]:
        masks: dict[str, torch.Tensor] = {}
        for name, param in model_ref.named_parameters():
            masks[name] = torch.ones_like(param.detach(), device="cpu")

        for layer_idx, old_out_features in growth_specs:
            layer_weight_name = f"hidden_layers.{layer_idx}.linear.weight"
            if layer_weight_name in masks:
                masks[layer_weight_name][:old_out_features, :] = 0.0

            layer_bias_name = f"hidden_layers.{layer_idx}.linear.bias"
            if layer_bias_name in masks:
                masks[layer_bias_name][:old_out_features] = 0.0

            if layer_idx < len(model_ref.hidden_layers) - 1:
                next_weight_name = f"hidden_layers.{layer_idx + 1}.linear.weight"
                if next_weight_name in masks:
                    masks[next_weight_name][:, :old_out_features] = 0.0
            else:
                out_weight_name = "output_layer.weight"
                if out_weight_name in masks:
                    masks[out_weight_name][:, :old_out_features] = 0.0

        return masks
    meta_controller = MetaController(
        MetaControllerConfig(
            warmup_epochs=1,
            decision_interval=1,
            min_confidence=0.25,
            growth_confidence_threshold=0.40,
            prune_confidence_threshold=0.35,
            min_loss_for_growth=0.15,
            cooldown_epochs=3,
        )
    )

    logger = logging.getLogger("sann")
    logger.info("Starting %s experiment | dynamic=%s | initial_params=%d", experiment_name, dynamic, count_parameters(model))
    logger.info("%s | MODEL_SUMMARY | %s", experiment_name, _model_summary(model))
    if dynamic and config.force_growth_debug:
        logger.info(
            "%s | FORCE_GROWTH_DEBUG=ON | interval_epochs=%d | neurons_per_force=%d",
            experiment_name,
            config.force_growth_interval_epochs,
            config.force_growth_neurons,
        )

    def _record_efficiency_state(epoch_ref: int, val_acc_ref: float) -> None:
        nonlocal best_efficiency_so_far, efficiency_decline_streak, no_growth_phase

        current_efficiency = efficiency_score(val_acc_ref, count_parameters(model))
        tolerance = adaptive_efficiency_drop_tolerance(config, epoch_ref)
        previous_efficiency = metrics.efficiency[-1] if metrics.efficiency else None

        if current_efficiency > best_efficiency_so_far:
            best_efficiency_so_far = current_efficiency
            logger.info(
                "%s | PEAK_EFFICIENCY_UPDATE | epoch=%03d | peak_efficiency=%.10f | tolerance=%.10f",
                experiment_name,
                epoch_ref,
                best_efficiency_so_far,
                tolerance,
            )

        if previous_efficiency is not None and current_efficiency < (previous_efficiency - tolerance):
            efficiency_decline_streak += 1
        else:
            efficiency_decline_streak = 0

        if (
            dynamic
            and not no_growth_phase
            and config.efficiency_decline_patience > 0
            and efficiency_decline_streak >= config.efficiency_decline_patience
        ):
            no_growth_phase = True
            logger.info(
                "%s | NO_GROWTH_PHASE_ENTERED | epoch=%03d | decline_streak=%d | current_efficiency=%.10f | peak_efficiency=%.10f | tolerance=%.10f",
                experiment_name,
                epoch_ref,
                efficiency_decline_streak,
                current_efficiency,
                best_efficiency_so_far,
                tolerance,
            )

        metrics.efficiency.append(current_efficiency)
        metrics.peak_efficiency.append(best_efficiency_so_far)
        metrics.candidate_growth_events.append(candidate_growth_event_count)
        metrics.exploration_growth_events.append(exploration_growth_event_count)
        metrics.accepted_growth_events.append(growth_event_count)
        metrics.rejected_growth_events.append(rejected_growth_event_count)
        metrics.growth_enabled.append(0 if no_growth_phase else 1)
        metrics.safe_growth_accepted.append(safe_growth_accepted_count)
        metrics.safe_growth_rejected_peak.append(safe_growth_rejected_peak_count)
        metrics.exploration_growth_accepted.append(exploration_growth_accepted_count)
        metrics.exploration_growth_rejected.append(exploration_growth_rejected_count)
        metrics.capacity_status.append(str(last_capacity_status))
        metrics.difficulty_score.append(float(last_difficulty_score))
        metrics.growth_allowed.append(1 if growth_allowed else 0)

    def _is_exploration_phase(epoch_ref: int) -> bool:
        return epoch_ref <= max(1, config.exploration_phase_epochs)

    def _resolve_pending_exploration(epoch_ref: int, val_loss_ref: float, val_acc_ref: float) -> tuple[float, float]:
        nonlocal model, optimizer, growth_event_count, rejected_growth_event_count, no_growth_phase
        nonlocal exploration_growth_accepted_count, exploration_growth_rejected_count
        nonlocal safe_growth_accepted_count
        if not pending_exploration_evals:
            return val_loss_ref, val_acc_ref

        current_efficiency = efficiency_score(val_acc_ref, count_parameters(model))
        remaining: list[dict[str, Any]] = []
        for item in pending_exploration_evals:
            if epoch_ref < int(item["decision_epoch"]):
                remaining.append(item)
                continue

            mode = str(item.get("mode", "exploration")).lower().strip()

            delay_tolerance = adaptive_efficiency_drop_tolerance(config, epoch_ref)
            baseline_eff = float(item["baseline_efficiency"])
            baseline_acc = float(item["baseline_accuracy"])
            origin_pre_eff = float(item.get("origin_pre_efficiency", baseline_eff))
            origin_pre_acc = float(item.get("origin_pre_accuracy", baseline_acc))
            if mode == "underfit_delayed":
                # Underfit delayed acceptance:
                # keep if accuracy improves vs baseline OR efficiency recovers vs baseline.
                keep_growth = (val_acc_ref > baseline_acc) or (current_efficiency >= baseline_eff)
                logger.info(
                    "%s | UNDERFIT_DELAYED_REEVAL | mark=%s | epoch=%03d | origin_epoch=%03d | current_acc=%.4f | baseline_acc=%.4f | current_eff=%.10f | baseline_eff=%.10f | decision=%s",
                    experiment_name,
                    "delayed_keep" if keep_growth else "delayed_reject",
                    epoch_ref,
                    int(item["origin_epoch"]),
                    val_acc_ref,
                    baseline_acc,
                    current_efficiency,
                    baseline_eff,
                    "keep" if keep_growth else "revert",
                )
            else:
                safe_keep = (
                    current_efficiency > baseline_eff
                    or (current_efficiency >= (baseline_eff - delay_tolerance) and val_acc_ref > baseline_acc)
                )
                peak_keep = current_efficiency >= (best_efficiency_so_far - delay_tolerance)
                keep_growth = safe_keep and peak_keep
                logger.info(
                    "%s | EXPLORATION_REEVAL | epoch=%03d | origin_epoch=%03d | current_eff=%.10f | baseline_eff=%.10f | current_acc=%.4f | baseline_acc=%.4f | tolerance=%.10f | safe_keep=%s | peak_keep=%s | decision=%s",
                    experiment_name,
                    epoch_ref,
                    int(item["origin_epoch"]),
                    current_efficiency,
                    baseline_eff,
                    val_acc_ref,
                    baseline_acc,
                    delay_tolerance,
                    safe_keep,
                    peak_keep,
                    "accepted" if keep_growth else "rejected",
                )

            layer_indices = [int(x) for x in item.get("layer_indices", [])]
            if controller is not None and layer_indices:
                val_acc_gain = float(val_acc_ref - origin_pre_acc)
                eff_delta = float(current_efficiency - origin_pre_eff)
                for layer_idx in layer_indices:
                    controller.record_growth_outcome(
                        GrowthOutcome(
                            layer_idx=layer_idx,
                            accepted=bool(keep_growth),
                            exploration=(mode != "underfit_delayed"),
                            val_acc_gain=val_acc_gain,
                            efficiency_delta=eff_delta,
                            epoch=epoch_ref,
                        )
                    )

            if not keep_growth:
                rejected_growth_event_count += int(item["event_count"])
                if mode == "underfit_delayed":
                    # Underfit delayed reject should not force a global no-growth phase.
                    pass
                else:
                    exploration_growth_rejected_count += int(item["event_count"])
                    no_growth_phase = True
                snapshot_model = item.get("snapshot_model")
                snapshot_optimizer_state = item.get("snapshot_optimizer_state")
                if snapshot_model is not None:
                    model = deepcopy(snapshot_model)
                    optimizer = build_optimizer(model, config)
                    if snapshot_optimizer_state is not None:
                        optimizer.load_state_dict(deepcopy(snapshot_optimizer_state))
                    analyzer.on_architecture_change(model)
                    val_loss_ref, val_acc_ref = evaluate(
                        model,
                        val_loader,
                        criterion,
                        device,
                        max_batches=config.max_val_batches,
                    )
                    current_efficiency = efficiency_score(val_acc_ref, count_parameters(model))
                if mode == "underfit_delayed":
                    logger.info(
                        "%s | UNDERFIT_DELAYED_RESULT | mark=delayed_reject | epoch=%03d | origin_epoch=%03d | reverted=true | val_loss=%.6f | val_acc=%.4f | efficiency=%.10f",
                        experiment_name,
                        epoch_ref,
                        int(item["origin_epoch"]),
                        val_loss_ref,
                        val_acc_ref,
                        current_efficiency,
                    )
                else:
                    logger.info(
                        "%s | EXPLORATION_REJECTED | epoch=%03d | origin_epoch=%03d | reverted=true | val_loss=%.6f | val_acc=%.4f | efficiency=%.10f | entering_no_growth_phase=true",
                        experiment_name,
                        epoch_ref,
                        int(item["origin_epoch"]),
                        val_loss_ref,
                        val_acc_ref,
                        current_efficiency,
                    )
            else:
                growth_event_count += int(item["event_count"])
                if mode == "underfit_delayed":
                    safe_growth_accepted_count += int(item["event_count"])
                    logger.info(
                        "%s | UNDERFIT_DELAYED_RESULT | mark=delayed_keep | epoch=%03d | origin_epoch=%03d | event_count=%d",
                        experiment_name,
                        epoch_ref,
                        int(item["origin_epoch"]),
                        int(item["event_count"]),
                    )
                else:
                    exploration_growth_accepted_count += int(item["event_count"])
                    logger.info(
                        "%s | EXPLORATION_ACCEPTED | epoch=%03d | origin_epoch=%03d | event_count=%d",
                        experiment_name,
                        epoch_ref,
                        int(item["origin_epoch"]),
                        int(item["event_count"]),
                    )

        pending_exploration_evals[:] = remaining
        return val_loss_ref, val_acc_ref

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_samples = 0
        if growth_recovery_remaining > 0:
            elapsed = growth_recovery_total - growth_recovery_remaining
            progress = (elapsed + 1) / float(max(1, growth_recovery_total))
            lr_scale = config.post_growth_lr_scale + (1.0 - config.post_growth_lr_scale) * progress
            scaled_lr = config.learning_rate * lr_scale
            for group in optimizer.param_groups:
                group["lr"] = scaled_lr
            logger.info(
                "%s | POST_GROWTH_LR | epoch=%03d | lr=%.6f | recovery_progress=%.3f",
                experiment_name,
                epoch,
                scaled_lr,
                progress,
            )
        else:
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate
        grad_sums: dict[int, float] = {idx: 0.0 for idx in range(len(model.hidden_layers))}
        grad_max: dict[int, float] = {idx: 0.0 for idx in range(len(model.hidden_layers))}
        grad_counts: dict[int, int] = {idx: 0 for idx in range(len(model.hidden_layers))}
        activation_sum: dict[int, float] = {idx: 0.0 for idx in range(len(model.hidden_layers))}
        activation_sq_sum: dict[int, float] = {idx: 0.0 for idx in range(len(model.hidden_layers))}
        activation_count: dict[int, int] = {idx: 0 for idx in range(len(model.hidden_layers))}

        for x, y, sample_indices in train_loader:
            x = x.to(device)
            y = y.to(device)
            sample_indices = sample_indices.to(device)
            if bool(getattr(model, "expects_flatten_input", False)):
                x = x.view(x.size(0), -1)

            optimizer.zero_grad(set_to_none=True)
            logits, hidden = model(x, return_hidden=True)
            per_sample_loss = criterion(logits, y)
            task_loss = per_sample_loss.mean()
            complexity_penalty_value = model_size_penalty(model, config, reference_params)
            loss = task_loss + complexity_penalty_value

            loss.backward()

            if growth_recovery_remaining > 0 and freeze_grad_masks:
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    mask = freeze_grad_masks.get(name)
                    if mask is None or mask.shape != param.grad.shape:
                        continue
                    param.grad.mul_(mask.to(param.grad.device))

            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            analyzer.update_gradients(model)

            for layer_idx in range(len(model.hidden_layers)):
                grad_key = f"hidden_layers.{layer_idx}.linear.weight"
                grad_norm = analyzer.gradient_norm_last.get(grad_key)
                if grad_norm is None:
                    continue
                grad_sums[layer_idx] += grad_norm
                grad_counts[layer_idx] += 1
                if grad_norm > grad_max[layer_idx]:
                    grad_max[layer_idx] = grad_norm

            for layer_idx, layer_act in enumerate(hidden):
                act = layer_act.detach()
                activation_sum[layer_idx] += float(act.sum().item())
                activation_sq_sum[layer_idx] += float((act * act).sum().item())
                activation_count[layer_idx] += int(act.numel())

            optimizer.step()

            analyzer.update_batch(sample_indices, per_sample_loss.detach(), hidden)

            epoch_loss_sum += float(task_loss.item()) * y.shape[0]
            epoch_samples += y.shape[0]

        train_loss = epoch_loss_sum / max(1, epoch_samples)
        analyzer.record_epoch_loss(train_loss)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, max_batches=config.max_val_batches)
        analyzer.record_val_loss(val_loss)
        val_loss, val_acc = _resolve_pending_exploration(epoch, val_loss, val_acc)
        current_size = count_parameters(model)

        metrics.epoch.append(epoch)
        metrics.train_loss.append(train_loss)
        metrics.val_loss.append(val_loss)
        metrics.val_accuracy.append(val_acc)
        metrics.model_size.append(current_size)
        metrics.growth_events.append(growth_event_count)
        metrics.complexity_penalty.append(model_size_penalty(model, config, reference_params))
        metrics.complexity_penalty_value.append(model_size_penalty(model, config, reference_params))

        meta_decision = meta_controller.decide(model, analyzer, epoch)
        metrics.meta_action.append(meta_decision.action)
        metrics.meta_confidence.append(meta_decision.confidence)

        logger.info(
            "%s | Epoch=%03d | train_loss=%.6f | val_loss=%.6f | val_acc=%.4f | params=%d | growth_events=%d | meta_action=%s | meta_conf=%.3f",
            experiment_name,
            epoch,
            train_loss,
            val_loss,
            val_acc,
            current_size,
            growth_event_count,
            meta_decision.action,
            meta_decision.confidence,
        )

        # Context-aware gating: estimate capacity and difficulty using current epoch signals.
        report_now = dict(analyzer.report(model))
        diag = prediction_diagnostics(
            model,
            val_loader,
            device,
            num_classes=config.num_classes,
            max_batches=config.max_entropy_val_batches,
        )
        entropy_norm = float(diag.get("entropy_norm", 0.0))
        val_error_rate = float(diag.get("error_rate", 0.0))
        val_error_entropy = float(diag.get("error_class_entropy_norm", 0.0))
        report_now["val_error_rate"] = val_error_rate
        report_now["val_error_class_entropy_norm"] = val_error_entropy
        last_capacity_status = capacity_estimator.estimate(
            analyzer=analyzer,
            train_loss=train_loss,
            val_loss=val_loss,
            report=report_now,
        )
        last_difficulty_score = difficulty_estimator.estimate(
            analyzer=analyzer,
            report=report_now,
            val_entropy_norm=entropy_norm,
            val_error_rate=val_error_rate,
            val_error_class_entropy_norm=val_error_entropy,
        )
        context_gating_enabled = bool(getattr(config, "context_gating", True))
        difficulty_threshold = float(getattr(config, "difficulty_threshold", 0.0))
        difficulty_score = float(last_difficulty_score)

        if not context_gating_enabled:
            growth_allowed = True
            growth_gate_reason = "context_gating=off (SANN)"
        else:
            if last_capacity_status == "underfit":
                growth_allowed = True
                growth_gate_reason = "capacity_status=underfit => allow_growth (ignore difficulty threshold)"
            elif last_capacity_status == "optimal":
                growth_allowed = difficulty_score > difficulty_threshold
                growth_gate_reason = "capacity_status=optimal => require difficulty_score > threshold"
            elif last_capacity_status == "overfit":
                growth_allowed = False
                growth_gate_reason = "capacity_status=overfit => block_growth"
            else:
                growth_allowed = difficulty_score > difficulty_threshold
                growth_gate_reason = "capacity_status=unknown => require difficulty_score > threshold"

        logger.info(
            "%s | GROWTH_GATE | epoch=%03d | context_gating=%s | capacity_status=%s | difficulty_score=%.3f | threshold=%.3f | growth_allowed=%s | reason=%s",
            experiment_name,
            epoch,
            context_gating_enabled,
            last_capacity_status,
            difficulty_score,
            difficulty_threshold,
            growth_allowed,
            growth_gate_reason,
        )

        if config.debug_log_details:
            grad_epoch_stats: dict[int, tuple[float, float]] = {}
            for layer_idx in range(len(model.hidden_layers)):
                if grad_counts[layer_idx] == 0:
                    continue
                grad_epoch_stats[layer_idx] = (
                    grad_sums[layer_idx] / float(grad_counts[layer_idx]),
                    grad_max[layer_idx],
                )

            activation_epoch_stats: dict[int, tuple[float, float]] = {}
            for layer_idx in range(len(model.hidden_layers)):
                count = activation_count[layer_idx]
                if count == 0:
                    continue
                mean = activation_sum[layer_idx] / float(count)
                second_moment = activation_sq_sum[layer_idx] / float(count)
                variance = max(0.0, second_moment - mean * mean)
                activation_epoch_stats[layer_idx] = (mean, variance)

            logger.info("%s | DEBUG_GRAD_NORMS | %s", experiment_name, _format_grad_stats(grad_epoch_stats))
            logger.info("%s | DEBUG_ACTIVATIONS | %s", experiment_name, _format_layer_scalar_stats(activation_epoch_stats))

        if not dynamic or controller is None:
            _record_efficiency_state(epoch, val_acc)
            continue

        # If an underfit-delayed growth decision is pending, suppress further architecture changes
        # to keep rollback semantics well-defined.
        if any(str(item.get("mode", "")).lower().strip() == "underfit_delayed" for item in pending_exploration_evals):
            logger.info(
                "%s | ARCH_SUPPRESSED | epoch=%03d | reason=pending_underfit_delayed_eval",
                experiment_name,
                epoch,
            )
            _record_efficiency_state(epoch, val_acc)
            continue

        actions: list[ArchitectureAction] = []
        actions = controller.decide(model, analyzer, epoch)

        if bool(getattr(config, "context_gating", True)) and (not growth_allowed):
            blocked = sum(1 for action in actions if action.kind == "grow" and action.n_new > 0)
            if blocked > 0:
                logger.info(
                    "%s | GROWTH_BLOCKED_BY_CONTEXT | epoch=%03d | blocked_growth_actions=%d | capacity_status=%s | difficulty_score=%.3f | threshold=%.3f",
                    experiment_name,
                    epoch,
                    blocked,
                    last_capacity_status,
                    float(last_difficulty_score),
                    float(config.difficulty_threshold),
                )
            actions = [action for action in actions if action.kind != "grow"]

        if no_growth_phase:
            growth_actions_removed = sum(1 for action in actions if action.kind == "grow" and action.n_new > 0)
            if growth_actions_removed > 0:
                logger.info(
                    "%s | NO_GROWTH_PHASE_ACTIVE | epoch=%03d | suppressed_growth_actions=%d | peak_efficiency=%.10f",
                    experiment_name,
                    epoch,
                    growth_actions_removed,
                    best_efficiency_so_far,
                )
            actions = [action for action in actions if action.kind != "grow"]

        if meta_decision.action == "noop":
            if config.allow_candidate_growth_when_meta_noop:
                actions = [a for a in actions if a.kind == "grow" and a.n_new > 0 and bool(getattr(a, "candidate", False))]
            else:
                actions = []

        if meta_decision.action in {"grow", "prune"}:
            actions = [
                action
                for action in actions
                if action.kind == meta_decision.action and action.layer_idx == meta_decision.target_layer
            ]
            if meta_decision.action == "grow":
                for action in actions:
                    scaled = max(1, int(round(action.n_new * meta_decision.grow_multiplier)))
                    action.n_new = scaled
                    action.reason = f"{action.reason}, meta={meta_decision.rationale}, grow_multiplier={meta_decision.grow_multiplier:.3f}"
            elif meta_decision.action == "prune":
                for action in actions:
                    target_fraction = meta_decision.prune_fraction
                    if action.neurons:
                        selected_count = max(1, int(round(len(action.neurons) * target_fraction)))
                        action.neurons = action.neurons[:selected_count]
                        action.reason = f"{action.reason}, meta={meta_decision.rationale}, prune_fraction={target_fraction:.3f}"

        budget_remaining = max(0, config.growth_budget_neurons - total_growth_added)
        if budget_remaining <= 0:
            if any(action.kind == "grow" and action.n_new > 0 for action in actions):
                logger.info(
                    "%s | GROWTH_BUDGET_EXHAUSTED | total_added=%d | budget=%d",
                    experiment_name,
                    total_growth_added,
                    config.growth_budget_neurons,
                )
            actions = [action for action in actions if action.kind != "grow"]
        else:
            for action in actions:
                if action.kind != "grow" or action.n_new <= 0:
                    continue
                original_n = action.n_new
                action.n_new = min(action.n_new, budget_remaining)
                budget_remaining -= action.n_new
                if action.n_new != original_n:
                    action.reason = f"{action.reason}, budget_capped_from={original_n}_to={action.n_new}"

        should_force_growth = (
            not no_growth_phase
            and
            config.force_growth_debug
            and config.force_growth_interval_epochs > 0
            and epoch % config.force_growth_interval_epochs == 0
        )
        if should_force_growth:
            existing_growth = any(action.kind == "grow" and action.n_new > 0 for action in actions)
            if not existing_growth:
                forced_layer = (epoch // config.force_growth_interval_epochs - 1) % max(1, len(model.hidden_layers))
                forced_neurons = max(1, config.force_growth_neurons)
                actions.append(
                    ArchitectureAction(
                        kind="grow",
                        layer_idx=forced_layer,
                        n_new=forced_neurons,
                        reason=(
                            f"forced_debug_growth interval={config.force_growth_interval_epochs}, "
                            f"epoch={epoch}, forced_neurons={forced_neurons}"
                        ),
                    )
                )
                logger.info(
                    "%s | FORCE_GROWTH_TRIGGER | epoch=%03d | layer=%d | add_neurons=%d",
                    experiment_name,
                    epoch,
                    forced_layer,
                    forced_neurons,
                )

        if not actions:
            _record_efficiency_state(epoch, val_acc)
            continue

        if config.debug_log_details:
            layer_desc = ", ".join(f"{a.kind}@L{a.layer_idx}" for a in actions)
            logger.info("%s | ARCH_TRIGGER | epoch=%03d | actions=%s | meta=%s(%.3f)", experiment_name, epoch, layer_desc, meta_decision.action, meta_decision.confidence)

        pre_mod_loss = val_loss
        pre_mod_acc = val_acc
        architecture_changed = False
        growth_applied = False
        growth_specs: list[tuple[int, int]] = []
        growth_records: list[tuple[int, int]] = []
        growth_neurons_added = 0
        params_before_mod = count_parameters(model)
        backup_model: DynamicMLP | None = None
        backup_optimizer_state: dict | None = None

        candidate_growths_in_epoch = sum(
            1
            for action in actions
            if action.kind == "grow" and action.n_new > 0 and bool(getattr(action, "candidate", False))
        )
        if candidate_growths_in_epoch > 0:
            candidate_growth_event_count += candidate_growths_in_epoch

        if any(action.kind == "grow" and action.n_new > 0 for action in actions):
            backup_model = deepcopy(model)
            backup_optimizer_state = deepcopy(optimizer.state_dict())

        eval_val_batches = config.architecture_eval_batches
        if eval_val_batches is not None and eval_val_batches <= 0:
            eval_val_batches = None
        if any(action.kind == "grow" and action.n_new > 0 for action in actions):
            pre_mod_loss, pre_mod_acc = evaluate(
                model,
                val_loader,
                criterion,
                device,
                max_batches=eval_val_batches,
            )

        for action in actions:
            if action.kind == "grow" and action.n_new > 0:
                neurons_before = model.hidden_layers[action.layer_idx].out_features
                output_in_before = model.output_layer.in_features
                if bool(getattr(action, "candidate", False)):
                    logger.info(
                        "%s | GROWTH_CANDIDATE | epoch=%03d | layer=%d | add_neurons=%d | reason=%s",
                        experiment_name,
                        epoch,
                        action.layer_idx,
                        action.n_new,
                        action.reason,
                    )
                logger.info(
                    "%s | ARCH_EVENT=GROW | layer=%d | before=%d | add_neurons=%d | reason=%s",
                    experiment_name,
                    action.layer_idx,
                    neurons_before,
                    action.n_new,
                    action.reason,
                )
                model.grow_layer(action.layer_idx, action.n_new)
                neurons_after = model.hidden_layers[action.layer_idx].out_features
                growth_specs.append((action.layer_idx, neurons_before))
                logger.info(
                    "%s | ARCH_EVENT=GROW_RESULT | layer=%d | neurons_before=%d | neurons_after=%d",
                    experiment_name,
                    action.layer_idx,
                    neurons_before,
                    neurons_after,
                )
                output_in_after = model.output_layer.in_features
                if output_in_after != output_in_before:
                    logger.info(
                        "%s | ARCH_EVENT=OUTPUT_LAYER_CHANGE | in_features_before=%d | in_features_after=%d",
                        experiment_name,
                        output_in_before,
                        output_in_after,
                    )
                if neurons_after <= neurons_before:
                    logger.error(
                        "%s | ARCH_EVENT=GROW_FAILED | layer=%d | neurons_before=%d | neurons_after=%d",
                        experiment_name,
                        action.layer_idx,
                        neurons_before,
                        neurons_after,
                    )
                growth_records.append((action.layer_idx, int(action.n_new)))
                growth_neurons_added += int(action.n_new)
                architecture_changed = True
                growth_applied = True
                logger.info("%s | MODEL_SUMMARY_POST_GROW | %s", experiment_name, _model_summary(model))

            if action.kind == "prune" and action.neurons:
                layer_mask = model.hidden_layers[action.layer_idx].mask
                active_before = int((layer_mask > 0).sum().item())
                logger.info(
                    "%s | ARCH_EVENT=PRUNE | layer=%d | requested_remove=%d | active_before=%d | reason=%s",
                    experiment_name,
                    action.layer_idx,
                    len(action.neurons),
                    active_before,
                    action.reason,
                )
                model.prune_layer_neurons(action.layer_idx, action.neurons)
                active_after = int((model.hidden_layers[action.layer_idx].mask > 0).sum().item())
                removed_effective = max(0, active_before - active_after)
                logger.info(
                    "%s | ARCH_EVENT=PRUNE_RESULT | layer=%d | active_after=%d | removed_effective=%d",
                    experiment_name,
                    action.layer_idx,
                    active_after,
                    removed_effective,
                )
                architecture_changed = True

        if architecture_changed:
            params_after_mod = count_parameters(model)
            if growth_applied:
                assert params_after_mod > params_before_mod, (
                    f"Growth event did not increase model size: before={params_before_mod}, after={params_after_mod}"
                )
            optimizer = build_optimizer(model, config)
            growth_recovery_remaining = max(growth_recovery_remaining, config.post_growth_recovery_epochs)
            growth_recovery_total = max(growth_recovery_total, config.post_growth_recovery_epochs)
            if growth_specs:
                freeze_grad_masks = _build_post_growth_freeze_masks(model, growth_specs)
                logger.info(
                    "%s | ARCH_EVENT=FREEZE_WINDOW | epochs=%d | impacted_layers=%s",
                    experiment_name,
                    growth_recovery_remaining,
                    ",".join(str(layer_idx) for layer_idx, _ in growth_specs),
                )
            analyzer.on_architecture_change(model)

            if growth_applied and config.architecture_eval_train_batches > 0:
                model.train()
                steps = 0
                for x, y, _ in train_loader:
                    x = x.to(device)
                    y = y.to(device)
                    if bool(getattr(model, "expects_flatten_input", False)):
                        x = x.view(x.size(0), -1)

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    per_sample_loss = criterion(logits, y)
                    task_loss = per_sample_loss.mean()
                    complexity_penalty_value = model_size_penalty(model, config, reference_params)
                    loss = task_loss + complexity_penalty_value
                    loss.backward()

                    if growth_recovery_remaining > 0 and freeze_grad_masks:
                        for name, param in model.named_parameters():
                            if param.grad is None:
                                continue
                            mask = freeze_grad_masks.get(name)
                            if mask is None or mask.shape != param.grad.shape:
                                continue
                            param.grad.mul_(mask.to(param.grad.device))

                    if config.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                    optimizer.step()

                    steps += 1
                    if steps >= config.architecture_eval_train_batches:
                        break

                logger.info(
                    "%s | ARCH_EVAL_TRAIN | epoch=%03d | batches=%d",
                    experiment_name,
                    epoch,
                    steps,
                )

            post_mod_loss, post_mod_acc = evaluate(
                model,
                val_loader,
                criterion,
                device,
                max_batches=eval_val_batches,
            )
            post_mod_params = count_parameters(model)
            logger.info(
                "%s | ARCH_EVENT=RESULT | eval_val_batches=%s | val_loss_before=%.6f | val_loss_after=%.6f | val_acc_before=%.4f | val_acc_after=%.4f | params=%d | checkpoint=%s",
                experiment_name,
                "full" if eval_val_batches is None else str(eval_val_batches),
                pre_mod_loss,
                post_mod_loss,
                pre_mod_acc,
                post_mod_acc,
                post_mod_params,
                "pending",
            )

            if growth_applied:
                acc_gain = post_mod_acc - pre_mod_acc
                param_cost = max(1, post_mod_params - params_before_mod)
                pre_eff = efficiency_score(pre_mod_acc, params_before_mod)
                post_eff = efficiency_score(post_mod_acc, post_mod_params)
                eff_delta = post_eff - pre_eff
                acc_gain_per_param = acc_gain / float(param_cost)
                efficiency_tolerance = adaptive_efficiency_drop_tolerance(config, epoch)
                efficiency_floor = best_efficiency_so_far - efficiency_tolerance
                beneficial_signal = (
                    acc_gain > float(config.growth_accept_min_val_acc_gain)
                    or eff_delta > float(config.growth_accept_min_efficiency_gain)
                )
                within_peak_tolerance = post_eff >= efficiency_floor
                safe_growth = (
                    eff_delta > float(config.growth_accept_min_efficiency_gain)
                    or (
                        eff_delta >= -efficiency_tolerance
                        and acc_gain > float(config.growth_accept_min_val_acc_gain)
                    )
                )
                exploration_budget_left = max(0, config.exploration_budget_events - exploration_growth_event_count)
                policy_allows_exploration = True
                if controller is not None and growth_records:
                    scores: list[float] = []
                    trials: list[int] = []
                    for layer_idx, _ in growth_records:
                        stats = controller.policy.layer.get(int(layer_idx))
                        if stats is None:
                            trials.append(0)
                        else:
                            trials.append(int(stats.trials))
                            scores.append(float(stats.score))
                    avg_score = (sum(scores) / float(len(scores))) if scores else 0.0
                    min_trials = min(trials) if trials else 0
                    policy_allows_exploration = (min_trials <= 1) or (avg_score >= -0.75)
                exploration_eligible = (
                    _is_exploration_phase(epoch)
                    and exploration_budget_left >= len(growth_records)
                    and acc_gain > float(config.growth_accept_min_val_acc_gain)
                    and policy_allows_exploration
                )
                exploration_growth = (not safe_growth) and beneficial_signal and exploration_eligible
                safe_acc_improved = acc_gain > 0.0
                safe_peak_ok = within_peak_tolerance
                growth_accepted = safe_growth and safe_acc_improved and safe_peak_ok
                safe_reject_reason = ""
                if safe_growth and not safe_acc_improved:
                    safe_reject_reason = "reject_safe_no_acc_improvement"
                elif safe_growth and not safe_peak_ok:
                    safe_reject_reason = "reject_safe_peak_efficiency_floor"
                elif not safe_growth and not exploration_growth:
                    safe_reject_reason = "reject_not_safe_or_exploration"
                elif not safe_growth and exploration_growth:
                    safe_reject_reason = "exploration_path"
                elif safe_growth and safe_acc_improved and safe_peak_ok:
                    safe_reject_reason = "accept_safe_peak_aware"

                logger.info(
                    "%s | GROWTH_EVAL | epoch=%03d | acc_before=%.4f | acc_after=%.4f | acc_gain=%+.6f | eff_before=%.10f | eff_after=%.10f | eff_delta=%+.10f | peak_eff=%.10f | tolerance=%.10f | efficiency_floor=%.10f | within_peak_tolerance=%s | beneficial_signal=%s | safe_growth=%s | safe_acc_improved=%s | safe_reject_reason=%s | exploration_growth=%s | exploration_budget_left=%d | param_cost=%d | acc_gain_per_param=%+.10f | decision=%s",
                    experiment_name,
                    epoch,
                    pre_mod_acc,
                    post_mod_acc,
                    acc_gain,
                    pre_eff,
                    post_eff,
                    eff_delta,
                    best_efficiency_so_far,
                    efficiency_tolerance,
                    efficiency_floor,
                    within_peak_tolerance,
                    beneficial_signal,
                    safe_growth,
                    safe_acc_improved,
                    safe_reject_reason,
                    exploration_growth,
                    exploration_budget_left,
                    param_cost,
                    acc_gain_per_param,
                    "safe growth" if growth_accepted else ("exploration growth" if exploration_growth else "rejected growth"),
                )

                if exploration_growth:
                    exploration_growth_event_count += len(growth_records)
                    total_growth_added += growth_neurons_added
                    layer_indices = sorted({int(layer_idx) for layer_idx, _ in growth_records})
                    pending_exploration_evals.append(
                        {
                            "origin_epoch": epoch,
                            "decision_epoch": epoch + max(1, config.exploration_eval_delay_epochs),
                            "baseline_efficiency": post_eff,
                            "baseline_accuracy": post_mod_acc,
                            "origin_pre_efficiency": pre_eff,
                            "origin_pre_accuracy": pre_mod_acc,
                            "event_count": len(growth_records),
                            "layer_indices": layer_indices,
                            "snapshot_model": deepcopy(backup_model) if backup_model is not None else None,
                            "snapshot_optimizer_state": deepcopy(backup_optimizer_state) if backup_optimizer_state is not None else None,
                        }
                    )
                    logger.info(
                        "%s | ARCH_EVENT=EXPLORATION_GROWTH | epoch=%03d | events=%d | decision_epoch=%03d | exploration_used=%d/%d",
                        experiment_name,
                        epoch,
                        len(growth_records),
                        epoch + max(1, config.exploration_eval_delay_epochs),
                        exploration_growth_event_count,
                        config.exploration_budget_events,
                    )
                    for layer_idx, n_new in growth_records:
                        controller.record_growth(layer_idx, n_new, epoch)
                    val_loss = post_mod_loss
                    val_acc = post_mod_acc
                    metrics.val_loss[-1] = val_loss
                    metrics.val_accuracy[-1] = val_acc
                    metrics.model_size[-1] = count_parameters(model)
                    checkpoint_path = save_checkpoint(
                        output_dir=output_dir,
                        checkpoint_dirname=config.checkpoint_dirname,
                        experiment_name=experiment_name,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        metrics=metrics,
                        tag="arch_change",
                    )
                    logger.info("%s | ARCH_EVENT=CHECKPOINT | checkpoint=%s", experiment_name, checkpoint_path.name)
                elif (
                    bool(getattr(config, "context_gating", True))
                    and (str(last_capacity_status).lower().strip() == "underfit")
                    and (not growth_accepted)
                    and (not exploration_growth)
                    and backup_model is not None
                ):
                    # Underfit early growth: keep temporarily and decide after a delay.
                    delay_epochs = int(getattr(config, "underfit_delayed_eval_epochs", 2))
                    delay_epochs = max(2, min(3, delay_epochs))
                    layer_indices = sorted({int(layer_idx) for layer_idx, _ in growth_records})
                    pending_exploration_evals.append(
                        {
                            "mode": "underfit_delayed",
                            "origin_epoch": epoch,
                            "decision_epoch": epoch + delay_epochs,
                            "baseline_efficiency": pre_eff,
                            "baseline_accuracy": pre_mod_acc,
                            "origin_pre_efficiency": pre_eff,
                            "origin_pre_accuracy": pre_mod_acc,
                            "event_count": len(growth_records),
                            "layer_indices": layer_indices,
                            "snapshot_model": deepcopy(backup_model),
                            "snapshot_optimizer_state": deepcopy(backup_optimizer_state) if backup_optimizer_state is not None else None,
                        }
                    )
                    logger.info(
                        "%s | UNDERFIT_DELAYED_SCHEDULE | mark=early_growth_underfit | epoch=%03d | events=%d | decision_epoch=%03d | baseline_acc=%.4f | baseline_eff=%.10f | post_acc=%.4f | post_eff=%.10f",
                        experiment_name,
                        epoch,
                        len(growth_records),
                        epoch + delay_epochs,
                        pre_mod_acc,
                        pre_eff,
                        post_mod_acc,
                        post_eff,
                    )

                    # Keep the architecture change for now and checkpoint it.
                    val_loss = post_mod_loss
                    val_acc = post_mod_acc
                    metrics.val_loss[-1] = val_loss
                    metrics.val_accuracy[-1] = val_acc
                    metrics.model_size[-1] = count_parameters(model)

                    checkpoint_path = save_checkpoint(
                        output_dir=output_dir,
                        checkpoint_dirname=config.checkpoint_dirname,
                        experiment_name=experiment_name,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        metrics=metrics,
                        tag="arch_change_underfit_early",
                    )
                    logger.info("%s | ARCH_EVENT=CHECKPOINT | checkpoint=%s", experiment_name, checkpoint_path.name)
                elif not growth_accepted and backup_model is not None:
                    logger.info(
                        "%s | ARCH_EVENT=GROWTH_REJECTED | rejected_events=%d | efficiency_delta=%+.10f | peak_efficiency=%.10f | tolerance=%.10f",
                        experiment_name,
                        len(growth_records),
                        eff_delta,
                        best_efficiency_so_far,
                        efficiency_tolerance,
                    )
                    if controller is not None:
                        for layer_idx, _ in growth_records:
                            controller.record_growth_outcome(
                                GrowthOutcome(
                                    layer_idx=int(layer_idx),
                                    accepted=False,
                                    exploration=False,
                                    val_acc_gain=float(acc_gain),
                                    efficiency_delta=float(eff_delta),
                                    epoch=epoch,
                                )
                            )
                    logger.info(
                        "%s | ARCH_EVENT=ROLLBACK | reason=growth_not_beneficial | acc_gain=%+.6f | eff_delta=%+.10f",
                        experiment_name,
                        acc_gain,
                        eff_delta,
                    )
                    model = backup_model
                    optimizer = build_optimizer(model, config)
                    if backup_optimizer_state is not None:
                        optimizer.load_state_dict(backup_optimizer_state)
                    analyzer.on_architecture_change(model)
                    growth_recovery_remaining = 0
                    growth_recovery_total = max(0, config.post_growth_recovery_epochs)
                    freeze_grad_masks = {}
                    rejected_growth_event_count += len(growth_records)
                else:
                    logger.info(
                        "%s | ARCH_EVENT=SAFE_GROWTH | accepted_events=%d | efficiency_delta=%+.10f",
                        experiment_name,
                        len(growth_records),
                        eff_delta,
                    )
                    for layer_idx, n_new in growth_records:
                        controller.record_growth(layer_idx, n_new, epoch)
                    for layer_idx, _ in growth_records:
                        controller.record_growth_outcome(
                            GrowthOutcome(
                                layer_idx=int(layer_idx),
                                accepted=True,
                                exploration=False,
                                val_acc_gain=float(acc_gain),
                                efficiency_delta=float(eff_delta),
                                epoch=epoch,
                            )
                        )
                    safe_growth_accepted_count += len(growth_records)
                    growth_event_count += len(growth_records)
                    total_growth_added += growth_neurons_added

                    val_loss = post_mod_loss
                    val_acc = post_mod_acc
                    metrics.val_loss[-1] = val_loss
                    metrics.val_accuracy[-1] = val_acc
                    metrics.model_size[-1] = count_parameters(model)

                    checkpoint_path = save_checkpoint(
                        output_dir=output_dir,
                        checkpoint_dirname=config.checkpoint_dirname,
                        experiment_name=experiment_name,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        metrics=metrics,
                        tag="arch_change",
                    )
                    logger.info("%s | ARCH_EVENT=CHECKPOINT | checkpoint=%s", experiment_name, checkpoint_path.name)

                if safe_growth and (not growth_accepted) and (not exploration_growth) and (not safe_peak_ok):
                    safe_growth_rejected_peak_count += len(growth_records)
            else:
                checkpoint_path = save_checkpoint(
                    output_dir=output_dir,
                    checkpoint_dirname=config.checkpoint_dirname,
                    experiment_name=experiment_name,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    metrics=metrics,
                    tag="arch_change",
                )
                logger.info("%s | ARCH_EVENT=CHECKPOINT | checkpoint=%s", experiment_name, checkpoint_path.name)

        if growth_recovery_remaining > 0:
            growth_recovery_remaining -= 1
            if growth_recovery_remaining == 0:
                freeze_grad_masks = {}
                logger.info("%s | ARCH_EVENT=FREEZE_WINDOW_END", experiment_name)

        _record_efficiency_state(epoch, val_acc)

    if pending_exploration_evals:
        final_val_loss, final_val_acc = _resolve_pending_exploration(
            config.epochs + max(1, config.exploration_eval_delay_epochs),
            metrics.val_loss[-1],
            metrics.val_accuracy[-1],
        )
        metrics.val_loss[-1] = final_val_loss
        metrics.val_accuracy[-1] = final_val_acc
        metrics.model_size[-1] = count_parameters(model)
        if metrics.accepted_growth_events:
            metrics.accepted_growth_events[-1] = growth_event_count
        if metrics.rejected_growth_events:
            metrics.rejected_growth_events[-1] = rejected_growth_event_count

    final_checkpoint_path = save_checkpoint(
        output_dir=output_dir,
        checkpoint_dirname=config.checkpoint_dirname,
        experiment_name=experiment_name,
        epoch=config.epochs,
        model=model,
        optimizer=optimizer,
        metrics=metrics,
        tag="final",
    )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device, max_batches=config.max_val_batches)

    policy_summary: dict[str, Any] = {}
    if controller is not None:
        layer_stats: dict[str, Any] = {}
        for layer_idx in sorted(controller.policy.layer.keys()):
            stats = controller.policy.layer[layer_idx]
            layer_stats[str(layer_idx)] = {
                "score": float(stats.score),
                "trials": int(stats.trials),
                "successes": int(stats.successes),
                "failures": int(stats.failures),
                "exploration_trials": int(stats.exploration_trials),
                "exploration_successes": int(stats.exploration_successes),
                "exploration_failures": int(stats.exploration_failures),
            }
        history_tail = controller.policy.history[-50:]
        policy_summary = {
            "config": {
                "enabled": bool(controller.policy.config.enabled),
                "lr": float(controller.policy.config.lr),
                "decay": float(controller.policy.config.decay),
                "bias_weight": float(controller.policy.config.bias_weight),
                "novelty_bonus": float(controller.policy.config.novelty_bonus),
            },
            "layer_stats": layer_stats,
            "history_tail": [
                {
                    "epoch": int(item.epoch),
                    "layer_idx": int(item.layer_idx),
                    "accepted": bool(item.accepted),
                    "exploration": bool(item.exploration),
                    "val_acc_gain": float(item.val_acc_gain),
                    "efficiency_delta": float(item.efficiency_delta),
                }
                for item in history_tail
            ],
        }

    return ExperimentResult(
        name=experiment_name,
        metrics=metrics,
        final_val_loss=metrics.val_loss[-1],
        final_val_accuracy=metrics.val_accuracy[-1],
        final_model_size=metrics.model_size[-1],
        growth_event_count=growth_event_count,
        candidate_growth_event_count=candidate_growth_event_count,
        exploration_growth_event_count=exploration_growth_event_count,
        rejected_growth_event_count=rejected_growth_event_count,
        safe_growth_accepted_count=safe_growth_accepted_count,
        safe_growth_rejected_peak_count=safe_growth_rejected_peak_count,
        exploration_growth_accepted_count=exploration_growth_accepted_count,
        exploration_growth_rejected_count=exploration_growth_rejected_count,
        policy_summary=policy_summary,
        final_efficiency=efficiency_score(metrics.val_accuracy[-1], metrics.model_size[-1]),
        final_checkpoint=str(final_checkpoint_path),
        test_loss=test_loss,
        test_accuracy=test_acc,
    )


def train(config: TrainingConfig) -> None:
    setup_logging()
    set_seed(config.seed)
    logger = logging.getLogger("sann")

    train_ds, val_ds, test_ds = make_datasets(config)

    train_loader = DataLoader(
        IndexedImageDataset(train_ds),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        IndexedImageDataset(val_ds),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        IndexedImageDataset(test_ds),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.device.startswith("cuda"),
    )

    device = torch.device(config.device)
    logger.info("Starting research-grade SANN experiment on device=%s", device)

    dynamic_result = run_experiment(
        experiment_name="dynamic",
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dynamic=True,
    )
    static_result = run_experiment(
        experiment_name="static",
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dynamic=False,
    )

    output_dir = Path(config.output_dir)
    ensure_directory(output_dir)
    plot_path = plot_experiment_metrics(
        output_dir=output_dir,
        metrics_by_name={"dynamic": dynamic_result.metrics, "static": static_result.metrics},
        plot_filename=config.plot_filename,
    )

    summary_payload = {
        "seed": config.seed,
        "complexity_lambda": config.complexity_lambda,
        "complexity_growth_power": config.complexity_growth_power,
        "growth_budget_neurons": config.growth_budget_neurons,
        "growth_min_val_acc_gain": config.growth_min_val_acc_gain,
        "growth_min_val_loss_drop": config.growth_min_val_loss_drop,
        "growth_accept_min_val_acc_gain": config.growth_accept_min_val_acc_gain,
        "growth_accept_min_efficiency_gain": config.growth_accept_min_efficiency_gain,
        "exploration_budget_events": config.exploration_budget_events,
        "exploration_phase_epochs": config.exploration_phase_epochs,
        "exploration_eval_delay_epochs": config.exploration_eval_delay_epochs,
        "enable_candidate_growth": config.enable_candidate_growth,
        "candidate_min_growth_priority": config.candidate_min_growth_priority,
        "candidate_min_underperformance_to_grow": config.candidate_min_underperformance_to_grow,
        "candidate_max_layer_importance_to_grow": config.candidate_max_layer_importance_to_grow,
        "candidate_weak_neuron_fraction_for_growth": config.candidate_weak_neuron_fraction_for_growth,
        "candidate_growth_neurons": config.candidate_growth_neurons,
        "va_sann": {
            "efficiency_drop_tolerance_early": config.efficiency_drop_tolerance_early,
            "efficiency_drop_tolerance_late": config.efficiency_drop_tolerance_late,
            "efficiency_decline_patience": config.efficiency_decline_patience,
        },
        "dynamic": {
            "final_val_loss": dynamic_result.final_val_loss,
            "final_val_accuracy": dynamic_result.final_val_accuracy,
            "final_model_size": dynamic_result.final_model_size,
            "growth_event_count": dynamic_result.growth_event_count,
            "candidate_growth_event_count": dynamic_result.candidate_growth_event_count,
            "exploration_growth_event_count": dynamic_result.exploration_growth_event_count,
            "rejected_growth_event_count": dynamic_result.rejected_growth_event_count,
            "safe_growth_accepted_count": dynamic_result.safe_growth_accepted_count,
            "safe_growth_rejected_peak_count": dynamic_result.safe_growth_rejected_peak_count,
            "exploration_growth_accepted_count": dynamic_result.exploration_growth_accepted_count,
            "exploration_growth_rejected_count": dynamic_result.exploration_growth_rejected_count,
            "final_efficiency": dynamic_result.final_efficiency,
            "meta_actions": dynamic_result.metrics.meta_action,
            "test_loss": dynamic_result.test_loss,
            "test_accuracy": dynamic_result.test_accuracy,
            "efficiency_over_time": dynamic_result.metrics.efficiency,
            "peak_efficiency_over_time": dynamic_result.metrics.peak_efficiency,
            "growth_enabled_over_time": dynamic_result.metrics.growth_enabled,
            "capacity_status_over_time": dynamic_result.metrics.capacity_status,
            "difficulty_score_over_time": dynamic_result.metrics.difficulty_score,
            "growth_allowed_over_time": dynamic_result.metrics.growth_allowed,
            "peak_efficiency": max(dynamic_result.metrics.peak_efficiency) if dynamic_result.metrics.peak_efficiency else 0.0,
            "no_growth_phase_entered": 0 in dynamic_result.metrics.growth_enabled,
            "candidate_growth_events_over_time": dynamic_result.metrics.candidate_growth_events,
            "exploration_growth_events_over_time": dynamic_result.metrics.exploration_growth_events,
            "accepted_growth_events_over_time": dynamic_result.metrics.accepted_growth_events,
            "rejected_growth_events_over_time": dynamic_result.metrics.rejected_growth_events,
            "safe_growth_accepted_over_time": dynamic_result.metrics.safe_growth_accepted,
            "safe_growth_rejected_peak_over_time": dynamic_result.metrics.safe_growth_rejected_peak,
            "exploration_growth_accepted_over_time": dynamic_result.metrics.exploration_growth_accepted,
            "exploration_growth_rejected_over_time": dynamic_result.metrics.exploration_growth_rejected,
            "final_checkpoint": dynamic_result.final_checkpoint,
        },
        "static": {
            "final_val_loss": static_result.final_val_loss,
            "final_val_accuracy": static_result.final_val_accuracy,
            "final_model_size": static_result.final_model_size,
            "growth_event_count": static_result.growth_event_count,
            "candidate_growth_event_count": static_result.candidate_growth_event_count,
            "exploration_growth_event_count": static_result.exploration_growth_event_count,
            "rejected_growth_event_count": static_result.rejected_growth_event_count,
            "safe_growth_accepted_count": static_result.safe_growth_accepted_count,
            "safe_growth_rejected_peak_count": static_result.safe_growth_rejected_peak_count,
            "exploration_growth_accepted_count": static_result.exploration_growth_accepted_count,
            "exploration_growth_rejected_count": static_result.exploration_growth_rejected_count,
            "final_efficiency": static_result.final_efficiency,
            "meta_actions": static_result.metrics.meta_action,
            "test_loss": static_result.test_loss,
            "test_accuracy": static_result.test_accuracy,
            "efficiency_over_time": static_result.metrics.efficiency,
            "peak_efficiency_over_time": static_result.metrics.peak_efficiency,
            "growth_enabled_over_time": static_result.metrics.growth_enabled,
            "capacity_status_over_time": static_result.metrics.capacity_status,
            "difficulty_score_over_time": static_result.metrics.difficulty_score,
            "growth_allowed_over_time": static_result.metrics.growth_allowed,
            "peak_efficiency": max(static_result.metrics.peak_efficiency) if static_result.metrics.peak_efficiency else 0.0,
            "no_growth_phase_entered": 0 in static_result.metrics.growth_enabled,
            "candidate_growth_events_over_time": static_result.metrics.candidate_growth_events,
            "exploration_growth_events_over_time": static_result.metrics.exploration_growth_events,
            "accepted_growth_events_over_time": static_result.metrics.accepted_growth_events,
            "rejected_growth_events_over_time": static_result.metrics.rejected_growth_events,
            "safe_growth_accepted_over_time": static_result.metrics.safe_growth_accepted,
            "safe_growth_rejected_peak_over_time": static_result.metrics.safe_growth_rejected_peak,
            "exploration_growth_accepted_over_time": static_result.metrics.exploration_growth_accepted,
            "exploration_growth_rejected_over_time": static_result.metrics.exploration_growth_rejected,
            "final_checkpoint": static_result.final_checkpoint,
        },
        "comparison": {
            "val_loss_delta_dynamic_minus_static": dynamic_result.final_val_loss - static_result.final_val_loss,
            "accuracy_delta_dynamic_minus_static": dynamic_result.final_val_accuracy - static_result.final_val_accuracy,
            "model_size_delta_dynamic_minus_static": dynamic_result.final_model_size - static_result.final_model_size,
            "test_accuracy_delta_dynamic_minus_static": dynamic_result.test_accuracy - static_result.test_accuracy,
            "test_loss_delta_dynamic_minus_static": dynamic_result.test_loss - static_result.test_loss,
        },
        "plot_path": str(plot_path),
    }
    summary_path = serialize_summary(output_dir, config.summary_filename, summary_payload)

    logger.info(
        "EXPERIMENT_RESULT | dynamic(val_loss=%.6f, acc=%.4f, params=%d, growth=%d) | static(val_loss=%.6f, acc=%.4f, params=%d, growth=%d)",
        dynamic_result.final_val_loss,
        dynamic_result.final_val_accuracy,
        dynamic_result.final_model_size,
        dynamic_result.growth_event_count,
        static_result.final_val_loss,
        static_result.final_val_accuracy,
        static_result.final_model_size,
        static_result.growth_event_count,
    )
    logger.info("Saved plot to %s", plot_path)
    logger.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    train(TrainingConfig())
