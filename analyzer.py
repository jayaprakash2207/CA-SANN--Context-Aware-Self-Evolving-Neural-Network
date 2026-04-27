from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Sequence

import torch
from torch import Tensor

from typing import Any


@dataclass
class AnalyzerConfig:
    sample_history: int = 24
    hard_error_percentile: float = 90.0
    hard_error_min_hits: int = 10
    hard_error_min_loss: float = 0.05
    dead_neuron_zero_fraction: float = 0.995
    min_activation_observations: int = 256
    gradient_ema_decay: float = 0.9
    loss_plateau_window: int = 3
    loss_plateau_min_delta: float = 3e-3
    loss_plateau_min_level: float = 0.01
    high_val_loss_threshold: float = 0.35
    high_val_loss_window: int = 3
    low_weight_norm_threshold: float = 5e-4
    low_activation_threshold: float = 1e-4
    max_prune_fraction_per_layer: float = 0.1
    importance_gradient_weight: float = 0.6
    importance_activation_weight: float = 0.4


class ErrorAnalyzer:
    """Tracks hard samples, dead neurons, and gradient dynamics for control decisions."""

    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config

        self.sample_losses: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=self.config.sample_history))
        self.epoch_losses: Deque[float] = deque(maxlen=256)
        self.val_epoch_losses: Deque[float] = deque(maxlen=256)

        self.activation_zero_counts: List[Tensor] = []
        self.activation_total_counts: List[int] = []
        self.activation_mean_sums: List[Tensor] = []
        self.neuron_grad_contrib_sums: List[Tensor] = []
        self.neuron_grad_contrib_counts: List[int] = []

        self.gradient_norm_last: Dict[str, float] = {}
        self.gradient_norm_ema: Dict[str, float] = {}

    def on_architecture_change(self, model: Any) -> None:
        self.activation_zero_counts = []
        self.activation_total_counts = []
        self.activation_mean_sums = []
        self.neuron_grad_contrib_sums = []
        self.neuron_grad_contrib_counts = []
        for dim in getattr(model, "hidden_dims", []):
            self.activation_zero_counts.append(torch.zeros(dim))
            self.activation_total_counts.append(0)
            self.activation_mean_sums.append(torch.zeros(dim))
            self.neuron_grad_contrib_sums.append(torch.zeros(dim))
            self.neuron_grad_contrib_counts.append(0)

    def update_batch(self, sample_indices: Tensor, per_sample_loss: Tensor, hidden_activations: Sequence[Tensor]) -> None:
        idx_cpu = sample_indices.detach().cpu()
        loss_cpu = per_sample_loss.detach().cpu()

        for idx, loss_val in zip(idx_cpu.tolist(), loss_cpu.tolist()):
            self.sample_losses[int(idx)].append(float(loss_val))

        if not self.activation_zero_counts:
            for act in hidden_activations:
                self.activation_zero_counts.append(torch.zeros(act.shape[1]))
                self.activation_total_counts.append(0)
                self.activation_mean_sums.append(torch.zeros(act.shape[1]))

        for layer_idx, act in enumerate(hidden_activations):
            act_cpu = act.detach().cpu()
            batch_size = act_cpu.shape[0]

            if act_cpu.ndim > 2:
                reduce_dims = tuple(range(2, act_cpu.ndim))
                act_cpu = act_cpu.mean(dim=reduce_dims)

            current_dim = act_cpu.shape[1]

            if self.activation_zero_counts[layer_idx].numel() != current_dim:
                self.activation_zero_counts[layer_idx] = torch.zeros(current_dim)
                self.activation_mean_sums[layer_idx] = torch.zeros(current_dim)
                self.activation_total_counts[layer_idx] = 0

            zero_counts = (act_cpu.abs() < 1e-8).sum(dim=0).float()
            self.activation_zero_counts[layer_idx] += zero_counts
            self.activation_total_counts[layer_idx] += int(batch_size)

            self.activation_mean_sums[layer_idx] += act_cpu.abs().mean(dim=0)

    def update_gradients(self, model: Any) -> None:
        current: Dict[str, float] = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            current[name] = float(param.grad.detach().norm(2).item())

        self.gradient_norm_last = current

        decay = self.config.gradient_ema_decay
        for name, value in current.items():
            prev = self.gradient_norm_ema.get(name, value)
            self.gradient_norm_ema[name] = decay * prev + (1.0 - decay) * value

        for layer_idx, layer in enumerate(getattr(model, "hidden_layers", [])):
            weight = getattr(getattr(layer, "linear", None), "weight", None)
            if weight is None:
                weight = getattr(getattr(layer, "conv", None), "weight", None)
            grad = weight.grad if weight is not None else None
            if grad is None:
                continue

            if grad.ndim == 2:
                row_contrib = grad.detach().abs().mean(dim=1).cpu()
            else:
                # Conv2d: [out_channels, in_channels, kH, kW] -> per-output-channel contribution.
                row_contrib = grad.detach().abs().mean(dim=(1, 2, 3)).cpu()
            if layer_idx >= len(self.neuron_grad_contrib_sums):
                self.neuron_grad_contrib_sums.append(torch.zeros_like(row_contrib))
                self.neuron_grad_contrib_counts.append(0)
            if self.neuron_grad_contrib_sums[layer_idx].numel() != row_contrib.numel():
                self.neuron_grad_contrib_sums[layer_idx] = torch.zeros_like(row_contrib)
                self.neuron_grad_contrib_counts[layer_idx] = 0
            self.neuron_grad_contrib_sums[layer_idx] += row_contrib
            self.neuron_grad_contrib_counts[layer_idx] += 1

    def record_epoch_loss(self, epoch_loss: float) -> None:
        self.epoch_losses.append(float(epoch_loss))

    def record_val_loss(self, val_loss: float) -> None:
        self.val_epoch_losses.append(float(val_loss))

    def loss_is_stagnating(self) -> bool:
        w = self.config.loss_plateau_window
        if len(self.epoch_losses) < w + 1:
            return False

        recent = list(self.epoch_losses)[-(w + 1) :]
        start = recent[0]
        end = recent[-1]
        if end < self.config.loss_plateau_min_level:
            return False
        improvement = start - end
        return improvement < self.config.loss_plateau_min_delta

    def identify_consistently_hard_samples(self) -> List[int]:
        if not self.sample_losses:
            return []

        sample_means: List[float] = []
        for history in self.sample_losses.values():
            if history:
                sample_means.append(sum(history) / float(len(history)))

        if len(sample_means) < 16:
            return []

        mean_tensor = torch.tensor(sample_means)
        if float(mean_tensor.max().item()) < self.config.hard_error_min_loss:
            return []

        q = self.config.hard_error_percentile / 100.0
        threshold = float(torch.quantile(mean_tensor, q=q).item())

        hard_ids: List[int] = []
        for sample_id, history in self.sample_losses.items():
            if len(history) < self.config.hard_error_min_hits:
                continue
            sample_mean = sum(history) / float(len(history))
            if sample_mean < self.config.hard_error_min_loss:
                continue
            hit_count = sum(1 for value in history if value >= threshold)
            if hit_count >= self.config.hard_error_min_hits:
                hard_ids.append(sample_id)
        return hard_ids

    def val_loss_is_consistently_high(self) -> bool:
        w = max(1, self.config.high_val_loss_window)
        if len(self.val_epoch_losses) < w:
            return False
        recent = list(self.val_epoch_losses)[-w:]
        return all(value >= self.config.high_val_loss_threshold for value in recent)

    def dead_neuron_indices(self) -> Dict[int, List[int]]:
        dead: Dict[int, List[int]] = {}
        for layer_idx, zero_counts in enumerate(self.activation_zero_counts):
            total = self.activation_total_counts[layer_idx]
            if total < self.config.min_activation_observations:
                dead[layer_idx] = []
                continue

            zero_fraction = zero_counts / float(total)
            dead[layer_idx] = torch.where(zero_fraction >= self.config.dead_neuron_zero_fraction)[0].tolist()
        return dead

    def low_contribution_neurons(self, model: DynamicMLP) -> Dict[int, List[int]]:
        candidates: Dict[int, List[int]] = {}

        for layer_idx, layer in enumerate(model.hidden_layers):
            dim = layer.out_features
            if dim == 0:
                candidates[layer_idx] = []
                continue

            weights_tensor = getattr(getattr(layer, "linear", None), "weight", None)
            if weights_tensor is None:
                weights_tensor = getattr(getattr(layer, "conv", None), "weight", None)
            if weights_tensor is None:
                candidates[layer_idx] = []
                continue
            weights = weights_tensor.detach().cpu()
            if weights.ndim == 2:
                row_norms = torch.norm(weights, p=2, dim=1)
            else:
                row_norms = torch.norm(weights.view(weights.shape[0], -1), p=2, dim=1)

            total_obs = max(self.activation_total_counts[layer_idx], 1)
            avg_activation = self.activation_mean_sums[layer_idx] / float(total_obs)
            grad_obs = max(self.neuron_grad_contrib_counts[layer_idx], 1)
            avg_grad_contrib = self.neuron_grad_contrib_sums[layer_idx] / float(grad_obs)

            low_weight = row_norms <= self.config.low_weight_norm_threshold
            low_act = avg_activation <= self.config.low_activation_threshold
            low_grad = avg_grad_contrib <= self.config.low_weight_norm_threshold
            inactive_mask = layer.mask.detach().cpu() <= 0

            candidate_mask = ((low_weight & low_act & low_grad) | inactive_mask)
            indices = torch.where(candidate_mask)[0].tolist()

            max_count = max(1, int(self.config.max_prune_fraction_per_layer * dim))
            candidates[layer_idx] = indices[:max_count]

        return candidates

    def neuron_importance(self, model: Any) -> Dict[int, Tensor]:
        importance_by_layer: Dict[int, Tensor] = {}

        grad_w = max(0.0, float(self.config.importance_gradient_weight))
        act_w = max(0.0, float(self.config.importance_activation_weight))
        total_w = max(1e-12, grad_w + act_w)
        grad_w /= total_w
        act_w /= total_w

        for layer_idx, layer in enumerate(getattr(model, "hidden_layers", [])):
            width = layer.out_features
            if width <= 0:
                importance_by_layer[layer_idx] = torch.zeros(0)
                continue

            grad_obs = max(1, self.neuron_grad_contrib_counts[layer_idx])
            avg_grad = self.neuron_grad_contrib_sums[layer_idx] / float(grad_obs)

            act_obs = max(1, self.activation_total_counts[layer_idx])
            avg_act = self.activation_mean_sums[layer_idx] / float(act_obs)

            grad_den = float(avg_grad.max().item()) if avg_grad.numel() > 0 else 0.0
            act_den = float(avg_act.max().item()) if avg_act.numel() > 0 else 0.0
            grad_norm = avg_grad / max(grad_den, 1e-12)
            act_norm = avg_act / max(act_den, 1e-12)

            importance = grad_w * grad_norm + act_w * act_norm
            active_mask = layer.mask.detach().cpu() > 0
            importance = importance * active_mask.float()
            importance_by_layer[layer_idx] = importance

        return importance_by_layer

    def layer_importance(self, model: Any) -> Dict[int, float]:
        neuron_imp = self.neuron_importance(model)
        layer_scores: Dict[int, float] = {}
        for layer_idx, layer in enumerate(getattr(model, "hidden_layers", [])):
            imp = neuron_imp.get(layer_idx)
            if imp is None or imp.numel() == 0:
                layer_scores[layer_idx] = 0.0
                continue
            active_mask = layer.mask.detach().cpu() > 0
            if int(active_mask.sum().item()) == 0:
                layer_scores[layer_idx] = 0.0
                continue
            layer_scores[layer_idx] = float(imp[active_mask].mean().item())
        return layer_scores

    @staticmethod
    def _rank_lowest(scores: Dict[int, float]) -> List[int]:
        return [idx for idx, _ in sorted(scores.items(), key=lambda item: item[1])]

    @staticmethod
    def _rank_neurons_lowest(importance: Tensor, max_count: int) -> List[int]:
        if importance.numel() == 0 or max_count <= 0:
            return []
        count = min(max_count, int(importance.numel()))
        return torch.argsort(importance, descending=False)[:count].tolist()

    def layer_gradient_score(self) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        for layer_idx in range(128):
            key_linear = f"hidden_layers.{layer_idx}.linear.weight"
            key_conv = f"hidden_layers.{layer_idx}.conv.weight"
            if key_linear in self.gradient_norm_ema:
                scores[layer_idx] = self.gradient_norm_ema[key_linear]
                continue
            if key_conv in self.gradient_norm_ema:
                scores[layer_idx] = self.gradient_norm_ema[key_conv]
                continue
            break
        return scores

    def report(self, model: Any) -> dict:
        dead = self.dead_neuron_indices()
        dead_ratio = {
            idx: (len(dead[idx]) / float(max(1, model.hidden_layers[idx].out_features))) for idx in dead
        }
        neuron_importance = self.neuron_importance(model)
        layer_importance = self.layer_importance(model)
        layer_underperformance = {idx: 1.0 - score for idx, score in layer_importance.items()}
        ranked_layers_weakest_first = self._rank_lowest(layer_importance)
        ranked_weak_neurons: Dict[int, List[int]] = {}
        for layer_idx, layer in enumerate(model.hidden_layers):
            width = max(1, layer.out_features)
            ranked_weak_neurons[layer_idx] = self._rank_neurons_lowest(
                neuron_importance.get(layer_idx, torch.zeros(width)),
                max_count=max(1, int(0.5 * width)),
            )

        return {
            "stagnating_loss": self.loss_is_stagnating(),
            "high_val_loss": self.val_loss_is_consistently_high(),
            "hard_sample_ids": self.identify_consistently_hard_samples(),
            "dead_neurons": dead,
            "dead_ratio": dead_ratio,
            "gradient_score": self.layer_gradient_score(),
            "low_contribution": self.low_contribution_neurons(model),
            "layer_importance": layer_importance,
            "layer_underperformance": layer_underperformance,
            "ranked_layers_weakest_first": ranked_layers_weakest_first,
            "ranked_weak_neurons": ranked_weak_neurons,
        }
