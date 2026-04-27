from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from analyzer import ErrorAnalyzer


CapacityStatus = Literal["underfit", "optimal", "overfit"]


@dataclass
class CapacityEstimatorConfig:
    # Gap thresholds are expressed in loss units for cross-entropy.
    overfit_gap_min: float = 0.15
    underfit_loss_min: float = 0.45
    underfit_stagnation_required: bool = True
    hard_sample_pressure_min: float = 0.20
    underfit_error_rate_min: float = 0.18
    overfit_train_loss_max: float = 0.30


@dataclass
class DifficultyEstimatorConfig:
    hard_sample_trigger: int = 32
    # Weights for the composite score.
    w_hard_pressure: float = 0.30
    w_entropy: float = 0.25
    w_grad_cv: float = 0.20
    w_loss_tail: float = 0.15
    w_error_rate: float = 0.05
    w_error_entropy: float = 0.05


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_div(num: float, den: float, eps: float = 1e-12) -> float:
    return num / (den if abs(den) > eps else eps)


@torch.no_grad()
def prediction_entropy_stats(
    model: Any,
    data_loader: DataLoader,
    device: torch.device,
    *,
    num_classes: int,
    max_batches: int | None = 8,
) -> tuple[float, float]:
    """Returns (mean_entropy_normalized, mean_max_prob).

    Entropy is normalized by log(num_classes) to fall in [0, 1] approximately.
    """
    model.eval()
    total_entropy = 0.0
    total_maxp = 0.0
    total = 0
    logk = float(math.log(max(2, int(num_classes))))

    for batch_idx, (x, _, _) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        if bool(getattr(model, "expects_flatten_input", False)):
            x = x.view(x.size(0), -1)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        maxp = probs.max(dim=1).values
        # Clamp for numerical stability.
        p = probs.clamp_min(1e-12)
        entropy = -(p * p.log()).sum(dim=1)

        total_entropy += float(entropy.sum().item())
        total_maxp += float(maxp.sum().item())
        total += int(x.shape[0])

    if total <= 0:
        return 0.0, 0.0
    mean_entropy = total_entropy / float(total)
    mean_entropy_norm = mean_entropy / max(1e-12, logk)
    mean_maxp = total_maxp / float(total)
    return _clip01(mean_entropy_norm), _clip01(mean_maxp)


@torch.no_grad()
def prediction_diagnostics(
    model: Any,
    data_loader: DataLoader,
    device: torch.device,
    *,
    num_classes: int,
    max_batches: int | None = 8,
) -> dict[str, float]:
    """Lightweight validation diagnostics used for context estimation.

    Returns a dictionary with:
    - entropy_norm: mean predictive entropy normalized by log(K)
    - mean_max_prob: mean max-softmax probability
    - error_rate: fraction misclassified on the probed validation batches
    - error_class_entropy_norm: entropy (normalized) of error counts by true class
      (high => errors spread across many classes; low => concentrated in few classes)
    """
    model.eval()
    total_entropy = 0.0
    total_maxp = 0.0
    total = 0
    incorrect = 0
    error_by_true = torch.zeros(int(num_classes), dtype=torch.float32)
    logk = float(math.log(max(2, int(num_classes))))

    for batch_idx, (x, y, _) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        if bool(getattr(model, "expects_flatten_input", False)):
            x = x.view(x.size(0), -1)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        maxp = probs.max(dim=1).values
        p = probs.clamp_min(1e-12)
        entropy = -(p * p.log()).sum(dim=1)
        preds = torch.argmax(logits, dim=1)
        is_incorrect = preds.ne(y)

        total_entropy += float(entropy.sum().item())
        total_maxp += float(maxp.sum().item())
        total += int(x.shape[0])
        incorrect += int(is_incorrect.sum().item())
        if int(num_classes) > 0:
            true_classes = y[is_incorrect].detach().to("cpu")
            if true_classes.numel() > 0:
                error_by_true.scatter_add_(0, true_classes, torch.ones_like(true_classes, dtype=torch.float32))

    if total <= 0:
        return {
            "entropy_norm": 0.0,
            "mean_max_prob": 0.0,
            "error_rate": 0.0,
            "error_class_entropy_norm": 0.0,
        }

    mean_entropy_norm = _clip01((total_entropy / float(total)) / max(1e-12, logk))
    mean_maxp = _clip01(total_maxp / float(total))
    error_rate = _clip01(float(incorrect) / float(total))

    if float(error_by_true.sum().item()) <= 0.0:
        error_class_entropy_norm = 0.0
    else:
        q = (error_by_true / error_by_true.sum()).clamp_min(1e-12)
        ent = float((-(q * q.log()).sum()).item())
        error_class_entropy_norm = _clip01(ent / max(1e-12, logk))

    return {
        "entropy_norm": mean_entropy_norm,
        "mean_max_prob": mean_maxp,
        "error_rate": error_rate,
        "error_class_entropy_norm": error_class_entropy_norm,
    }


def gradient_cv(gradient_scores: Iterable[float]) -> float:
    values = [float(v) for v in gradient_scores if v is not None]
    if len(values) < 2:
        return 0.0
    mean = sum(values) / float(len(values))
    var = sum((v - mean) ** 2 for v in values) / float(len(values))
    std = math.sqrt(max(0.0, var))
    return float(_safe_div(std, abs(mean) + 1e-12))


def loss_tail_ratio(sample_means: list[float]) -> float:
    if len(sample_means) < 32:
        return 0.0
    t = torch.tensor(sample_means)
    q50 = float(torch.quantile(t, q=0.50).item())
    q90 = float(torch.quantile(t, q=0.90).item())
    ratio = _safe_div(max(0.0, q90 - q50), abs(q50) + 1e-12)
    # Squash to [0, 1).
    return float(math.tanh(ratio))


class CapacityEstimator:
    def __init__(self, config: CapacityEstimatorConfig | None = None) -> None:
        self.config = config or CapacityEstimatorConfig()

    def estimate(
        self,
        *,
        analyzer: ErrorAnalyzer,
        train_loss: float,
        val_loss: float,
        report: dict,
    ) -> CapacityStatus:
        gap = float(val_loss - train_loss)
        stagnating = bool(report.get("stagnating_loss", False))
        high_val_loss = bool(report.get("high_val_loss", False))
        hard_ids = report.get("hard_sample_ids", [])
        hard_pressure = min(1.0, len(hard_ids) / float(max(1, self.config.hard_sample_pressure_min * 32.0)))
        val_error_rate = float(report.get("val_error_rate", 0.0))
        val_error_entropy = float(report.get("val_error_class_entropy_norm", 0.0))

        # Overfit: training does well but validation deteriorates with a large gap.
        if high_val_loss and gap >= self.config.overfit_gap_min and train_loss <= self.config.overfit_train_loss_max:
            return "overfit"

        # Underfit: both losses high, plateauing, and many persistent hard samples.
        if val_loss >= self.config.underfit_loss_min and train_loss >= (0.85 * self.config.underfit_loss_min):
            if (not self.config.underfit_stagnation_required) or stagnating:
                if hard_pressure >= self.config.hard_sample_pressure_min or val_error_rate >= self.config.underfit_error_rate_min:
                    return "underfit"
            return "underfit"

        # Underfit can also manifest as high error rate with broad misclassifications, even if loss gap is small.
        if val_error_rate >= self.config.underfit_error_rate_min and val_error_entropy >= 0.65 and high_val_loss:
            return "underfit"

        return "optimal"


class DifficultyEstimator:
    def __init__(self, config: DifficultyEstimatorConfig | None = None) -> None:
        self.config = config or DifficultyEstimatorConfig()

    def estimate(
        self,
        *,
        analyzer: ErrorAnalyzer,
        report: dict,
        val_entropy_norm: float,
        val_error_rate: float = 0.0,
        val_error_class_entropy_norm: float = 0.0,
    ) -> float:
        hard_ids = report.get("hard_sample_ids", [])
        hard_pressure = min(1.0, len(hard_ids) / float(max(1, self.config.hard_sample_trigger)))
        grad_scores = report.get("gradient_score", {}).values()
        grad_cv_val = gradient_cv(grad_scores)
        grad_cv_norm = float(math.tanh(grad_cv_val))

        sample_means = []
        for history in analyzer.sample_losses.values():
            if len(history) == 0:
                continue
            sample_means.append(sum(history) / float(len(history)))
        tail = loss_tail_ratio(sample_means)

        score = (
            self.config.w_hard_pressure * hard_pressure
            + self.config.w_entropy * _clip01(val_entropy_norm)
            + self.config.w_grad_cv * _clip01(grad_cv_norm)
            + self.config.w_loss_tail * _clip01(tail)
            + self.config.w_error_rate * _clip01(val_error_rate)
            + self.config.w_error_entropy * _clip01(val_error_class_entropy_norm)
        )
        return _clip01(score)
