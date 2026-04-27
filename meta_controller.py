from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple

from analyzer import ErrorAnalyzer
from model import DynamicMLP


MetaActionType = Literal["grow", "prune", "noop"]


@dataclass
class MetaControllerConfig:
    warmup_epochs: int = 3
    decision_interval: int = 1
    min_confidence: float = 0.35
    growth_confidence_threshold: float = 0.55
    prune_confidence_threshold: float = 0.45
    min_loss_for_growth: float = 0.03
    stagnation_window: int = 5
    cooldown_epochs: int = 2
    max_growth_frequency: int = 12
    gradient_vanishing_threshold: float = 5e-4
    activation_dead_ratio_threshold: float = 0.2
    hard_sample_pressure_threshold: float = 0.25
    loss_trend_weight: float = 0.35
    gradient_weight: float = 0.35
    activation_weight: float = 0.30
    prune_low_activation_weight: float = 0.45
    prune_low_gradient_weight: float = 0.25
    growth_bias: float = 0.15
    prune_bias: float = 0.10


@dataclass
class MetaDecision:
    action: MetaActionType
    target_layer: int | None
    confidence: float
    rationale: str
    grow_multiplier: float = 1.0
    prune_fraction: float = 0.0


@dataclass
class MetaState:
    last_decision_epoch: int = -10_000
    last_growth_epoch: int = -10_000
    decision_history: list[MetaDecision] = field(default_factory=list)


class MetaController:
    """Policy-based meta-controller that decides when the network should grow, prune, or hold."""

    def __init__(self, config: MetaControllerConfig | None = None) -> None:
        self.config = config or MetaControllerConfig()
        self.state = MetaState()

    def decide(self, model: DynamicMLP, analyzer: ErrorAnalyzer, epoch: int) -> MetaDecision:
        report = analyzer.report(model)
        current_loss = float(analyzer.epoch_losses[-1]) if analyzer.epoch_losses else 0.0

        if epoch < self.config.warmup_epochs:
            return self._record(MetaDecision("noop", None, 0.0, "warmup"), epoch)
        if epoch % self.config.decision_interval != 0:
            return self._record(MetaDecision("noop", None, 0.0, "interval_skip"), epoch)
        if epoch - self.state.last_growth_epoch < self.config.cooldown_epochs:
            return self._record(MetaDecision("noop", None, 0.0, "cooldown"), epoch)
        if current_loss < self.config.min_loss_for_growth:
            return self._record(MetaDecision("noop", None, 0.0, "loss_floor"), epoch)

        loss_score = self._loss_trend_score(analyzer)
        gradient_scores = report.get("gradient_score", {})
        activation_scores = self._activation_scores(model, analyzer, report)
        hard_sample_pressure = min(1.0, len(report.get("hard_sample_ids", [])) / 32.0)
        if hard_sample_pressure < self.config.hard_sample_pressure_threshold and loss_score < 0.5:
            return self._record(MetaDecision("noop", None, 0.0, "low_pressure"), epoch)

        best_layer, growth_confidence, growth_rationale = self._score_growth_layer(
            gradient_scores=gradient_scores,
            activation_scores=activation_scores,
            hard_sample_pressure=hard_sample_pressure,
            loss_score=loss_score,
            model=model,
        )
        prune_layer, prune_confidence, prune_rationale = self._score_prune_layer(
            gradient_scores=gradient_scores,
            activation_scores=activation_scores,
            model=model,
        )

        if growth_confidence < self.config.min_confidence and prune_confidence < self.config.min_confidence:
            return self._record(MetaDecision("noop", None, max(growth_confidence, prune_confidence), "low_confidence"), epoch)

        if growth_confidence >= self.config.growth_confidence_threshold and growth_confidence >= prune_confidence:
            decision = MetaDecision(
                action="grow",
                target_layer=best_layer,
                confidence=growth_confidence,
                rationale=growth_rationale,
                grow_multiplier=self._growth_multiplier(loss_score, hard_sample_pressure),
            )
            return self._record(decision, epoch, growth=True)

        if prune_confidence >= self.config.prune_confidence_threshold:
            decision = MetaDecision(
                action="prune",
                target_layer=prune_layer,
                confidence=prune_confidence,
                rationale=prune_rationale,
                prune_fraction=self._prune_fraction(activation_scores, prune_layer),
            )
            return self._record(decision, epoch)

        return self._record(MetaDecision("noop", None, max(growth_confidence, prune_confidence), "policy_hold"), epoch)

    def _record(self, decision: MetaDecision, epoch: int, growth: bool = False) -> MetaDecision:
        self.state.last_decision_epoch = epoch
        if growth and decision.action == "grow":
            self.state.last_growth_epoch = epoch
        self.state.decision_history.append(decision)
        return decision

    def _loss_trend_score(self, analyzer: ErrorAnalyzer) -> float:
        if len(analyzer.epoch_losses) < 2:
            return 0.0

        window = list(analyzer.epoch_losses)[-self.config.stagnation_window :]
        if len(window) < 2:
            return 0.0

        start = window[0]
        end = window[-1]
        if start <= 0.0:
            return 0.0

        relative_improvement = (start - end) / max(start, 1e-12)
        return max(0.0, min(1.0, 1.0 - relative_improvement))

    def _activation_scores(self, model: DynamicMLP, analyzer: ErrorAnalyzer, report: dict) -> Dict[int, float]:
        dead = report.get("dead_neurons", {})
        low_contribution = report.get("low_contribution", {})
        activation_scores: Dict[int, float] = {}

        for layer_idx, layer in enumerate(model.hidden_layers):
            dead_ratio = len(dead.get(layer_idx, [])) / float(max(1, layer.out_features))
            low_ratio = len(low_contribution.get(layer_idx, [])) / float(max(1, layer.out_features))
            activation_scores[layer_idx] = max(dead_ratio, low_ratio)

        return activation_scores

    def _score_growth_layer(
        self,
        *,
        gradient_scores: Dict[int, float],
        activation_scores: Dict[int, float],
        hard_sample_pressure: float,
        loss_score: float,
        model: DynamicMLP,
    ) -> Tuple[int, float, str]:
        best_layer = 0
        best_score = -1.0
        reasons = []

        max_grad = max(gradient_scores.values(), default=0.0)
        min_grad = min(gradient_scores.values(), default=0.0)
        grad_span = max(max_grad - min_grad, 1e-12)

        for layer_idx, layer in enumerate(model.hidden_layers):
            grad = float(gradient_scores.get(layer_idx, 0.0))
            activation_pressure = float(activation_scores.get(layer_idx, 0.0))
            vanishing_grad = 1.0 if grad <= self.config.gradient_vanishing_threshold else 0.0
            normalized_grad_deficit = 1.0 - ((grad - min_grad) / grad_span)
            score = (
                self.config.loss_trend_weight * loss_score
                + self.config.gradient_weight * normalized_grad_deficit
                + self.config.activation_weight * activation_pressure
                + self.config.growth_bias * hard_sample_pressure
                + 0.2 * vanishing_grad
            )
            reasons.append(
                f"layer={layer_idx},grad={grad:.6f},act={activation_pressure:.3f},loss_score={loss_score:.3f},hard={hard_sample_pressure:.3f}"
            )
            if score > best_score:
                best_score = score
                best_layer = layer_idx

        return best_layer, best_score, f"growth_policy({'; '.join(reasons)})"

    def _score_prune_layer(
        self,
        *,
        gradient_scores: Dict[int, float],
        activation_scores: Dict[int, float],
        model: DynamicMLP,
    ) -> Tuple[int, float, str]:
        best_layer = 0
        best_score = -1.0
        reasons = []

        max_grad = max(gradient_scores.values(), default=0.0)
        for layer_idx, layer in enumerate(model.hidden_layers):
            grad = float(gradient_scores.get(layer_idx, 0.0))
            activation_pressure = float(activation_scores.get(layer_idx, 0.0))
            low_gradient_pressure = 1.0 - (grad / max(max_grad, 1e-12)) if max_grad > 0.0 else 1.0
            score = (
                self.config.prune_low_activation_weight * activation_pressure
                + self.config.prune_low_gradient_weight * low_gradient_pressure
                + self.config.prune_bias
            )
            reasons.append(f"layer={layer_idx},grad={grad:.6f},act={activation_pressure:.3f},score={score:.3f}")
            if score > best_score:
                best_score = score
                best_layer = layer_idx

        return best_layer, best_score, f"prune_policy({'; '.join(reasons)})"

    def _growth_multiplier(self, loss_score: float, hard_sample_pressure: float) -> float:
        multiplier = 1.0 + 0.5 * loss_score + 0.25 * hard_sample_pressure
        return max(1.0, min(2.0, multiplier))

    def _prune_fraction(self, activation_scores: Dict[int, float], layer_idx: int) -> float:
        activation_pressure = float(activation_scores.get(layer_idx, 0.0))
        return max(0.05, min(0.25, 0.10 + 0.15 * activation_pressure))