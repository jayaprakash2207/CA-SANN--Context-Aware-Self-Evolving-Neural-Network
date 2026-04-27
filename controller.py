from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from analyzer import ErrorAnalyzer
from typing import Any


@dataclass
class GrowthConfig:
    warmup_epochs: int = 5
    decision_interval: int = 2
    min_growth_loss: float = 0.01
    gradient_vanishing_threshold: float = 1e-4
    activation_dead_ratio_threshold: float = 0.35
    hard_sample_trigger: int = 32
    high_val_loss_trigger: bool = True
    force_growth_after_epochs: int = 4
    force_growth_neurons: int = 2
    max_model_params: int = 500000
    base_growth: int = 8
    max_growth_per_event: int = 32
    max_growth_per_layer: int = 24
    cooldown_epochs: int = 3
    max_total_growth_ratio: float = 0.25
    min_layer_growth: int = 4
    dead_neuron_priority_boost: float = 0.6
    low_activation_priority_boost: float = 0.35
    vanishing_gradient_priority_boost: float = 0.5
    hard_sample_priority_boost: float = 0.25
    min_growth_priority: float = 0.75
    require_growth_stagnation: bool = True
    min_hard_sample_pressure: float = 0.15
    total_growth_budget_neurons: int = 48
    min_underperformance_to_grow: float = 0.30
    max_layer_importance_to_grow: float = 0.72
    weak_neuron_fraction_for_growth: float = 0.30

    allow_candidate_growth: bool = True
    candidate_min_growth_priority: float = 0.45
    candidate_min_underperformance_to_grow: float = 0.05
    candidate_max_layer_importance_to_grow: float = 0.90
    candidate_weak_neuron_fraction_for_growth: float = 0.10
    candidate_growth_neurons: int = 2


@dataclass
class PruneConfig:
    enabled: bool = True
    prune_interval: int = 3
    max_neurons_per_event: int = 16
    min_active_fraction: float = 0.35
    require_stagnation: bool = True
    skip_if_high_val_loss: bool = True


@dataclass
class ControllerConfig:
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    prune: PruneConfig = field(default_factory=PruneConfig)


@dataclass
class ArchitectureAction:
    kind: str
    layer_idx: int
    neurons: List[int] | None = None
    n_new: int = 0
    reason: str = ""
    candidate: bool = False


@dataclass
class LayerGrowthSignal:
    layer_idx: int
    priority: float
    policy_bias: float
    policy_score: float
    gradient_norm: float
    dead_neuron_count: int
    dead_ratio: float
    low_activation_ratio: float
    hard_sample_pressure: float
    layer_importance: float
    underperformance: float
    weak_neuron_ratio: float


@dataclass
class GrowthOutcome:
    layer_idx: int
    accepted: bool
    exploration: bool
    val_acc_gain: float
    efficiency_delta: float
    epoch: int


@dataclass
class LearningPolicyConfig:
    enabled: bool = True
    lr: float = 0.25
    decay: float = 0.985
    score_clip: float = 4.0
    bias_weight: float = 0.20
    # Small extra preference for under-explored layers early on.
    novelty_bonus: float = 0.10
    max_history: int = 200


@dataclass
class LayerPolicyStats:
    score: float = 0.0
    trials: int = 0
    successes: int = 0
    failures: int = 0
    exploration_trials: int = 0
    exploration_successes: int = 0
    exploration_failures: int = 0


class LearningPolicy:
    """Lightweight online 'learning' for growth targeting.

    Tracks growth outcomes and turns them into a heuristic score per layer.
    That score biases future priorities (more likely to grow in layers where
    growth historically helped; less likely where it hurt).
    """

    def __init__(self, config: LearningPolicyConfig) -> None:
        self.config = config
        self.layer: Dict[int, LayerPolicyStats] = {}
        self.history: list[GrowthOutcome] = []

    def layer_bias(self, layer_idx: int) -> tuple[float, float]:
        if not self.config.enabled:
            return 0.0, 0.0
        stats = self.layer.get(layer_idx)
        if stats is None:
            return 0.0, 0.0
        # Saturating transform so bias doesn't dominate core signals.
        bias = self.config.bias_weight * math.tanh(stats.score)
        return bias, stats.score

    def update(self, outcome: GrowthOutcome) -> None:
        if not self.config.enabled:
            return

        stats = self.layer.setdefault(outcome.layer_idx, LayerPolicyStats())
        stats.trials += 1
        if outcome.exploration:
            stats.exploration_trials += 1

        # Reward: prefer acc gains and efficiency gains, punish reverts/rejections.
        # Keep magnitudes modest so this stays a bias, not a hard rule.
        reward = 0.0
        if outcome.accepted:
            stats.successes += 1
            if outcome.exploration:
                stats.exploration_successes += 1
            reward += 1.0
            reward += 20.0 * float(outcome.val_acc_gain)
            reward += 2.5e5 * float(outcome.efficiency_delta)
        else:
            stats.failures += 1
            if outcome.exploration:
                stats.exploration_failures += 1
            reward -= 1.25
            reward += 5.0 * float(outcome.val_acc_gain)
            reward += 1.0e5 * float(outcome.efficiency_delta)

        # Give a mild novelty bump so early learning doesn't get stuck.
        if stats.trials <= 3:
            reward += self.config.novelty_bonus

        stats.score = stats.score * self.config.decay + self.config.lr * reward
        stats.score = float(max(-self.config.score_clip, min(self.config.score_clip, stats.score)))

        self.history.append(outcome)
        if len(self.history) > self.config.max_history:
            self.history = self.history[-self.config.max_history :]


class GrowthController:
    """Policy engine that converts analyzer signals into growth/pruning actions."""

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        self._last_growth_epoch = -10_000
        self._last_growth_epoch_by_layer: Dict[int, int] = {}
        self._growth_added_by_layer: Dict[int, int] = {}
        self._total_growth_added = 0
        self.policy = LearningPolicy(LearningPolicyConfig())

    def record_growth_outcome(self, outcome: GrowthOutcome) -> None:
        self.policy.update(outcome)

    def decide(self, model: Any, analyzer: ErrorAnalyzer, epoch: int) -> List[ArchitectureAction]:
        actions: List[ArchitectureAction] = []
        report = analyzer.report(model)

        if model.num_trainable_parameters >= self.config.growth.max_model_params:
            return actions
        if self._total_growth_added >= self.config.growth.total_growth_budget_neurons:
            return actions

        if self._can_consider_growth(epoch, analyzer):
            growth_signals = self._score_layers(model, analyzer, report)
            if growth_signals:
                best_signal = min(
                    growth_signals,
                    key=lambda signal: (
                        signal.layer_importance,
                        -signal.underperformance,
                        -signal.priority,
                    ),
                )
                high_val_loss = bool(report.get("high_val_loss", False))
                stagnating = bool(report.get("stagnating_loss", False))
                should_grow = best_signal.priority >= self.config.growth.min_growth_priority
                if self.config.growth.require_growth_stagnation and not stagnating and not high_val_loss:
                    should_grow = False
                if best_signal.hard_sample_pressure < self.config.growth.min_hard_sample_pressure and not high_val_loss:
                    should_grow = False
                if best_signal.underperformance < self.config.growth.min_underperformance_to_grow and not high_val_loss:
                    should_grow = False
                if best_signal.layer_importance > self.config.growth.max_layer_importance_to_grow and not high_val_loss:
                    should_grow = False
                if self.config.growth.high_val_loss_trigger and high_val_loss:
                    should_grow = True

                is_candidate = False
                if not should_grow and self.config.growth.allow_candidate_growth:
                    candidate_ok = (
                        best_signal.priority >= self.config.growth.candidate_min_growth_priority
                        and best_signal.underperformance >= self.config.growth.candidate_min_underperformance_to_grow
                        and best_signal.layer_importance <= self.config.growth.candidate_max_layer_importance_to_grow
                        and best_signal.weak_neuron_ratio >= self.config.growth.candidate_weak_neuron_fraction_for_growth
                    )
                    if candidate_ok:
                        should_grow = True
                        is_candidate = True

                if should_grow:
                    if is_candidate:
                        n_new = max(1, int(self.config.growth.candidate_growth_neurons))
                    else:
                        n_new = self._compute_growth_size(best_signal, model)
                    budget_left = max(0, self.config.growth.total_growth_budget_neurons - self._total_growth_added)
                    n_new = min(n_new, budget_left)
                    if n_new > 0:
                        actions.append(
                            ArchitectureAction(
                                kind="grow",
                                layer_idx=best_signal.layer_idx,
                                n_new=n_new,
                                reason=(
                                    f"candidate={is_candidate}, "
                                    f"priority={best_signal.priority:.4f}, gradient_norm={best_signal.gradient_norm:.6f}, "
                                    f"policy_bias={best_signal.policy_bias:.4f}, policy_score={best_signal.policy_score:.4f}, "
                                    f"dead_ratio={best_signal.dead_ratio:.3f}, low_activation_ratio={best_signal.low_activation_ratio:.3f}, "
                                    f"hard_sample_pressure={best_signal.hard_sample_pressure:.3f}, layer_importance={best_signal.layer_importance:.3f}, "
                                    f"underperformance={best_signal.underperformance:.3f}, weak_neuron_ratio={best_signal.weak_neuron_ratio:.3f}, "
                                    f"high_val_loss={high_val_loss}, n_new={n_new}"
                                ),
                                candidate=is_candidate,
                            )
                        )

        if (
            epoch >= self.config.growth.warmup_epochs
            and epoch - self._last_growth_epoch >= self.config.growth.force_growth_after_epochs
            and len(actions) == 0
        ):
            report = analyzer.report(model)
            growth_signals = self._score_layers(model, analyzer, report)
            if growth_signals:
                target = min(
                    growth_signals,
                    key=lambda signal: (
                        signal.layer_importance,
                        -signal.underperformance,
                        -signal.priority,
                    ),
                )
                strict_ok = (
                    target.underperformance >= self.config.growth.min_underperformance_to_grow
                    and target.layer_importance <= self.config.growth.max_layer_importance_to_grow
                )
                candidate_ok = (
                    self.config.growth.allow_candidate_growth
                    and target.underperformance >= self.config.growth.candidate_min_underperformance_to_grow
                    and target.layer_importance <= self.config.growth.candidate_max_layer_importance_to_grow
                    and target.weak_neuron_ratio >= self.config.growth.candidate_weak_neuron_fraction_for_growth
                )
                if strict_ok or candidate_ok:
                    fallback_n = max(1, self.config.growth.force_growth_neurons)
                    is_candidate = False
                    if not strict_ok and candidate_ok:
                        fallback_n = max(1, int(self.config.growth.candidate_growth_neurons))
                        is_candidate = True
                    actions.append(
                        ArchitectureAction(
                            kind="grow",
                            layer_idx=target.layer_idx,
                            n_new=fallback_n,
                            reason=(
                                f"fallback_force_growth=no_growth_for_{epoch - self._last_growth_epoch}_epochs, "
                                f"candidate={is_candidate}, target_layer={target.layer_idx}, underperformance={target.underperformance:.3f}, "
                                f"layer_importance={target.layer_importance:.3f}, weak_neuron_ratio={target.weak_neuron_ratio:.3f}, n_new={fallback_n}"
                            ),
                            candidate=is_candidate,
                        )
                    )

        should_consider_prune = (
            self.config.prune.enabled
            and epoch >= self.config.growth.warmup_epochs
            and epoch % self.config.prune.prune_interval == 0
        )
        if should_consider_prune:
            if self.config.prune.require_stagnation and not bool(report.get("stagnating_loss", False)):
                should_consider_prune = False
            if self.config.prune.skip_if_high_val_loss and bool(report.get("high_val_loss", False)):
                should_consider_prune = False

        if should_consider_prune:
            low_contribution = report["low_contribution"]
            for layer_idx, neurons in low_contribution.items():
                if not neurons:
                    continue

                layer = model.hidden_layers[layer_idx]
                active_count = int((layer.mask > 0).sum().item())
                min_active = max(1, int(round(layer.out_features * self.config.prune.min_active_fraction)))
                max_prunable = max(0, active_count - min_active)
                if max_prunable <= 0:
                    continue

                prune_cap = min(self.config.prune.max_neurons_per_event, max_prunable)
                selected = neurons[:prune_cap]
                if selected:
                    actions.append(
                        ArchitectureAction(
                            kind="prune",
                            layer_idx=layer_idx,
                            neurons=selected,
                            reason=(
                                f"low_contribution_count={len(selected)}, active_count={active_count}, "
                                f"min_active={min_active}"
                            ),
                        )
                    )

        return actions

    def _can_consider_growth(self, epoch: int, analyzer: ErrorAnalyzer) -> bool:
        if epoch < self.config.growth.warmup_epochs:
            return False
        if epoch % self.config.growth.decision_interval != 0:
            return False
        if analyzer.epoch_losses and float(analyzer.epoch_losses[-1]) < self.config.growth.min_growth_loss:
            return False
        if epoch - self._last_growth_epoch < self.config.growth.cooldown_epochs:
            return False
        return True

    def _score_layers(self, model: Any, analyzer: ErrorAnalyzer, report: dict) -> List[LayerGrowthSignal]:
        gradient_score = report.get("gradient_score", {})
        dead_neurons = report.get("dead_neurons", {})
        dead_ratio = report.get("dead_ratio", {})
        low_contribution = report.get("low_contribution", {})
        layer_importance = report.get("layer_importance", {})
        layer_underperformance = report.get("layer_underperformance", {})
        ranked_weak_neurons = report.get("ranked_weak_neurons", {})
        hard_sample_pressure = min(1.0, len(report.get("hard_sample_ids", [])) / float(max(1, self.config.growth.hard_sample_trigger)))

        layer_signals: List[LayerGrowthSignal] = []
        max_grad = max((float(v) for v in gradient_score.values()), default=0.0)
        min_grad = min((float(v) for v in gradient_score.values()), default=0.0)
        grad_span = max(max_grad - min_grad, 1e-12)

        for layer_idx, layer in enumerate(model.hidden_layers):
            grad_norm = float(gradient_score.get(layer_idx, 0.0))
            dead_count = len(dead_neurons.get(layer_idx, []))
            layer_width = max(1, layer.out_features)
            layer_dead_ratio = float(dead_ratio.get(layer_idx, 0.0))
            layer_imp = float(layer_importance.get(layer_idx, 0.0))
            underperf = float(layer_underperformance.get(layer_idx, 1.0))

            low_contrib_count = len(low_contribution.get(layer_idx, []))
            low_activation_ratio = low_contrib_count / float(layer_width)
            weak_neuron_ratio = len(ranked_weak_neurons.get(layer_idx, [])) / float(layer_width)

            normalized_grad_deficit = 1.0 - ((grad_norm - min_grad) / grad_span)
            vanishing_grad_flag = 1.0 if grad_norm <= self.config.growth.gradient_vanishing_threshold else 0.0
            dead_neuron_flag = min(1.0, layer_dead_ratio / max(self.config.growth.activation_dead_ratio_threshold, 1e-12))
            low_activation_flag = min(1.0, low_activation_ratio / max(self.config.growth.activation_dead_ratio_threshold, 1e-12))

            priority = (
                0.35 * normalized_grad_deficit
                + 0.45 * underperf
                + self.config.growth.vanishing_gradient_priority_boost * vanishing_grad_flag
                + self.config.growth.dead_neuron_priority_boost * dead_neuron_flag
                + self.config.growth.low_activation_priority_boost * low_activation_flag
                + self.config.growth.hard_sample_priority_boost * hard_sample_pressure
            )

            if dead_count > 0:
                priority += min(0.25, dead_count / float(layer_width))

            if low_activation_ratio > self.config.growth.activation_dead_ratio_threshold:
                priority += 0.1

            if layer_imp > self.config.growth.max_layer_importance_to_grow:
                priority *= 0.35
            if underperf < self.config.growth.min_underperformance_to_grow:
                priority *= 0.5
            if weak_neuron_ratio < self.config.growth.weak_neuron_fraction_for_growth:
                priority *= 0.75

            policy_bias, policy_score = self.policy.layer_bias(layer_idx)
            priority += policy_bias

            layer_signals.append(
                LayerGrowthSignal(
                    layer_idx=layer_idx,
                    priority=priority,
                    policy_bias=policy_bias,
                    policy_score=policy_score,
                    gradient_norm=grad_norm,
                    dead_neuron_count=dead_count,
                    dead_ratio=layer_dead_ratio,
                    low_activation_ratio=low_activation_ratio,
                    hard_sample_pressure=hard_sample_pressure,
                    layer_importance=layer_imp,
                    underperformance=underperf,
                    weak_neuron_ratio=weak_neuron_ratio,
                )
            )

        layer_signals.sort(key=lambda signal: signal.priority, reverse=True)
        return layer_signals

    def _compute_growth_size(self, signal: LayerGrowthSignal, model: Any) -> int:
        layer_width = model.hidden_layers[signal.layer_idx].out_features
        per_layer_cap = max(1, min(self.config.growth.max_growth_per_layer, int(layer_width * self.config.growth.max_total_growth_ratio)))

        scaled_by_pressure = int(round(self.config.growth.base_growth * (1.0 + signal.hard_sample_pressure)))
        scaled_by_pressure += int(round(self.config.growth.base_growth * signal.underperformance * signal.weak_neuron_ratio))
        if signal.dead_neuron_count > 0:
            scaled_by_pressure += max(1, signal.dead_neuron_count // 2)

        if signal.gradient_norm <= self.config.growth.gradient_vanishing_threshold:
            scaled_by_pressure += self.config.growth.min_layer_growth

        proposed = max(self.config.growth.min_layer_growth, scaled_by_pressure)
        proposed = min(proposed, self.config.growth.max_growth_per_event)
        proposed = min(proposed, per_layer_cap)

        last_growth = self._last_growth_epoch_by_layer.get(signal.layer_idx, -10_000)
        if proposed > 0 and last_growth >= 0:
            proposed = max(0, proposed)

        return proposed

    def record_growth(self, layer_idx: int, n_new: int, epoch: int) -> None:
        self._last_growth_epoch = epoch
        self._last_growth_epoch_by_layer[layer_idx] = epoch
        amount = int(n_new)
        self._growth_added_by_layer[layer_idx] = self._growth_added_by_layer.get(layer_idx, 0) + amount
        self._total_growth_added += amount
