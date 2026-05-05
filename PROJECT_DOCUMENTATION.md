# CA-SANN: Context-Aware Self-Evolving Neural Network
## Project Documentation

**Version:** 1.0  
**Technology:** Python 3.10+ · PyTorch 2.2+ · torchvision 2.17+  
**Repository:** [CA-SANN on GitHub](https://github.com/jayaprakash2207/CA-SANN--Context-Aware-Self-Evolving-Neural-Network)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Solution Overview](#3-solution-overview)
4. [System Architecture](#4-system-architecture)
5. [Module Descriptions](#5-module-descriptions)
6. [Core Algorithms](#6-core-algorithms)
7. [Experimental Design & Benchmarking](#7-experimental-design--benchmarking)
8. [Installation & Setup](#8-installation--setup)
9. [Usage Guide](#9-usage-guide)
10. [Configuration Reference](#10-configuration-reference)
11. [Inputs & Outputs](#11-inputs--outputs)
12. [Visualization Tools](#12-visualization-tools)
13. [Technical Dependencies](#13-technical-dependencies)
14. [Glossary](#14-glossary)

---

## 1. Executive Summary

CA-SANN (Context-Aware Self-Evolving Neural Network) is a research-grade deep learning framework that trains neural networks capable of **autonomously adjusting their own architecture** during the training process. Unlike conventional neural networks where the architecture (number of layers, neurons per layer) is fixed before training begins, CA-SANN networks grow and prune their neurons in real time — and, crucially, only do so when the training context justifies it.

The project benchmarks three model variants side by side:

| Variant | Dynamic Architecture | Context Gate | Description |
|---------|:-------------------:|:------------:|-------------|
| **Static** | No | — | Fixed-width baseline; standard training |
| **SANN** | Yes | No | Grows/prunes whenever the controller detects low capacity |
| **CA-SANN** | Yes | Yes | Grows only when the network is underfitting **and** the data is sufficiently difficult |

The key innovation is the **context gate**: a dual-signal mechanism that evaluates both the model's capacity state (underfitting, optimal, overfitting) and the intrinsic difficulty of the data before allowing any structural change. This prevents wasteful capacity additions and overfitting due to premature growth.

---

## 2. Problem Statement & Motivation

### The Fixed-Architecture Problem

Traditional neural network design requires engineers to pre-specify:
- Number of hidden layers
- Width (neurons/channels) of each layer
- Activation functions and normalization choices

This design is done before any training data is observed, meaning the final architecture is based on heuristics, prior experience, or expensive neural architecture search (NAS). There is no mechanism for the network to respond to what it encounters during training.

**Consequences of fixed architecture:**
- **Over-parameterized models** waste computation and risk overfitting on limited data.
- **Under-parameterized models** fail to learn complex patterns, leading to high training error (underfitting).
- Re-training from scratch is required when a chosen architecture proves inadequate.

### The SANN Limitation

Previous self-evolving approaches (SANN — Self-Adaptive Neural Networks) do allow dynamic growth, but they grow indiscriminately based on low-level signals (gradient norms, dead neuron ratios) without considering whether additional capacity is actually warranted by the data or the model's current fitting state. This can lead to:
- Uncontrolled parameter growth
- Overfitting on noisy labels
- Wasted training time from unnecessary architecture changes

### CA-SANN's Answer

CA-SANN addresses both problems by introducing a **context-aware gating layer** on top of the existing growth mechanics. Growth decisions are conditioned on:
1. **Capacity Status** — Is the model underfitting, well-fitted, or overfitting?
2. **Difficulty Score** — Is the training data genuinely challenging, or has the model plateaued on easy patterns?

Only when both signals indicate that more capacity would be beneficial does the network grow.

---

## 3. Solution Overview

### High-Level Flow

```
Training Data
     │
     ▼
Dynamic Neural Network (DynamicMLP or DynamicCNN)
     │  per-sample losses, hidden activations, gradient norms
     ▼
ErrorAnalyzer
     │  hard sample IDs, dead neuron IDs, gradient EMAs, neuron importance
     ▼
Context Estimators
  ├── CapacityEstimator  →  "underfit" | "optimal" | "overfit"
  └── DifficultyEstimator →  score ∈ [0.0, 1.0]
     │  growth_allowed = bool
     ▼
GrowthController  (+ MetaController second opinion)
     │  ArchitectureActions: grow / prune / noop
     ▼
model.grow_layer()  /  model.prune_layer_neurons()
     │  snapshot → rollback if exploration phase fails
     ▼
Updated Model → next epoch
```

### Three-Variant Comparison

At each experimental run, three separate models are trained on identical data splits, and their performance is compared at every epoch and at the end of training:

- **Static**: a fixed MLP or CNN, trained with AdamW; serves as the performance lower-bound.
- **SANN**: the same network with `context_gating=False`; all controller-approved growth events are executed.
- **CA-SANN**: the full system with `context_gating=True`; growth events are additionally filtered through the context gate.

This three-way comparison lets researchers isolate the specific contribution of context-awareness versus pure dynamic growth.

---

## 4. System Architecture

### Component Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│  DynamicMLP / DynamicCNN  (model.py)                                   │
│                                                                        │
│  Input → [HiddenLayer₀ (mask)] → [HiddenLayer₁ (mask)] → ... → Output │
│               ↕ grow_layer()         ↕ prune_layer_neurons()           │
└────────────────────┬───────────────────────────────────────────────────┘
                     │ activations, losses
                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│  ErrorAnalyzer  (analyzer.py)                                          │
│  • sample_losses    → hard_sample_ids (90th percentile)                │
│  • activation stats → dead_neuron_indices, low_contribution            │
│  • gradient EMA     → layer_gradient_score                             │
│  • neuron_importance = w_grad×grad_contrib + w_act×activation          │
└────────────────────┬───────────────────────────────────────────────────┘
                     │ analyzer.report(model)
                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Context Estimators  (context_estimators.py)                           │
│  ┌────────────────────────────┐  ┌────────────────────────────────┐    │
│  │  CapacityEstimator         │  │  DifficultyEstimator           │    │
│  │  Inputs: train_loss,       │  │  Inputs: hard_pressure,        │    │
│  │    val_loss, report        │  │    entropy_norm, grad_cv,      │    │
│  │  Output: "underfit" /      │  │    loss_tail, error_rate,      │    │
│  │    "optimal" / "overfit"   │  │    error_class_entropy         │    │
│  └────────────────────────────┘  └────────────────────────────────┘    │
└────────────────────┬───────────────────────────────────────────────────┘
                     │ growth_allowed: bool
                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│  GrowthController  (controller.py)                                     │
│  • Score each hidden layer on: grad deficit, dead ratio,               │
│    activation pressure, hard sample pressure, policy bias              │
│  • LearningPolicy: online reward signal updates layer scores           │
│  • Outputs: List[ArchitectureAction] (grow / prune / noop)             │
└────────────────────┬───────────────────────────────────────────────────┘
                     │ (filtered by context gate)
                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│  MetaController  (meta_controller.py)                                  │
│  • Independent second-opinion policy: grow / prune / noop              │
│  • Emits confidence score; scales grow_multiplier and prune_fraction   │
└────────────────────┬───────────────────────────────────────────────────┘
                     │ model.grow_layer() / model.prune_layer_neurons()
                     ▼
              Exploration-Phase Management
              snapshot → rollback / keep after eval_delay_epochs
```

---

## 5. Module Descriptions

### 5.1 `model.py` — Dynamic Neural Network Models

This module defines two dynamically-mutable neural network architectures. Both support in-place neuron/channel growth and soft pruning without interrupting training.

#### `DynamicMLPConfig` (dataclass)

Configuration for a Multilayer Perceptron.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input_dim` | int | — | Input feature dimension |
| `hidden_dims` | List[int] | — | Width of each hidden layer |
| `output_dim` | int | — | Number of output classes |
| `activation` | str | `"relu"` | Activation: `relu`, `gelu`, or `silu` |

#### `DynamicHiddenLayer`

A single MLP hidden layer consisting of a `nn.Linear`, a nonlinearity, and a **neuron mask** buffer. The mask allows individual neurons to be soft-pruned (zeroed out) without removing weights structurally.

Key methods:
- `grow(n_new)`: Expands the layer by `n_new` neurons. New weights are bootstrapped from the highest-norm existing neurons plus small Gaussian noise, encouraging useful representations from the start.
- `disable_neurons(indices)`: Sets mask and weights to zero for the specified neuron indices (soft pruning).

#### `DynamicMLP`

A complete MLP built from `DynamicHiddenLayer` instances plus a final `nn.Linear` output layer.

Key methods:
- `forward(x, return_hidden)`: Standard forward pass; optionally returns all hidden activations (required by the analyzer).
- `grow_layer(layer_idx, n_new)`: Grows a specific hidden layer and simultaneously expands the downstream layer's input dimension to maintain tensor compatibility.
- `prune_layer_neurons(layer_idx, indices)`: Soft-prunes specific neurons and zeroes downstream input columns.
- `architecture_summary()`: Returns a formatted string of current hidden widths and parameter count.

#### `DynamicCNNConfig` / `DynamicConvLayer` / `DynamicCNN`

Convolutional equivalents of the above, designed for image classification tasks (e.g., CIFAR-10). `DynamicConvLayer` wraps `nn.Conv2d` + optional `BatchNorm2d` + activation + channel mask. `DynamicCNN` chains these layers and ends with adaptive average pooling and a linear classifier.

The growth and pruning logic mirrors the MLP version but operates on convolutional filter channels rather than individual neurons.

---

### 5.2 `analyzer.py` — Error Analyzer

`ErrorAnalyzer` is the observational core of the system. It accumulates training statistics across batches and epochs and synthesizes them into structured reports that drive growth/pruning decisions.

#### Key tracked quantities

| Quantity | How tracked | Output |
|----------|------------|--------|
| Per-sample loss history | Rolling deque of length 24 per sample | `hard_sample_ids` |
| Layer activation zero-counts | Count of near-zero activations per neuron | `dead_neuron_indices` |
| Layer activation mean sums | Cumulative mean activation magnitude | `low_contribution` |
| Gradient L2 norms (EMA) | Exponential moving average per parameter group | `layer_gradient_score` |
| Per-neuron gradient contribution | Mean absolute gradient per output neuron | `neuron_importance` |

#### Key methods

- `update_batch(sample_indices, per_sample_loss, hidden_activations)`: Called every training batch. Records sample losses and accumulates activation statistics.
- `update_gradients(model)`: Called after `loss.backward()`. Records gradient norms and per-neuron gradient contributions.
- `record_epoch_loss(loss)` / `record_val_loss(loss)`: Records scalar epoch-level losses for trend analysis.
- `loss_is_stagnating()`: Returns `True` if the training loss has not improved by more than `loss_plateau_min_delta` over the last `loss_plateau_window` epochs.
- `val_loss_is_consistently_high()`: Returns `True` if validation loss has been above `high_val_loss_threshold` for the last `high_val_loss_window` epochs.
- `identify_consistently_hard_samples()`: Returns IDs of samples whose rolling-average loss exceeds the 90th percentile and that have been consistently hard across at least `hard_error_min_hits` observations.
- `dead_neuron_indices()`: Per-layer lists of neurons whose activation is near-zero ≥99.5% of the time.
- `low_contribution_neurons(model)`: Identifies neurons with low weight norm, low activation, and low gradient contribution — primary pruning candidates.
- `neuron_importance(model)`: Returns a per-layer tensor of importance scores: `w_grad × normalized_grad_contrib + w_act × normalized_activation`.
- `report(model)`: Aggregates all signals into a single dictionary used by downstream controllers.

---

### 5.3 `context_estimators.py` — Capacity & Difficulty Estimation

This module translates raw training signals into two high-level context signals used by the growth gate.

#### `CapacityEstimator`

Classifies the model's current capacity state as one of three categories:

| Status | Conditions |
|--------|-----------|
| `"overfit"` | `val_loss - train_loss ≥ overfit_gap_min` **and** `train_loss ≤ overfit_train_loss_max` |
| `"underfit"` | Both `val_loss` and `train_loss` are high **and** model is stagnating **and** there is hard-sample pressure |
| `"optimal"` | Neither of the above |

#### `DifficultyEstimator`

Computes a scalar difficulty score in [0, 1] as a weighted sum of six signals:

| Signal | Weight | Source |
|--------|--------|--------|
| Hard sample pressure | 0.30 | Fraction of samples above 90th-percentile loss threshold |
| Prediction entropy (normalized) | 0.25 | Mean predictive entropy / log(num_classes) |
| Gradient coefficient of variation | 0.20 | std(grad_norms) / mean(grad_norms) |
| Loss tail ratio | 0.15 | (Q90 − Q50) / Q50 of per-sample loss distribution |
| Validation error rate | 0.05 | Fraction misclassified on probed batches |
| Error class entropy | 0.05 | Entropy of error distribution across true classes |

A higher score indicates that the data is harder and that additional model capacity is more likely to be beneficial.

#### Helper functions

- `prediction_entropy_stats(model, data_loader, ...)`: Efficiently computes mean predictive entropy and mean max-probability over a validation sample.
- `prediction_diagnostics(model, data_loader, ...)`: Extended diagnostics including per-class error entropy, used by the difficulty estimator.

---

### 5.4 `controller.py` — Growth Controller

The `GrowthController` is the primary decision engine for architecture mutations. It consumes the analyzer report and outputs a list of `ArchitectureAction` objects.

#### `GrowthConfig` (key fields)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `warmup_epochs` | 5 | No growth decisions before this epoch |
| `decision_interval` | 2 | Minimum epochs between growth decisions |
| `min_growth_priority` | 0.75 | Score threshold for growth approval |
| `cooldown_epochs` | 3 | Minimum epochs between two growth events |
| `total_growth_budget_neurons` | 48 | Hard cap on total neurons added across the run |
| `max_model_params` | 500 000 | Hard cap on trainable parameters |
| `force_growth_after_epochs` | 4 | Fallback growth if no growth has occurred for this many epochs |
| `base_growth` | 8 | Base number of neurons added per growth event |
| `require_growth_stagnation` | True | Growth only allowed when loss is stagnating or val loss is high |

#### Layer Scoring Formula

For each hidden layer, the controller computes:

```
layer_score = loss_trend_weight    × loss_score
            + gradient_weight      × normalized_grad_deficit
            + activation_weight    × activation_pressure
            + growth_bias          × hard_sample_pressure
            + 0.2                  × vanishing_grad_flag
```

The layer with the **highest score**, **lowest importance**, and **highest underperformance** is selected as the growth target.

#### `LearningPolicy`

An online reward-based policy that tracks growth outcomes per layer and biases future decisions:
- **Reward**: +1 for any accepted growth; +20 × accuracy gain; +2.5×10⁵ × efficiency gain
- **Penalty**: −1.25 for rejected/reverted growth
- Scores are EMA-smoothed and clamped to ±4 to prevent domination

#### `PruneConfig`

Controls neuron pruning behavior. Pruning only fires when:
- `prune_interval` epochs have passed since the last pruning event
- Loss is stagnating (if `require_stagnation=True`)
- Validation loss is not consistently high (if `skip_if_high_val_loss=True`)

---

### 5.5 `meta_controller.py` — Meta-Controller

The `MetaController` provides a **second, independent opinion** on architecture decisions. It operates on the same analyzer report but uses a simpler, more conservative scoring approach.

#### Decision Logic

1. Skip if in warmup, cooldown, or if loss is already at the floor.
2. Compute `loss_score` (trend over a stagnation window), `gradient_scores`, and per-layer `activation_scores`.
3. Score each layer for growth and for pruning independently.
4. Emit:
   - `"grow"` if growth confidence ≥ `growth_confidence_threshold` (default 0.55)
   - `"prune"` if prune confidence ≥ `prune_confidence_threshold` (default 0.45)
   - `"noop"` otherwise

The MetaController's `grow_multiplier` (1.0–2.0) and `prune_fraction` (0.05–0.25) scale the intensity of the primary controller's actions. This second-opinion mechanism prevents pathological growth or pruning from occurring without dual confirmation.

---

### 5.6 `train.py` — Training Loop & Experiment Runner

This is the central module that wires all components together.

#### `TrainingConfig` (key fields — full list in Section 10)

```python
@dataclass
class TrainingConfig:
    seed: int = 13
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 8e-4
    weight_decay: float = 1e-5
    dataset_name: str = "mnist"      # mnist | fashion_mnist | cifar10
    model_arch: str = "mlp"          # mlp | cnn
    context_gating: bool = True      # True = CA-SANN; False = SANN
    difficulty_threshold: float = 0.55
    growth_budget_neurons: int = 48
    max_model_params: int = 500_000
    complexity_lambda: float = 1e-6
    train_label_noise: float = 0.15
    ...
```

#### `run_experiment` — Training Loop Summary

1. **Forward pass**: Compute per-sample `CrossEntropyLoss` (unreduced) + complexity penalty.
2. **Backward pass**: Gradient clipping at `grad_clip_norm=1.0` → `AdamW.step()`.
3. **Analyzer update**: Collect hidden activations, per-sample losses, gradient norms.
4. **Context estimation**: `CapacityEstimator` + `DifficultyEstimator` → set `growth_allowed`.
5. **Controller decision**: `GrowthController.decide()` filtered by context gate + `MetaController` vote.
6. **Architecture update**: Execute grow/prune actions; manage snapshot/rollback for the exploration phase.
7. **Checkpointing**: Save `_best` (peak val accuracy) and `_final` (last epoch) snapshots.
8. **Logging**: Emit structured log: `train_loss | val_loss | val_acc | params | growth_events | meta_action | capacity_status | difficulty_score`.

#### Complexity Penalty

To discourage unbounded growth, a super-linear complexity term is added to the loss:

```
complexity_penalty = lambda × (num_params / 1000)^growth_power
```

This means adding the 1 000th neuron costs disproportionately more than adding the 100th, creating a soft parameter budget.

#### Dataset Support

| Dataset | Normalization | Default Architecture |
|---------|--------------|---------------------|
| MNIST | mean=0.1307, std=0.3081 | MLP (784→128→64→10) |
| FashionMNIST | same as MNIST | MLP |
| CIFAR-10 | per-channel mean/std | CNN (3→32→64→128→10) |

All datasets are downloaded automatically on first run. Label noise (default 15%) is applied to the training split via `NoisyLabelDataset` to stress-test the context gate under noisy conditions.

---

### 5.7 `experiment_compare.py` — Single-Seed Comparison

Command-line entry point for running all three model variants (Static, SANN, CA-SANN) on a single dataset with a single random seed. Produces:
- `comparison_results.json` / `.csv` — tabular metrics
- `metrics.png` — 6-panel training curves
- `accuracy_vs_model_size.png` — scatter plot
- `checkpoints/` — model snapshots

---

### 5.8 `experiment_pipeline.py` — Full Benchmark Pipeline

Extends the single-seed comparison to a **multi-seed × multi-dataset** cross-product. Iterates over all combinations of `--datasets` and `--seeds`, aggregates results with mean ± std, and produces:
- `aggregate_results.json/.csv`
- `analysis.txt` — narrative research report
- `efficiency_per_100k_params.png`
- Per-seed subdirectories with individual result trees

---

### 5.9 `experiment.py` — Research Pipeline (Multi-Seed)

A higher-level research entry point that calls `experiment_compare.run_comparison` across all specified seed/dataset combinations and aggregates metrics with bootstrap-style analysis. Provides additional CLI flags for fine-grained control over exploration phases, candidate growth, and efficiency thresholds.

---

### 5.10 `visualize_3d.py` — 3D Visualization CLI

Generates static 3D matplotlib plots from experiment outputs:

| Mode | Input | Output |
|------|-------|--------|
| `metrics` | Benchmark results CSVs | 3D scatter: model_size × test_accuracy × growth_events |
| `network` | Single `.pt` checkpoint | 3D layered network graph colored by neuron importance |
| `sequence` | Directory of checkpoints | Animated growth timeline across all epochs |

---

### 5.11 `utils.py` — Utilities

Lightweight utility functions:
- `write_json` / `write_csv`: Atomic file writes with directory creation.
- `summarize_numeric(values)`: Returns `(mean, population_std)`.
- `format_table(headers, rows)`: ASCII table formatting for console output.
- `render_research_analysis(aggregate_rows, verdict_counts)`: Generates the narrative `analysis.txt` report comparing Static, SANN, and CA-SANN across seeds.

---

## 6. Core Algorithms

### 6.1 Dynamic Neuron Growth

When `grow_layer(layer_idx, n_new)` is called:

1. **Identify source neurons**: Sort existing neurons by L2 weight norm (descending).
2. **Bootstrap new weights**: Each new neuron copies the highest-norm existing neuron's weights and adds small Gaussian noise (`std=0.02`). This warm-starts new neurons with useful representations rather than random initialization.
3. **Expand downstream layer**: The input dimension of the downstream linear/conv layer is extended by `n_new` columns, using the same copy-and-perturb initialization from high-norm input connections.
4. **Update masks**: The neuron mask is extended with ones for all new neurons.
5. **Update config**: `config.hidden_dims[layer_idx]` is incremented.

This approach differs from standard random initialization, which can stall training while new neurons learn useful features from scratch.

### 6.2 Soft Pruning

When `prune_layer_neurons(layer_idx, indices)` is called:

1. The **neuron mask** is set to 0 for the specified indices.
2. Corresponding **weight rows and biases** are zeroed.
3. **Downstream input columns** are zeroed (so zeroed neurons do not contribute to the next layer).
4. The structural size of the network does not change — pruned neurons remain as dead weight but do not influence the forward or backward pass through the mask.

Soft pruning is preferred over hard pruning (structural removal) because it avoids the complexity of reshaping weight tensors and optimizer states, making it compatible with in-place growth at arbitrary later epochs.

### 6.3 Context Gating

```python
capacity_status  = CapacityEstimator.estimate(train_loss, val_loss, report)
difficulty_score = DifficultyEstimator.estimate(report, entropy_norm, error_rate, ...)

if capacity_status == "underfit":
    growth_allowed = True                              # always grow
elif capacity_status == "optimal":
    growth_allowed = difficulty_score > difficulty_threshold   # grow only if task is hard
elif capacity_status == "overfit":
    growth_allowed = False                             # never grow
```

The `difficulty_threshold` (default 0.55) is the key hyperparameter controlling how conservative the CA-SANN is. A higher threshold makes growth rarer; a lower threshold approaches SANN behavior.

### 6.4 Exploration-Phase Rollback

During the first `exploration_phase_epochs` epochs, speculative growth events are handled as follows:

1. Before applying a growth event, save a **model snapshot** (deep copy of the state dict).
2. Record the current validation efficiency (`accuracy / sqrt(params)`).
3. Train for `exploration_eval_delay_epochs` epochs.
4. **Evaluate**:
   - If `current_efficiency ≥ snapshot_efficiency` and accuracy did not drop: **keep** the growth.
   - Otherwise: **revert** to the snapshot and enter a `no_growth_phase`.

This allows the network to speculatively grow without permanently committing to changes that hurt performance.

### 6.5 Online Learning Policy

The `LearningPolicy` class maintains a per-layer score updated after each growth event outcome:

```
reward = +1.0 + 20.0×acc_gain + 2.5e5×efficiency_gain   (if accepted)
reward = -1.25 + 5.0×acc_gain + 1.0e5×efficiency_gain   (if rejected)

score = score × decay + lr × reward     (EMA update, clamped to [-4, +4])
```

This score biases the layer priority in the controller's growth scoring, creating a feedback loop where layers that historically benefited from growth are prioritized in future decisions.

---

## 7. Experimental Design & Benchmarking

### Setup

- **Datasets**: MNIST, FashionMNIST, CIFAR-10 (10 classes each; auto-downloaded)
- **Random seeds**: 13, 23, 33, 43, 53 (5 seeds for statistical robustness)
- **Training epochs**: 15 (configurable)
- **Training samples**: 20 000; Validation: 5 000; Test: 5 000 (subsampled for speed)
- **Label noise**: 15% random label corruption on the training split
- **Optimizer**: AdamW (lr=8×10⁻⁴, weight_decay=10⁻⁵)
- **Initial architectures**: MLP (784→128→64→10) for MNIST/FashionMNIST; CNN (3→32→64→128→10) for CIFAR-10

### Metrics Tracked Per Epoch

| Metric | Description |
|--------|-------------|
| `train_loss` | Mean per-sample cross-entropy + complexity penalty |
| `val_loss` | Mean cross-entropy on validation set |
| `val_accuracy` | Fraction correctly classified |
| `model_size` | Trainable parameter count |
| `growth_events` | Count of accepted growth events |
| `efficiency` | `val_accuracy / sqrt(model_size)` |
| `capacity_status` | `underfit` / `optimal` / `overfit` |
| `difficulty_score` | Scalar in [0, 1] |
| `meta_action` | MetaController's vote: `grow` / `prune` / `noop` |

### Aggregate Reporting

After all seeds complete, the pipeline computes:
- Mean ± population std for all numeric metrics
- Verdict counts: how many seeds each variant won on test accuracy
- CA-SANN vs Static and SANN vs Static deltas
- Model size ratio: `SANN_params / Static_params`

---

## 8. Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip
- (Optional) NVIDIA GPU with CUDA for faster training

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/jayaprakash2207/CA-SANN--Context-Aware-Self-Evolving-Neural-Network.git
cd CA-SANN--Context-Aware-Self-Evolving-Neural-Network

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt` contents:**
```
torch>=2.2.0
matplotlib>=3.8.0
torchvision>=0.17.0
```

For visualization notebooks, also install:
```bash
pip install pandas seaborn plotly jupyter
```

CUDA is auto-detected. If available, all tensor operations run on GPU automatically.

---

## 9. Usage Guide

### 9.1 Quick Start — Single-Seed Comparison

```bash
python experiment_compare.py \
  --dataset mnist \
  --epochs 10 \
  --output-dir runs/quickstart
```

Trains Static, SANN, and CA-SANN on MNIST and writes all outputs to `runs/quickstart/`.

### 9.2 Full Benchmark Pipeline

```bash
python experiment_pipeline.py \
  --datasets mnist fashion_mnist cifar10 \
  --epochs 15 \
  --seeds 13 23 33 43 53 \
  --train-samples 20000 \
  --val-samples 5000 \
  --test-samples 5000 \
  --output-dir runs/ca_sann_benchmark \
  --difficulty-threshold 0.55
```

### 9.3 Programmatic API

```python
from train import TrainingConfig, run_experiment, make_datasets, IndexedImageDataset, setup_logging
from torch.utils.data import DataLoader

setup_logging()

config = TrainingConfig(
    dataset_name="mnist",
    epochs=10,
    context_gating=True,    # CA-SANN mode
    difficulty_threshold=0.55,
)

train_ds, val_ds, test_ds = make_datasets(config)

train_loader = DataLoader(IndexedImageDataset(train_ds), batch_size=256, shuffle=True)
val_loader   = DataLoader(IndexedImageDataset(val_ds),   batch_size=256)
test_loader  = DataLoader(IndexedImageDataset(test_ds),  batch_size=256)

result = run_experiment(
    experiment_name="ca_sann_run_1",
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    dynamic=True,    # False = Static baseline
)

print(f"Test accuracy: {result.test_accuracy:.4f}")
print(f"Final parameters: {result.final_model_size:,}")
print(f"Growth events: {result.growth_event_count}")
```

### 9.4 Generating 3D Plots

```bash
# Efficiency scatter from benchmark results
python visualize_3d.py --mode metrics \
  --input-root runs/ca_sann_benchmark --output-dir runs/ca_sann_benchmark

# Network architecture view from a checkpoint
python visualize_3d.py --mode network \
  --checkpoint runs/quickstart/checkpoints/sann_epoch010_final.pt \
  --output-dir runs/quickstart

# Growth timeline across all checkpoints
python visualize_3d.py --mode sequence \
  --checkpoint-dir runs/quickstart/checkpoints \
  --output-dir runs/quickstart
```

### 9.5 Interactive Browser Visualizer

```bash
# Serve from repo root to avoid CORS issues
python -m http.server 8080
# Open http://localhost:8080 in a browser
```

The Three.js-based visualizer (`index.html`) reads `data.json` and provides:
- Epoch scrubber to step through training
- Model switcher (Static / SANN / CA-SANN)
- Rotatable 3D network scene with animated neuron addition/removal

---

## 10. Configuration Reference

All fields of `TrainingConfig` with defaults and descriptions:

### Core Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | `13` | Random seed for reproducibility |
| `epochs` | `5` | Number of training epochs |
| `batch_size` | `256` | Mini-batch size |
| `learning_rate` | `8e-4` | AdamW initial learning rate |
| `weight_decay` | `1e-5` | AdamW weight decay (L2 regularization) |
| `grad_clip_norm` | `1.0` | Maximum gradient norm for clipping |

### Dataset & Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_name` | `"mnist"` | `mnist` / `fashion_mnist` / `cifar10` |
| `model_arch` | `"mlp"` | `mlp` / `cnn` |
| `input_dim` | `784` | MLP input dimension (28×28 for MNIST) |
| `num_classes` | `10` | Number of output classes |
| `hidden_dims` | `(128, 64)` | Initial MLP hidden layer widths |
| `cnn_in_channels` | `3` | CNN input channels |
| `cnn_channels` | `(32, 64, 128)` | Initial CNN filter counts |
| `cnn_kernel_size` | `3` | CNN kernel size |
| `cnn_use_batchnorm` | `True` | Enable BatchNorm in CNN layers |
| `train_label_noise` | `0.15` | Fraction of training labels randomly corrupted |
| `train_samples` | `10 000` | Training set size (subset) |
| `val_samples` | `2 000` | Validation set size |
| `test_samples` | `5 000` | Test set size |

### Context Gating (CA-SANN specific)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `context_gating` | `True` | `True` = CA-SANN; `False` = SANN |
| `difficulty_threshold` | `0.55` | Min difficulty score for growth when capacity is "optimal" |

### Growth & Pruning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `growth_budget_neurons` | `48` | Maximum total neurons added across all layers per run |
| `max_model_params` | `500 000` | Hard cap on trainable parameter count |
| `complexity_lambda` | `1e-6` | Complexity penalty coefficient in loss |
| `complexity_growth_power` | `1.35` | Super-linear exponent for complexity penalty |
| `enable_candidate_growth` | `True` | Allow weaker growth signals to trigger smaller growth events |
| `candidate_growth_neurons` | `2` | Neurons added per candidate growth event |

### Post-Growth Recovery

| Parameter | Default | Description |
|-----------|---------|-------------|
| `post_growth_lr_scale` | `0.5` | LR multiplier immediately after a growth event |
| `post_growth_recovery_epochs` | `2` | Epochs to ramp LR back + freeze new neuron gradients |

### Exploration Phase

| Parameter | Default | Description |
|-----------|---------|-------------|
| `exploration_budget_events` | `2` | Maximum speculative growth events per run |
| `exploration_phase_epochs` | `6` | Epoch window during which exploration is active |
| `exploration_eval_delay_epochs` | `2` | Epochs to wait before evaluating a speculative growth event |
| `efficiency_drop_tolerance_early` | `2.0e-7` | Efficiency drop forgiven in early exploration |
| `efficiency_drop_tolerance_late` | `5.0e-8` | Efficiency drop forgiven in late exploration |
| `efficiency_decline_patience` | `3` | Epochs of declining efficiency before rollback |

### Output & Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_dir` | `"runs/mnist_sann_experiment"` | Root output directory |
| `checkpoint_dirname` | `"checkpoints"` | Subdirectory for `.pt` files |
| `plot_filename` | `"metrics.png"` | Training curve plot filename |
| `summary_filename` | `"summary.json"` | Per-epoch metric time-series filename |
| `debug_log_details` | `True` | Log extended per-epoch architecture details |

---

## 11. Inputs & Outputs

### Inputs

| Input | Source | Format |
|-------|--------|--------|
| Training images | MNIST / FashionMNIST / CIFAR-10 | Normalized `torch.Tensor` |
| Training labels | Same datasets | `torch.long` integers |
| Random seed | CLI / config | Integer |
| Architecture config | `TrainingConfig` | Python dataclass |

### Output Files

| File | Location | Description |
|------|----------|-------------|
| `checkpoints/*_best.pt` | `output_dir/checkpoints/` | Checkpoint at peak validation accuracy |
| `checkpoints/*_final.pt` | `output_dir/checkpoints/` | Checkpoint at final epoch |
| `metrics.png` | `output_dir/` | 6-panel training chart (loss, accuracy, model size, efficiency, difficulty, capacity status) |
| `summary.json` | `output_dir/` | Full per-epoch metric time-series |
| `comparison_results.json` | `output_dir/` | Static / SANN / CA-SANN comparison per seed |
| `comparison_results.csv` | `output_dir/` | Tabular version of comparison results |
| `aggregate_results.json` | Benchmark root | Multi-seed mean ± std aggregates |
| `aggregate_results.csv` | Benchmark root | Tabular version of aggregates |
| `analysis.txt` | Benchmark root | Narrative research report |
| `accuracy_vs_model_size.png` | `output_dir/` | Scatter: test accuracy vs parameter count |
| `efficiency_per_100k_params.png` | Benchmark root | Bar: accuracy per 100 000 parameters |
| `*_3d.png` | `output_dir/` | 3D matplotlib visualizations |

### Checkpoint Format

```python
{
    "epoch": int,                    # epoch number
    "experiment_name": str,          # e.g. "ca_sann"
    "model_state_dict": dict,        # full state dict (weights, masks, BN stats)
    "optimizer_state_dict": dict,    # AdamW optimizer state
    "model_size": int,               # trainable parameter count at this epoch
    "metrics": {                     # all per-epoch metric lists up to this epoch
        "train_loss": [...],
        "val_loss": [...],
        "val_accuracy": [...],
        "model_size": [...],
        ...
    }
}
```

---

## 12. Visualization Tools

### 12.1 Training Metrics Plot (`metrics.png`)

A 6-panel matplotlib figure generated automatically at the end of each experiment:

1. **Train / Val Loss** over epochs
2. **Val Accuracy** over epochs
3. **Model Size** (trainable parameters) over epochs — shows growth events as steps
4. **Efficiency** (accuracy / √params) + growth event markers
5. **Difficulty Score** over epochs
6. **Capacity Status** (encoded as: 0=optimal, 1=underfit, 2=overfit)

### 12.2 3D Matplotlib Visualizer (`visualize_3d.py`)

Three operating modes:
- **`metrics`**: Scatter plot in model_size × accuracy × growth_events space, one point per seed/model combination.
- **`network`**: Layered sphere graph where each layer is drawn as a ring of spheres; sphere size and color encode neuron importance.
- **`sequence`**: Animated timeline showing how the network structure changes epoch by epoch.

### 12.3 Interactive Three.js Visualizer (`index.html`)

A self-contained web application (no server-side component) that reads `data.json` and renders:
- A 3D scene of the network structure with animated neurons
- Controls for epoch scrubbing, model selection, rotation speed
- Color-coded layers showing growth/pruning events in real time

### 12.4 Jupyter Notebooks

| Notebook | Content |
|----------|---------|
| `SANN_Visualization_Pipeline.ipynb` | Loss curves, accuracy, model size evolution, growth event breakdowns, efficiency comparison charts with Seaborn/Matplotlib |
| `SANN_3D_Visualization.ipynb` | Interactive Plotly 3D animation of network structure over epochs; requires `plotly` |

---

## 13. Technical Dependencies

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| `torch` | 2.2.0 | Core deep learning framework (autograd, layers, optimizers) |
| `torchvision` | 0.17.0 | Dataset loaders (MNIST, FashionMNIST, CIFAR-10) and transforms |
| `matplotlib` | 3.8.0 | Training metrics plots and 3D visualizations |
| `pandas` | any | (Notebooks only) DataFrame manipulation |
| `seaborn` | any | (Notebooks only) Statistical visualization |
| `plotly` | any | (Notebooks only) Interactive 3D animation |
| `jupyter` | any | (Notebooks only) Notebook execution environment |

Python standard library modules used: `argparse`, `collections`, `copy`, `csv`, `dataclasses`, `json`, `logging`, `math`, `pathlib`, `random`, `statistics`, `typing`.

**Hardware Requirements:**
- Minimum: Any modern CPU (training will be slower)
- Recommended: NVIDIA GPU (CUDA 11.8+) for practical experiment runtimes
- RAM: 4 GB minimum; 8 GB recommended for multi-seed runs

---

## 14. Glossary

| Term | Definition |
|------|-----------|
| **SANN** | Self-Adaptive/Self-Evolving Neural Network — a network capable of modifying its own architecture during training |
| **CA-SANN** | Context-Aware SANN — SANN with a context gate that restricts growth to when it is justified by capacity and difficulty signals |
| **Dynamic architecture** | An architecture whose structure (neuron/channel count) can change during training |
| **Soft pruning** | Zeroing neuron weights and mask values without structurally removing them from the network |
| **Capacity status** | Classification of the model's fitting state: `underfit`, `optimal`, or `overfit` |
| **Difficulty score** | A scalar in [0,1] measuring the intrinsic difficulty of the current training context |
| **Context gate** | The decision logic that combines capacity status and difficulty score to allow or block growth |
| **Hard sample** | A training sample whose rolling-average loss persistently exceeds the 90th-percentile threshold |
| **Dead neuron** | A neuron whose output is near-zero (< 10⁻⁸) on more than 99.5% of observed samples |
| **Low contribution neuron** | A neuron with low weight norm, low mean activation, and low gradient contribution — a primary pruning candidate |
| **Gradient EMA** | Exponential moving average of gradient L2 norms, used to estimate per-layer learning signal over time |
| **Neuron importance** | A weighted combination of normalized gradient contribution and normalized mean activation |
| **Exploration phase** | The early training window during which speculative growth events are tried with rollback capability |
| **Learning policy** | An online reward-based policy that tracks per-layer growth outcomes and biases future growth targeting |
| **Complexity penalty** | A super-linear regularization term added to the loss that grows with the number of trainable parameters |
| **Efficiency** | `val_accuracy / sqrt(trainable_parameters)` — a combined measure of accuracy and compactness |
| **MetaController** | A secondary controller that provides an independent confidence-scored vote on grow/prune/noop decisions |
| **bootstrap initialization** | Initializing new neurons by copying weights from high-norm existing neurons plus small noise, rather than random initialization |
| **AdamW** | Adam optimizer with decoupled weight decay; the default optimizer used throughout |

---

*Document prepared from source code analysis of the CA-SANN repository.*  
*All descriptions reflect the implementation as found in the repository codebase.*
