<div align="center">

# 🧠 CA-SANN
### Context-Aware Self-Evolving Neural Network

*A neural network that grows, prunes, and governs itself — guided by real-time signals about capacity and data difficulty.*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: TODO](https://img.shields.io/badge/License-TODO-lightgrey)](#license)
[![Stars](https://img.shields.io/github/stars/jayaprakash2207/CA-SANN--Context-Aware-Self-Evolving-Neural-Network?style=social)](https://github.com/jayaprakash2207/CA-SANN--Context-Aware-Self-Evolving-Neural-Network)

</div>

---

## 📖 Table of Contents

1. [What is CA-SANN?](#-what-is-ca-sann)
2. [Key Features](#-key-features)
3. [Architecture Overview](#-architecture-overview)
4. [Quickstart](#-quickstart)
5. [Detailed Usage](#-detailed-usage)
6. [Project Structure](#-project-structure)
7. [Key Algorithms](#-key-algorithms)
8. [Inputs & Outputs](#-inputs--outputs)
9. [Configuration Reference](#-configuration-reference)
10. [Results & Reproduction](#-results--reproduction)
11. [Roadmap](#-roadmap)
12. [Contributing](#-contributing)
13. [License](#-license)
14. [Citation](#-citation)
15. [Acknowledgments](#-acknowledgments)

---

## 🤔 What is CA-SANN?

Most neural networks are trained with a **fixed architecture** decided before any data is seen. CA-SANN challenges that assumption.

It trains a network that **grows and prunes its own layers** during training — and, critically, only does so when *context signals* justify it.

The framework benchmarks **three variants** head-to-head:

| Variant | Grows? | Context Gate? | Description |
|---------|:------:|:-------------:|-------------|
| **Static** | ✗ | — | Fixed-width baseline network |
| **SANN** | ✓ | ✗ | Grows whenever the controller detects low capacity |
| **CA-SANN** | ✓ | ✓ | Grows only when the network is truly underfitting *and* the data is hard enough |

The **context-aware gate** (the *CA* in CA-SANN) fuses two real-time signals every epoch:

1. **Capacity Status** — `underfit` / `optimal` / `overfit` — derived from the train/val loss gap, validation error rate, prediction entropy, and the pressure of persistently hard training samples.
2. **Difficulty Score** — a scalar in [0, 1] weighting hard-sample pressure, prediction entropy, gradient coefficient of variation, loss tail ratio, and per-class error entropy.

Growth is **allowed** when the model is underfitting, **gated** by a threshold when optimal, and **blocked entirely** when overfitting.

---

## ✨ Key Features

| Feature | Where |
|---------|-------|
| 🔧 **Dynamic architecture** — neurons/filters added or soft-pruned in-place with no training interruption | `model.py` |
| 🌱 **Smart growth initialization** — new neurons bootstrapped from highest-norm existing units + Gaussian noise | `model.py` |
| 🧭 **Context-aware growth gate** — prevents wasted capacity additions when the model is already well-fitted | `train.py` |
| ⏪ **Exploration-phase rollback** — early growth events are speculative; model reverts if efficiency doesn't improve | `train.py` |
| 📚 **Online learning policy** — tracks per-layer growth outcomes, biasing future decisions toward historically beneficial layers | `controller.py` |
| 🎛️ **Meta-controller** — a second-opinion policy votes `grow` / `prune` / `noop` with a confidence score | `meta_controller.py` |
| 💰 **Complexity regularization** — super-linear penalty discourages unnecessary growth directly in the loss | `train.py` |
| 📊 **Multi-seed benchmarking** — 3 datasets × 5 seeds × 3 model variants, with aggregate mean ± std reporting | `experiment_pipeline.py` |
| 🌐 **Interactive 3D visualizer** — browser-based Three.js viewer + Plotly notebook for animated epoch-by-epoch inspection | `index.html`, notebooks |

---

## 🏗️ Architecture Overview

```
Input Batch
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  DynamicMLP / DynamicCNN                                    │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐           │
│  │ Hidden L0  │──▶│ Hidden L1  │──▶│   ...      │──▶ Output │
│  │ (+ mask)   │   │ (+ mask)   │   │ (+ mask)   │           │
│  └─────┬──────┘   └─────┬──────┘   └────────────┘           │
│    grow_layer()    prune_layer_neurons()                     │
└─────────────────────────────────────────────────────────────┘
         │  hidden activations + per-sample losses
         ▼
┌─────────────────────────────────────────────────────────────┐
│  ErrorAnalyzer                                              │
│  • Per-sample loss history  →  hard sample IDs              │
│  • Activation zero counts   →  dead neuron IDs              │
│  • Gradient EMA per layer   →  vanishing gradient flags     │
│  • Neuron importance = w·grad_contrib + w·activation        │
└────────────────────┬────────────────────────────────────────┘
                     │  analyzer.report(model)
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  Context Estimators                                          │
│  ┌─────────────────────────┐  ┌──────────────────────────┐   │
│  │  CapacityEstimator      │  │  DifficultyEstimator     │   │
│  │  → "underfit"           │  │  → score ∈ [0, 1]        │   │
│  │    "optimal"            │  │    (entropy, grad_cv,    │   │
│  │    "overfit"            │  │     hard_pressure, ...)  │   │
│  └─────────────────────────┘  └──────────────────────────┘   │
└────────────────────┬─────────────────────────────────────────┘
                     │  growth_allowed = bool
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  GrowthController                                           │
│  • Score each layer: grad deficit + dead ratio +            │
│    activation pressure + hard pressure + policy bias        │
│  • Emit grow / prune / noop ArchitectureActions             │
│  • LearningPolicy: online reward signal per layer           │
└────────────────────┬────────────────────────────────────────┘
                     │  (filtered by context gate)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  MetaController                                             │
│  • Second-opinion grow / prune / noop with confidence       │
│  • Scales grow_multiplier, sets prune_fraction              │
└────────────────────┬────────────────────────────────────────┘
                     │
            model.grow_layer()
            model.prune_layer_neurons()
            → snapshot / rollback (exploration phase)
            → post-growth LR warmup + freeze masks
```

**Training loop summary** (`run_experiment`, `train.py`):

1. Forward pass → per-sample `CrossEntropyLoss` (unreduced) + complexity penalty
2. Backward → gradient clipping → `AdamW.step()`
3. Analyzer collects activations, per-sample losses, gradient norms
4. `CapacityEstimator` + `DifficultyEstimator` → set `growth_allowed`
5. `GrowthController.decide()` → filtered by context gate + `MetaController`
6. Execute grow/prune → manage snapshot/rollback for exploration phase
7. Log: `train_loss | val_loss | val_acc | params | growth_events | meta_action`

---

## 🚀 Quickstart

### Prerequisites

```bash
git clone https://github.com/jayaprakash2207/CA-SANN--Context-Aware-Self-Evolving-Neural-Network.git
cd CA-SANN--Context-Aware-Self-Evolving-Neural-Network
pip install -r requirements.txt
# torch>=2.2.0  torchvision>=0.17.0  matplotlib>=3.8.0
```

> **GPU**: Optional but recommended. CUDA is auto-detected. Datasets download automatically to `data/`.

### Run a quick comparison (MNIST, 1 seed)

```bash
python experiment_compare.py \
  --dataset mnist \
  --epochs 10 \
  --output-dir runs/quickstart
```

This trains **Static**, **SANN**, and **CA-SANN** on 20 000 MNIST samples and writes results, plots, and checkpoints to `runs/quickstart/`.

---

## 📋 Detailed Usage

### 1 · Single-seed comparison (`experiment_compare.py`)

Runs all three variants on one dataset, one seed.

```bash
python experiment_compare.py \
  --dataset mnist          # mnist | fashion_mnist | cifar10
  --epochs 15 \
  --batch-size 256 \
  --hidden-dims 128,64 \   # initial hidden layer widths
  --difficulty-threshold 0.55 \
  --train-samples 20000 \
  --val-samples 5000 \
  --test-samples 5000 \
  --seed 13 \
  --output-dir runs/my_run
```

**Output files:**
```
runs/my_run/
  comparison_results.json     ← full per-epoch time-series + final metrics
  comparison_results.csv      ← tabular summary
  accuracy_vs_model_size.png  ← scatter plot
  metrics.png                 ← 6-panel training curves
  checkpoints/
    static_epoch015_final.pt
    sann_epoch015_final.pt
    ca_sann_epoch015_final.pt
```

---

### 2 · Full benchmark pipeline (`experiment_pipeline.py`)

Multi-seed × multi-dataset cross-product (the canonical research benchmark).

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

**Aggregate output:**
```
runs/ca_sann_benchmark/
  aggregate_results.{json,csv}    ← mean ± std per model per dataset
  analysis.txt                    ← narrative research report
  accuracy_vs_model_size.png
  efficiency_per_100k_params.png
  <dataset>/seed_<N>/             ← per-seed result trees
```

---

### 3 · Programmatic API (`train.py`)

```python
from train import TrainingConfig, run_experiment, make_datasets, IndexedImageDataset, setup_logging
from torch.utils.data import DataLoader

setup_logging()
config = TrainingConfig(dataset_name="mnist", epochs=10)
train_ds, val_ds, test_ds = make_datasets(config)

train_loader = DataLoader(IndexedImageDataset(train_ds), batch_size=256, shuffle=True)
val_loader   = DataLoader(IndexedImageDataset(val_ds),   batch_size=256)
test_loader  = DataLoader(IndexedImageDataset(test_ds),  batch_size=256)

result = run_experiment(
    experiment_name="ca_sann",
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    dynamic=True,      # False = Static baseline
)
print(f"Test accuracy: {result.test_accuracy:.4f} | Params: {result.final_model_size:,}")
```

---

### 4 · Visualization

#### 3D matplotlib plots (`visualize_3d.py`)

```bash
# Efficiency scatter plots from benchmark results
python visualize_3d.py --mode metrics \
  --input-root runs/ca_sann_benchmark --output-dir runs/ca_sann_benchmark

# 3D architecture view from a checkpoint
python visualize_3d.py --mode network \
  --checkpoint runs/my_run/checkpoints/sann_epoch015_final.pt \
  --output-dir runs/my_run

# Growth timeline across all checkpoints
python visualize_3d.py --mode sequence \
  --checkpoint-dir runs/my_run/checkpoints --output-dir runs/my_run
```

#### Interactive browser visualizer

```bash
python -m http.server 8080   # avoids CORS issues
# open http://localhost:8080
```

The Three.js visualizer (`index.html`) reads `data.json` and lets you scrub through epochs, switch between Static / SANN / CA-SANN, and rotate the 3D network scene.

#### Jupyter notebooks

| Notebook | Purpose |
|----------|---------|
| [`SANN_Visualization_Pipeline.ipynb`](SANN_Visualization_Pipeline.ipynb) | Loss curves, accuracy, model size evolution, growth event breakdowns, efficiency charts |
| [`SANN_3D_Visualization.ipynb`](SANN_3D_Visualization.ipynb) | Interactive Plotly 3D animation of network structure over epochs |

```bash
pip install pandas seaborn plotly   # additional notebook dependencies
jupyter notebook
```

---

## 📁 Project Structure

```
CA-SANN/
│
├── 🧩 Core modules
│   ├── model.py                  # DynamicMLP, DynamicCNN — grow_layer, prune_layer_neurons
│   ├── analyzer.py               # ErrorAnalyzer — hard samples, dead neurons, gradient EMA
│   ├── context_estimators.py     # CapacityEstimator, DifficultyEstimator
│   ├── controller.py             # GrowthController, LearningPolicy, ArchitectureAction
│   └── meta_controller.py        # MetaController — second-opinion grow/prune/noop
│
├── 🏃 Entry points
│   ├── train.py                  # TrainingConfig, run_experiment — main training loop
│   ├── experiment_compare.py     # CLI: single-seed Static vs SANN vs CA-SANN
│   ├── experiment.py             # CLI: multi-seed, multi-dataset pipeline (research)
│   └── experiment_pipeline.py   # CLI: multi-seed, multi-dataset pipeline (benchmark)
│
├── 📊 Visualization
│   ├── visualize_3d.py           # CLI: matplotlib 3D plots from checkpoints/results
│   ├── index.html                # Interactive Three.js network visualizer
│   ├── script.js                 # Three.js scene logic
│   ├── style.css                 # Visualizer styles
│   ├── data.json                 # Sample CIFAR-10 metrics for web viewer
│   ├── SANN_Visualization_Pipeline.ipynb
│   └── SANN_3D_Visualization.ipynb
│
├── 🛠️ Utilities
│   ├── utils.py                  # write_json, write_csv, format_table, render_research_analysis
│   ├── requirements.txt          # torch, torchvision, matplotlib
│   └── vendor/three/             # Bundled Three.js + postprocessing shaders
│
└── runs/                         # (git-ignored) all experiment outputs
    └── <run_name>/
        ├── checkpoints/          # .pt model snapshots (best + final)
        ├── metrics.png           # 6-panel training chart
        └── summary.json          # per-epoch metric time-series
```

---

## 🔬 Key Algorithms

### Dynamic architecture mutation (`model.py`)

**`grow_layer(layer_idx, n_new)`**
- Extends the target hidden layer by `n_new` neurons/channels.
- New weights are initialized from the **highest-norm existing rows** plus Gaussian noise (`init_std=0.02`), bootstrapping useful representations while adding diversity.
- The downstream layer's input dimension is expanded simultaneously — no broken forward pass.

**`prune_layer_neurons(layer_idx, indices)`**
- Sets the neuron mask to 0 for selected indices and zeros corresponding weight rows and downstream input columns.
- Pruned neurons remain structurally present (soft pruning) — their gradient is zeroed by the mask.

---

### ErrorAnalyzer signals (`analyzer.py`)

| Signal | Source | Used for |
|--------|--------|----------|
| `hard_sample_ids` | Per-sample loss history; 90th-percentile threshold | Growth trigger |
| `dead_neuron_indices` | Activation zero-fraction ≥ 0.995 | Pruning target |
| `low_contribution` | Low weight norm + low activation + low grad contrib | Pruning target |
| `gradient_score` per layer | EMA of L2 gradient norm | Growth layer scoring |
| `neuron_importance` | `w_grad × grad_contrib + w_act × activation` | Layer importance ranking |

---

### Context gating (`train.py`)

```python
capacity_status  = CapacityEstimator.estimate(train_loss, val_loss, report)
difficulty_score = DifficultyEstimator.estimate(report, entropy_norm, error_rate, ...)

if capacity_status == "underfit":
    growth_allowed = True               # always grow — need more capacity
elif capacity_status == "optimal":
    growth_allowed = difficulty_score > difficulty_threshold   # grow only if task is hard
elif capacity_status == "overfit":
    growth_allowed = False              # never grow — adding capacity makes things worse
```

---

### Growth decision scoring (`controller.py`)

```
layer_score = loss_trend_weight    × loss_score
            + gradient_weight      × normalized_grad_deficit
            + activation_weight    × activation_pressure
            + growth_bias          × hard_sample_pressure
            + 0.2                  × vanishing_grad_flag
```

The layer with the **highest score**, **lowest importance**, and **highest underperformance** is selected as the growth target.

---

### Exploration-phase rollback (`train.py`)

During the first `exploration_phase_epochs` epochs, non-immediately-safe growth events are queued as *explorations* alongside a **model snapshot**. After `exploration_eval_delay_epochs` epochs:

- **Keep** if `current_efficiency ≥ baseline_efficiency` and accuracy improved.
- **Revert** to snapshot otherwise → enter `no_growth_phase`.

This allows the network to speculatively grow and automatically undo mistakes before they compound.

---

## 📥 Inputs & Outputs

### Supported datasets

| Dataset | Auto-download | Default arch | Classes | Input size |
|---------|:---:|:---:|:---:|:---:|
| MNIST | ✓ | MLP | 10 | 784 |
| FashionMNIST | ✓ | MLP | 10 | 784 |
| CIFAR-10 | ✓ | CNN | 10 | 3 × 32 × 32 |

> **Label noise**: 15 % of training labels are randomly flipped by default (`--train-label-noise 0.15`) to make the task harder and stress-test the context gate. Set to `0.0` to disable.

### Checkpoint format (`.pt`)

```python
{
    "epoch": int,
    "experiment_name": str,
    "model_state_dict": dict,     # full state dict incl. masks and BN running stats
    "optimizer_state_dict": dict,
    "model_size": int,            # trainable parameter count
    "metrics": { ... }            # all per-epoch metric lists
}
```

### Output artifacts

| File | Description |
|------|-------------|
| `checkpoints/*.pt` | `_best` (peak val accuracy) and `_final` (last epoch) snapshots |
| `metrics.png` | 6-panel chart: loss, accuracy, model size, efficiency+growth events, difficulty score, capacity status |
| `summary.json` | Full per-epoch metric time-series |
| `comparison_results.{json,csv}` | Static / SANN / CA-SANN comparison table per seed |
| `aggregate_results.{json,csv}` | Multi-seed mean ± std aggregates |
| `analysis.txt` | Narrative research report with verdict counts |
| `accuracy_vs_model_size.png` | Test accuracy vs parameter count scatter |
| `efficiency_per_100k_params.png` | Accuracy / 100 k parameters bar chart |
| `*_3d.png` | 3D matplotlib visualizations |

---

## ⚙️ Configuration Reference

Key fields of `TrainingConfig` (`train.py`). All parameters are also exposed as CLI flags.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | `5` | Training epochs |
| `batch_size` | `256` | Mini-batch size |
| `learning_rate` | `8e-4` | AdamW LR |
| `weight_decay` | `1e-5` | AdamW weight decay |
| `hidden_dims` | `(128, 64)` | Initial MLP hidden widths |
| `cnn_channels` | `(32, 64, 128)` | Initial CNN channel widths |
| `dataset_name` | `"mnist"` | `mnist` / `fashion_mnist` / `cifar10` |
| `model_arch` | `"mlp"` | `mlp` / `cnn` |
| `train_label_noise` | `0.15` | Fraction of training labels corrupted |
| **`context_gating`** | `True` | Enable CA-SANN gate (`False` = SANN mode) |
| **`difficulty_threshold`** | `0.55` | Min difficulty score for growth when optimal |
| `growth_budget_neurons` | `48` | Max total neurons added across all layers |
| `max_model_params` | `500 000` | Hard cap on trainable parameters |
| `complexity_lambda` | `1e-6` | Complexity penalty coefficient |
| `complexity_growth_power` | `1.35` | Super-linear exponent for complexity penalty |
| `grad_clip_norm` | `1.0` | Max gradient norm for clipping |
| `post_growth_lr_scale` | `0.5` | LR multiplier immediately after growth |
| `post_growth_recovery_epochs` | `2` | LR ramp-up + gradient freeze window post-growth |
| `exploration_budget_events` | `2` | Max speculative growth events per run |
| `exploration_phase_epochs` | `6` | Epoch window during which exploration is allowed |
| `seed` | `13` | Random seed |
| `output_dir` | `"runs/mnist_sann_experiment"` | Output directory |

Run `python experiment_compare.py --help` for the full CLI reference.

---

## 📈 Results & Reproduction

> ⚠️ **No pre-computed results are bundled** in this repository. The `data.json` file contains a small 4-epoch CIFAR-10 sample used only by the web visualizer. All full results must be generated locally.

### Reproduce benchmark figures

```bash
# 1. Run the full multi-seed, multi-dataset benchmark
python experiment_pipeline.py \
  --datasets mnist fashion_mnist cifar10 \
  --epochs 15 \
  --seeds 13 23 33 43 53 \
  --output-dir runs/ca_sann_benchmark

# 2. Generate 3D efficiency scatter plots
python visualize_3d.py \
  --mode metrics \
  --input-root runs/ca_sann_benchmark \
  --output-dir runs/ca_sann_benchmark
```

### Reproduce notebook charts

1. Populate `runs/` with the benchmark pipeline above.
2. Open `SANN_Visualization_Pipeline.ipynb` — it auto-discovers results under `runs/`.
3. Open `SANN_3D_Visualization.ipynb` — it auto-discovers the latest `result.json`.

### Metrics reported

| Metric | Description |
|--------|-------------|
| Test accuracy | Mean ± std across seeds |
| Test loss | Mean ± std |
| Model size | Trainable parameter count |
| Growth events | Accepted / candidate / exploration / rejected |
| Efficiency | `accuracy / model_size` |
| Accuracy / 100k params | Normalized efficiency score |

Verdict categories (CA-SANN vs Static, per seed): **better** · **similar** · **trade-off** · **worse**

---

## 🗺️ Roadmap

- [ ] Add a `LICENSE` file
- [ ] Add CI/CD (GitHub Actions) with a smoke-test workflow
- [ ] Publish pre-trained checkpoints and pre-computed benchmark results
- [ ] Support YAML/JSON config files
- [ ] Add a `main.py` convenience entry point
- [ ] Add `pandas`, `seaborn`, `plotly` to `requirements.txt` (needed by notebooks)
- [ ] Add unit tests under `tests/`
- [ ] Hard structural pruning — physically remove masked neurons from weight matrices
- [ ] Extend to CIFAR-100, SVHN, and other datasets
- [ ] Multi-GPU / distributed training support
- [ ] Publish arXiv technical report with full ablation study

---

## 🤝 Contributing

Contributions are very welcome! Please follow these steps:

1. **Fork** the repository and create a feature branch from `main`.
2. Match the existing code style: type annotations everywhere, `dataclass`-based configs, `logging` for all runtime output.
3. Include a test or reproduce command showing your change works.
4. Run the smoke test before submitting a PR:
   ```bash
   python experiment_compare.py \
     --dataset mnist --epochs 3 \
     --train-samples 1000 --val-samples 200 --test-samples 200
   ```
5. Open a pull request with a clear description and a before/after metric comparison where applicable.

---

## 📄 License

> **TODO** — No `LICENSE` file is present in this repository. Please add one before accepting external contributions. Recommended choices for research code: [MIT](https://opensource.org/licenses/MIT) or [Apache 2.0](https://opensource.org/licenses/Apache-2.0).

---

## 📚 Citation

If you use CA-SANN in your research, please cite:

```bibtex
@software{ca_sann_2024,
  author  = {Jayaprakash},
  title   = {{CA-SANN}: Context-Aware Self-Evolving Neural Network},
  year    = {2024},
  url     = {https://github.com/jayaprakash2207/CA-SANN--Context-Aware-Self-Evolving-Neural-Network},
  note    = {TODO: replace with arXiv/DOI when available}
}
```

---

## 🙏 Acknowledgments

- Network visualization powered by [Three.js](https://threejs.org/) (bundled in `vendor/three/`).
- Interactive notebook visualization uses [Plotly](https://plotly.com/python/).
- Datasets provided by [torchvision](https://pytorch.org/vision/stable/datasets.html).
