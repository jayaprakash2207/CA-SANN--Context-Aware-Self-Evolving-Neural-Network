<div align="center">

<img src="https://img.shields.io/badge/CA--SANN-Context--Aware%20Self--Evolving%20Neural%20Network-blueviolet?style=for-the-badge&logo=pytorch&logoColor=white" alt="CA-SANN"/>

# 🧠 CA-SANN
### *Context-Aware Self-Evolving Neural Network*

> **Networks that grow, prune, and adapt — intelligently, autonomously, continuously.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/jayaprakash2207/CA-SANN--Context-Aware-Self-Evolving-Neural-Network?style=flat-square&color=yellow)](https://github.com/jayaprakash2207/CA-SANN--Context-Aware-Self-Evolving-Neural-Network/stargazers)

</div>

---

## 🌟 What is CA-SANN?

**CA-SANN** is a pioneering neural network framework that breaks free from the constraints of static architectures. Instead of fixing your network's size before training and hoping for the best, CA-SANN **continuously monitors its own performance**, detects bottlenecks, and surgically **grows or prunes neurons and channels in real-time** — all without human intervention.

Traditional approach → *You design the architecture. You hope it's right.*

**CA-SANN approach** → *The network reads context. The network decides. The network evolves.*

```
Epoch 1  →  [128 → 64]  →  accuracy: 72%   (underfit detected)
Epoch 10 →  [160 → 80]  →  accuracy: 89%   (growth triggered)
Epoch 25 →  [160 → 72]  →  accuracy: 91%   (pruning applied)
Epoch 50 →  [176 → 72]  →  accuracy: 93%   (optimal capacity)
```

---

## ✨ Key Highlights

| Feature | Description |
|---|---|
| 🔁 **Self-Evolving** | Grows and prunes neurons dynamically during training |
| 🧭 **Context-Aware** | Reads capacity status (underfit / optimal / overfit) in real time |
| 🔬 **Hard Sample Detection** | Identifies and prioritizes persistently misclassified samples |
| 📊 **Gradient Flow Analysis** | Monitors vanishing gradients layer-by-layer |
| 🧬 **Meta-Learning Policy** | Learns *which layers* benefit most from growth over time |
| 🧩 **Dual Architecture** | Supports both MLP (tabular/image) and CNN (vision) workloads |
| 📈 **3D Visualization** | Animated 3D network evolution timeline |
| ⚙️ **Fully Configurable** | Every growth, pruning, and estimation parameter is tunable |

---

## 🗂️ Project Structure

```
CA-SANN/
│
├── 📦 Core Architecture
│   ├── model.py                  # DynamicMLP & DynamicCNN — growable/prunable layers
│   └── meta_controller.py        # Online meta-learning policy for growth decisions
│
├── 🧠 Intelligence Layer
│   ├── controller.py             # GrowthController — decision engine for grow/prune actions
│   ├── context_estimators.py     # CapacityEstimator & DifficultyEstimator
│   └── analyzer.py               # ErrorAnalyzer — tracks hard samples, layer importance
│
├── 🚀 Training & Experiments
│   ├── train.py                  # Full training loop with evolution logic
│   ├── experiment.py             # Single experiment runner
│   ├── experiment_compare.py     # Comparative analysis across multiple runs
│   ├── experiment_pipeline.py    # Full end-to-end training pipeline
│   └── utils.py                  # Shared utilities
│
├── 📊 Visualization
│   ├── visualize_3d.py           # 3D network evolution visualization
│   ├── SANN_3D_Visualization.ipynb          # Interactive 3D notebook
│   └── SANN_Visualization_Pipeline.ipynb    # Complete analysis notebook
│
├── 🌐 Web Interface
│   ├── index.html                # Interactive visualization dashboard
│   ├── script.js                 # Dashboard logic
│   └── style.css                 # Dashboard styling
│
└── 📄 Config & Deps
    ├── data.json                 # Configuration data
    └── requirements.txt          # Python dependencies
```

---

## 🏗️ Architecture Deep Dive

### Dynamic Network Models

CA-SANN provides two fully dynamic network types, each supporting live neuron-level manipulation:

#### `DynamicMLP` — Fully Connected Networks

```python
from model import DynamicMLP, DynamicMLPConfig

config = DynamicMLPConfig(
    input_dim=784,          # e.g., flattened MNIST
    hidden_dims=[128, 64],  # initial hidden layers
    output_dim=10,          # number of classes
    activation="relu"       # relu | gelu | silu
)
model = DynamicMLP(config)

# Grow layer 0 by 16 neurons
model.grow_layer(layer_idx=0, n_new=16)

# Prune specific neurons from layer 1
model.prune_layer_neurons(layer_idx=1, neuron_indices=[3, 7, 12])

print(model.architecture_summary())
# DynamicMLP  input=784 → [144, 64] → output=10  params=130,954
```

#### `DynamicCNN` — Convolutional Networks

```python
from model import DynamicCNN, DynamicCNNConfig

config = DynamicCNNConfig(
    in_channels=3,
    channels=[32, 64],      # initial conv channels
    fc_hidden=[256],
    output_dim=10,
    use_batchnorm=True,
    activation="gelu"
)
model = DynamicCNN(config)

# Grow conv layer 0 by 8 channels
model.grow_layer(layer_idx=0, n_new=8)
```

---

### 🧭 The Intelligence Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Every N Training Epochs                   │
│                                                             │
│  ① ErrorAnalyzer        ② CapacityEstimator                 │
│     ↓ hard samples          ↓ underfit/optimal/overfit       │
│     ↓ layer importance      ↓ val/train loss gap             │
│     ↓ gradient scores       ↓ error distribution            │
│                                                             │
│  ③ DifficultyEstimator  ④ GrowthController                  │
│     ↓ composite score       ↓ layer priority scores         │
│     ↓ entropy / tail        ↓ grow / prune decisions        │
│                                                             │
│  ⑤ MetaController (LearningPolicy)                          │
│     ↓ tracks outcomes per layer                             │
│     ↓ biases future decisions toward beneficial layers      │
│                                                             │
│  ⑥ Apply actions → update DynamicMLP / DynamicCNN           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔑 Core Concepts

### 📈 Growth Triggers

The `GrowthController` decides *when* and *where* to grow based on multiple signals:

| Trigger | Signal | Action |
|---|---|---|
| 🌊 **Gradient Vanishing** | Layer gradient norm < `1e-4` | Expand that layer |
| 💀 **Dead Neurons** | Activation dead ratio > 35% | Replace with fresh neurons |
| 📉 **High Validation Loss** | Loss above threshold | Add capacity |
| 🎯 **Hard Samples** | Persistent misclassifications > 32 samples | Targeted expansion |
| 🔻 **Layer Underperformance** | Layer error rate > `min_underperformance_to_grow` | Grow that layer |

### ✂️ Pruning Strategy

Pruning keeps the network efficient — removing neurons that contribute little:

| Condition | Behavior |
|---|---|
| **Low contribution** | Identify neurons with minimal activation or weight norms |
| **Stagnating loss** | Only prune when loss has plateaued (`require_stagnation=True`) |
| **Minimum active fraction** | Always preserve ≥ 35% of neurons (`min_active_fraction=0.35`) |
| **High validation loss** | Skip pruning to prevent further capacity damage |

### 🧬 Meta-Learning Policy

The `MetaController` turns CA-SANN into a system that learns how to learn:

```
For each layer:
  → After growth: measure accuracy improvement
  → Record outcome (improved / no change / degraded)
  → Compute policy_bias = weighted moving average of outcomes
  → Apply bias to future priority scores for this layer
```

Over time, the policy converges on the layers that consistently benefit from expansion — and stops wasting budget on layers that don't.

---

## 📡 Context Estimation

### Capacity Status

`CapacityEstimator` classifies the model's training state every few epochs:

```
"underfit"  → Both train & val loss high, many hard samples → trigger growth
"overfit"   → Train loss low, val loss high, large gap → hold or prune  
"optimal"   → Balanced losses, good generalization → maintain
```

### Difficulty Score

`DifficultyEstimator` computes a **composite difficulty score** [0, 1] using:

```python
score = (
    0.30 × hard_sample_pressure     # persistent misclassification pressure
  + 0.25 × prediction_entropy       # model uncertainty (normalized)
  + 0.20 × gradient_cv              # variability in gradient norms
  + 0.15 × loss_tail_ratio          # heavy tail in sample loss distribution
  + 0.05 × error_rate               # raw fraction misclassified
  + 0.05 × error_class_entropy      # spread of errors across classes
)
```

High difficulty → more aggressive growth. Low difficulty → pruning permitted.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jayaprakash2207/CA-SANN--Context-Aware-Self-Evolving-Neural-Network
cd CA-SANN--Context-Aware-Self-Evolving-Neural-Network

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install torch>=2.2.0 torchvision>=0.17.0 matplotlib>=3.8.0
```

### Run a Training Experiment

```bash
# Run the full training pipeline on MNIST (MLP)
python experiment_pipeline.py

# Run a single experiment
python experiment.py

# Compare multiple runs
python experiment_compare.py
```

### Training Programmatically

```python
from train import TrainingConfig, run_training

config = TrainingConfig(
    dataset_name="mnist",
    model_arch="mlp",          # "mlp" or "cnn"
    hidden_dims=(128, 64),
    epochs=50,
    batch_size=256,
    learning_rate=8e-4,
    output_dir="runs/my_experiment"
)

results = run_training(config)
print(f"Final accuracy: {results['test_accuracy']:.2%}")
print(f"Final architecture: {results['architecture_summary']}")
```

### CIFAR-10 with CNN

```python
from train import TrainingConfig, run_training

config = TrainingConfig(
    dataset_name="cifar10",
    model_arch="cnn",
    epochs=50,
    output_dir="runs/cifar10_cnn"
)

results = run_training(config)
```

---

## ⚙️ Configuration Reference

### Growth Configuration (`GrowthConfig`)

| Parameter | Default | Description |
|---|---|---|
| `warmup_epochs` | `5` | Epochs before growth is allowed |
| `decision_interval` | `2` | Epochs between growth evaluations |
| `min_growth_priority` | `0.75` | Minimum score to trigger growth |
| `base_growth` | `8` | Default neurons added per event |
| `max_growth_per_layer` | `24` | Max neurons grown in a single layer per event |
| `total_growth_budget_neurons` | `48` | Total neuron budget per growth event |
| `max_model_params` | `500,000` | Hard cap on model parameter count |
| `cooldown_epochs` | `3` | Minimum epochs between consecutive growths |
| `gradient_vanishing_threshold` | `1e-4` | Gradient norm below this triggers growth |
| `activation_dead_ratio_threshold` | `0.35` | Dead neuron ratio that triggers growth |

### Pruning Configuration (`PruneConfig`)

| Parameter | Default | Description |
|---|---|---|
| `enabled` | `True` | Enable/disable pruning entirely |
| `prune_interval` | `3` | Epochs between pruning evaluations |
| `max_neurons_per_event` | `16` | Max neurons pruned per event |
| `min_active_fraction` | `0.35` | Minimum fraction of neurons to always keep |
| `require_stagnation` | `True` | Only prune when loss has plateaued |
| `skip_if_high_val_loss` | `True` | Prevent pruning during high-loss phases |

### Training Configuration (`TrainingConfig`)

| Parameter | Default | Description |
|---|---|---|
| `epochs` | `5` | Total training epochs (quick-start default; use 50–100 for real experiments) |
| `batch_size` | `256` | Training batch size |
| `learning_rate` | `8e-4` | Adam optimizer learning rate |
| `weight_decay` | `1e-5` | L2 regularization strength |
| `complexity_lambda` | `1e-6` | Complexity penalty coefficient |
| `dataset_name` | `"mnist"` | Dataset: `"mnist"` or `"cifar10"` |
| `model_arch` | `"mlp"` | Architecture: `"mlp"` or `"cnn"` |

---

## 📊 Visualization

### 3D Evolution Timeline

```bash
python visualize_3d.py
```

Generates an interactive 3D visualization showing:
- Layer-by-layer neuron counts over training epochs
- Growth and pruning events as animated transitions
- Parameter count trajectory

### Jupyter Notebooks

| Notebook | Contents |
|---|---|
| `SANN_3D_Visualization.ipynb` | 3D architecture evolution, parameter tracking, interactive exploration |
| `SANN_Visualization_Pipeline.ipynb` | Full experiment execution, loss curves, layer-wise metrics, comparative analysis |

### Web Dashboard

Open `index.html` in your browser for the interactive visualization dashboard showing real-time training metrics and architecture state.

---

## 🔬 Research Applications

### Neural Architecture Search (NAS)
CA-SANN serves as a dynamic NAS framework — instead of searching over a discrete set of fixed architectures, the network continuously discovers its own optimal size via gradient-informed growth and contribution-based pruning.

### Continual Learning
As task difficulty shifts, CA-SANN gracefully expands capacity in relevant layers. The hard-sample detection mechanism prevents catastrophic forgetting by keeping "difficult" knowledge accessible.

### Resource-Constrained Deployment
The `max_model_params` constraint and pruning policies allow CA-SANN to find the smallest network that still achieves target accuracy — ideal for edge deployment.

### Transfer Learning Enhancement
Pre-train with architecture evolution on a large source dataset, then fine-tune on a target task with continued capacity adaptation. The evolved architecture often generalizes better than hand-designed equivalents.

---

## 📚 Notebooks & Examples

### `SANN_Visualization_Pipeline.ipynb`

Full end-to-end experiment notebook:
1. Dataset loading (MNIST / CIFAR-10)
2. Model initialization with dynamic architecture
3. Training with evolution
4. Loss and accuracy curves
5. Layer-wise growth/pruning event timeline
6. Hard sample analysis

### `SANN_3D_Visualization.ipynb`

3D network evolution notebook:
1. Architecture snapshots at each epoch
2. Animated 3D evolution timeline
3. Channel/neuron count heatmaps
4. Parameter count trajectory

---

## 🤝 Contributing

Contributions are welcome! Here are exciting areas to explore:

- 🤖 **Transformer support** — Dynamic attention head growth/pruning
- 🏆 **RL-based growth policy** — Replace heuristics with reinforcement learning
- 🎯 **Multi-objective optimization** — Jointly optimize accuracy and FLOPs
- 🌐 **Distributed training** — Synchronized evolution across GPU workers
- 📦 **ONNX export** — Serialize evolved architectures for deployment

```bash
# Fork, clone, and install dev dependencies
git clone https://github.com/<your-username>/CA-SANN--Context-Aware-Self-Evolving-Neural-Network
pip install -r requirements.txt
```

Please open an issue first to discuss proposed changes!

---

## 📄 Citation

If CA-SANN is useful in your research, please consider citing:

```bibtex
@software{ca-sann2024,
  title   = {CA-SANN: Context-Aware Self-Evolving Neural Network},
  author  = {Jayaprakash},
  year    = {2024},
  url     = {https://github.com/jayaprakash2207/CA-SANN--Context-Aware-Self-Evolving-Neural-Network}
}
```

---

## ❓ FAQ

**Q: How much overhead does architecture evolution add?**
> Growth decisions run every 2 epochs by default. The overhead is approximately 5–10% additional training time, depending on dataset and model size.

**Q: Can I use CA-SANN with my own dataset?**
> Yes! Provide a standard PyTorch `DataLoader` and configure `TrainingConfig` with your `input_dim` and `num_classes`. See `train.py` for the expected data format.

**Q: Is pruning permanent?**
> Current pruning masks out neurons (sets their outputs to zero). Growth can re-expand those layers, effectively replacing pruned capacity.

**Q: What is the default maximum network size?**
> 500,000 parameters, configurable via `GrowthConfig.max_model_params`.

**Q: Does CA-SANN support GPU training?**
> Yes. The device defaults to `cuda` if available, otherwise `cpu`. Configure via `TrainingConfig.device`.

**Q: Can I implement my own growth policy?**
> Absolutely. Subclass or replace `GrowthController` and `MetaController` with your own logic. The `DynamicMLP` and `DynamicCNN` classes expose `grow_layer()` and `prune_layer_neurons()` directly.

---

## 📎 Resources

- 📘 **Module Docstrings** — Every class and function is documented inline
- 📓 **Notebooks** — `SANN_3D_Visualization.ipynb` and `SANN_Visualization_Pipeline.ipynb`
- 🐛 **Issues** — [GitHub Issues](https://github.com/jayaprakash2207/CA-SANN--Context-Aware-Self-Evolving-Neural-Network/issues) for bug reports and feature requests
- 💬 **Discussions** — [GitHub Discussions](https://github.com/jayaprakash2207/CA-SANN--Context-Aware-Self-Evolving-Neural-Network/discussions) for questions and ideas

---

## 📜 License

This project is released under the **MIT License** — free to use, modify, and distribute with attribution.

---

<div align="center">

**Created with 🧠 by [Jayaprakash](https://github.com/jayaprakash2207)**

*Inspired by Neural Architecture Search, meta-learning, and adaptive computation research*

---

**⭐ Star this repository if CA-SANN sparks your interest!**

*Let networks evolve intelligently. Let evolution guide architecture design.*

</div>
