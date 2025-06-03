# DYNAMO
# DYNAMO: Dynamic Causal-Temporal Adaptation in Transformer Architectures

[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **DYNAMO** addresses two fundamental limitations of Large Language Models: **temporal bias** from static training data and **causal fragility** from correlational learning through parameter-efficient temporal-causal adapters.

## 🚀 Key Highlights

- **89.7% temporal accuracy** (+17.3% vs GPT-4-turbo) on recent events
- **87.5% causal F1 score** (+8.2 points improvement)
- **142× faster updates** (42 minutes vs 48+ hours)
- **0.11% trainable parameters** - extremely parameter efficient
- **Theoretical guarantees** for bounded output drift

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)
- [Contributing](#contributing)

## 🔍 Overview

Large Language Models suffer from:
1. **Temporal Bias**: Performance degrades on recent events (LLaMA-3 drops to 31.7% on post-2023 events)
2. **Causal Fragility**: Poor causal reasoning due to correlational learning patterns

DYNAMO solves both through:

### Core Components

1. **Time2Vec Embeddings** - Continuous temporal representation
   ```
   φ(t) = [ωₖt + φₖ, sin(ωₖt + φₖ), cos(ωₖt + φₖ)]
   ```

2. **Causal Graph Projections** - Dynamic causal structure modeling
   ```
   f_g(𝒢ₜ) = GNN(𝐀ₜ) where 𝐀ₜ[i,j] = wᵢⱼᵗ · 𝟙[t ∈ τᵢⱼ]
   ```

3. **Parameter-Efficient Adapters** - Lightweight adaptation modules
   ```
   Δ𝐇ˡ = 𝐖ₒσ(𝐖ₜφ(t) + 𝐖_g f_g(𝒢ₜ))
   ```

4. **Causal Invariance Regularization** - Stable reasoning across time
   ```
   R(Ψ) = Σ D_JS(P_Ψ^(tᵢ) ∥ P_Ψ^(tⱼ))
   ```

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/dynamo-llm/dynamo.git
cd dynamo

# Create virtual environment
python -m venv dynamo_env
source dynamo_env/bin/activate  # On Windows: dynamo_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install DYNAMO
pip install -e .
```

### Hardware Requirements
- **Training**: 4× A100 GPUs (recommended)
- **Inference**: 1× RTX 3090 or better
- **RAM**: 32GB+ system memory

## 🚀 Quick Start

### Basic Usage

```python
from dynamo import DynamoModel, Time2VecEmbedding
import torch

# Initialize DYNAMO model
model = DynamoModel.from_pretrained("llama-3-8b")

# Add temporal-causal adapters
model.add_dynamo_adapters(
    temporal_dim=64,
    causal_dim=128,
    num_layers=32
)

# Prepare input with temporal context
text = "What is the current status of renewable energy adoption?"
timestamp = "2025-06-02"  # Current date
causal_graph = load_causal_graph("energy_domain.json")

# Generate response
response = model.generate(
    text=text,
    timestamp=timestamp,
    causal_graph=causal_graph,
    max_length=256
)

print(response)
```

### Training Your Own Model

```python
from dynamo import DynamoTrainer, DynamoConfig

# Configure training
config = DynamoConfig(
    base_model="llama-3-8b",
    temporal_dim=64,
    causal_dim=128,
    learning_rate=1e-4,
    batch_size=16,
    causal_reg_weight=0.1
)

# Initialize trainer
trainer = DynamoTrainer(config)

# Load your temporal-causal dataset
train_data = load_dataset("your_temporal_dataset.json")

# Train adapters
trainer.train(
    train_data=train_data,
    num_epochs=10,
    save_path="./dynamo_checkpoints/"
)
```

## 🏗️ Architecture

### Adapter Architecture

```
Input Sequence
      ↓
┌─────────────────┐
│  Base LLM Layer │
└─────────────────┘
      ↓
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Time2Vec      │    │   Causal     │    │    Adapter      │
│   φ(t)          │ +  │   GNN        │ →  │   Modulation    │
│                 │    │   f_g(𝒢ₜ)    │    │   Δ𝐇ˡ          │
└─────────────────┘    └──────────────┘    └─────────────────┘
                                                   ↓
                                          ┌─────────────────┐
                                          │  Enhanced       │
                                          │  Representation │
                                          └─────────────────┘
```

### Mathematical Formulation

The adapter modification for each layer ℓ:

```
𝐇ₒᵤₜˡ = 𝐇ˡ + Δ𝐇ˡ(t, 𝒢ₜ)

Δ𝐇ˡ = 𝐖ₒ · σ(𝐖ₜφ(t) + 𝐖_g f_g(𝒢ₜ))
```

Where:
- `φ(t)`: Time2Vec embedding capturing temporal patterns
- `f_g(𝒢ₚ)`: Graph neural network processing time-varying causal structure
- `σ`: GELU activation function

## 🧪 Experiments

### Datasets
- **FreshBench**: Temporal question answering across 2020-2025
- **CausalBank**: Extended causal reasoning benchmark

### Evaluation Metrics
- **Temporal Accuracy**: % correct on time-sensitive QA
- **Causal F1**: Precision/recall on causal relation extraction
- **Update Efficiency**: Training time for knowledge updates
- **Parameter Efficiency**: % of total parameters needed

### Running Evaluations

```bash
# Evaluate on FreshBench
python evaluate.py --dataset freshbench --model dynamo --checkpoint ./checkpoints/dynamo_best.pt

# Evaluate on CausalBank
python evaluate.py --dataset causalbank --model dynamo --checkpoint ./checkpoints/dynamo_best.pt

# Compare with baselines
python compare_baselines.py --datasets freshbench,causalbank --metrics temporal_acc,causal_f1
```

## 📊 Results

### Main Results

| Model | Temporal Acc (%) | Causal F1 (%) | Update Time | Trainable Params |
|-------|------------------|---------------|-------------|------------------|
| LLaMA-3 | 52.3 | 58.2 | 48.2h | 100% |
| T5-XXL (2025 ft) | 74.6 | 66.7 | 24.1h | 100% |
| GPT-4-turbo | 82.4 | 79.3 | <1h | N/A |
| **DYNAMO** | **89.7** | **87.5** | **0.7h** | **0.11%** |

### Temporal Performance Over Time

DYNAMO maintains consistent performance across years:
- **2020**: 91.2% accuracy
- **2025**: 87.9% accuracy  
- **Degradation**: <3% (vs 57.5% for baselines)

### Ablation Study

| Component | Temporal Acc | Causal F1 | Consistency |
|-----------|-------------|-----------|-------------|
| Full DYNAMO | 89.7% | 87.5% | 93.4% |
| - Time2Vec | 85.2% | 84.7% | 89.1% |
| - Causal GNN | 86.3% | 72.4% | 84.7% |
| - Causal Reg. | 88.1% | 79.6% | 85.3% |

## 🔬 Theoretical Analysis

### Bounded Output Drift Theorem

Under L-Lipschitz continuity of adapter parameters:

```
‖ℳ(x,t) - ℳ(x,t+Δ)‖₂ ≤ L κ |Δ| ‖φ'(t)‖₂
```

Where `κ = ‖𝐖ₒ‖₂ · ‖𝐖ₜ‖₂` is the condition number.

This guarantees stable model behavior under temporal shifts.

## 📝 Citation

If you find DYNAMO useful in your research, please cite:




### Development Setup

```bash
# Clone for development
git clone https://github.com/dynamo-llm/dynamo.git
cd dynamo

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
black . && flake8 .
```

### Areas for Contribution
- [ ] Multimodal integration (vision, audio)
- [ ] Automated causal graph construction
- [ ] Additional temporal embedding methods
- [ ] Efficiency optimizations
- [ ] New evaluation benchmarks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Paper**: [arXiv:2025.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)
- **Documentation**: [docs.dynamo-llm.com](https://docs.dynamo-llm.com)
- **Issues**: [GitHub Issues](https://github.com/dynamo-llm/dynamo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dynamo-llm/dynamo/discussions)

## 🙏 Acknowledgments

- FreshBench team for temporal evaluation framework
- CausalNLP community for causal reasoning resources
- Llama for base model architectures
- Open source contributors and reviewers

---

**Made with ❤️ and a passion for mathematics by the DYNAMO team (Just Karthik), but I am grateful for the mentors who have helped guide this paper**

*For questions and support, please open an issue or join our community discussions.*
