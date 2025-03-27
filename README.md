# NormFit: A Lightweight Solution for Few-Shot Federated Learning with Heterogeneous and Imbalanced Data

This repository provides the implementation of **NormFit**, a novel fine-tuning framework designed specifically for few-shot federated learning scenarios characterized by heterogeneous and imbalanced data distributions.

## Overview

NormFit fine-tunes only the Pre-LayerNorm of the CLIP image encoder using focal loss. This approach effectively addresses data heterogeneity and class imbalance by prioritizing hard-to-classify samples and adapting feature representations efficiently.

## Key Features

- **Efficiency**: Extremely lightweight, reducing communication and computational overhead significantly.
- **Performance**: Robust against heterogeneous and imbalanced data, outperforming or matching state-of-the-art methods.
- **Versatility**: Can function as a standalone solution or seamlessly integrate as an add-on to existing fine-tuning methods.

## Repository Structure
```
NormFit/
├── datasets/
│   └── [Dataset placeholders: CIFAR10, Caltech101, OxfordPets, EuroSAT, etc.]
├── models/
│   └── clip.py
├── experiments/
│   ├── train_normfit.py
│   └── ablations.py
├── configs/
│   └── config.yaml
├── results/
│   └── [Experimental results and logs]
└── utils/
    └── utils.py
```

## Installation

1. Clone the repository:

```bash
git clone [REPOSITORY_URL]
cd NormFit
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments

### Federated Learning Setup

To run NormFit under federated learning settings:

```bash
python experiments/train_normfit.py --config configs/config.yaml --federated True
```

### Centralized Setup

To run NormFit in a centralized training setting:

```bash
python experiments/train_normfit.py --config configs/config.yaml --federated False
```

### Ablation Studies

To reproduce ablation studies:

```bash
python experiments/ablations.py --config configs/config.yaml
```

## Configuration

Modify `config.yaml` to customize experiments, datasets, hyperparameters, and more.

Example:
```yaml
dataset: CIFAR10
learning_rate: 0.0005
batch_size: 64
focal_loss_alpha: 0.25
focal_loss_gamma: 1.0
federated:
  num_clients: 100
  dirichlet_beta: 0.1
```

## Citation

Please cite our paper if you find this repository useful:

```
Anonymous Authors, "NormFit: A Lightweight Solution for Few-Shot Federated Learning with Heterogeneous and Imbalanced Data," submitted to ICML 2025.
```

## License

This project is released under the MIT License.

