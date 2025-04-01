# NormFit: A Lightweight Solution for Few-Shot Federated Learning with Heterogeneous and Imbalanced Data

This repository provides the implementation of **NormFit**, a novel fine-tuning framework designed specifically for few-shot federated learning scenarios characterized by heterogeneous and imbalanced data distributions.

## Overview

NormFit fine-tunes only the Pre-LayerNorm of the CLIP image encoder using focal loss. This approach effectively addresses data heterogeneity and class imbalance by prioritizing hard-to-classify samples and adapting feature representations efficiently.

## Key Features

- **Efficiency**: Extremely lightweight, reducing communication and computational overhead significantly.
- **Performance**: Robust against heterogeneous and imbalanced data, outperforming or matching state-of-the-art methods.
- **Versatility**: Can function as a standalone solution or seamlessly integrate as an add-on to existing fine-tuning methods.



## Installation

1. Clone the repository:

```bash
git clone [REPOSITORY_URL]
cd NormFit
```



## Citation

Please cite our paper if you find this repository useful:

```
Anonymous Authors, "NormFit: A Lightweight Solution for Few-Shot Federated Learning with Heterogeneous and Imbalanced Data," submitted to ICML 2025.
```

## License

This project is released under the MIT License.

