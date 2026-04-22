# The-Self-Pruning-Neural-Network
# Self-Pruning Neural Network 🧠✂️

A PyTorch implementation of a feed-forward neural network that dynamically learns to prune its own architecture during training. Instead of relying on post-training magnitude pruning, this network uses learnable gating parameters and L1 regularization to identify and sever weak connections on the fly.

This project was built as a case study for AI Engineering, demonstrating advanced PyTorch layer customization and robust training pipelines.

## 🚀 Key Features

* **Custom `PrunableLinear` Layer:** A drop-in replacement for `nn.Linear` that maintains a set of learnable `gate_scores` alongside standard weights.
* **Dynamic On-the-Fly Pruning:** Uses a Sigmoid activation bounded by an L1 Sparsity Loss to constantly push non-essential gate values to exactly `0`.
* **Robust Training Pipeline:** Implements industry best practices including **Batch Normalization**, **Dropout**, **Kaiming Uniform Initialization**, and a **Cosine Annealing Learning Rate Scheduler** to ensure stable training from scratch on CIFAR-10.
* **Comprehensive Evaluation:** Automatically tracks and plots classification loss, sparsity levels, test accuracy, and the final distribution of gate values across different sparsity pressure ($\lambda$) values.

## 🧮 How It Works

Each weight in the network is multiplied by a learnable gate value between 0 and 1:
```math
\text{gate} = \sigma(\text{gate\_score})
\text{pruned\_weight} = \text{weight} \times \text{gate}

During training, a custom L1 Sparsity Loss is added to the standard Cross-Entropy loss:
\text{Total Loss} = \text{Classification Loss} + \lambda \sum |\text{gate}|

\text{Total Loss} = \text{Classification Loss} + \lambda \sum |\text{gate}|
