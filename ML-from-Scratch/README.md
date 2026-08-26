# Machine Learning From Scratch

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557C?style=flat)](https://matplotlib.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-ArpitKadam-181717?style=flat&logo=github)](https://github.com/ArpitKadam/Attention-Is-All-You-Code)

> Ground-up, framework-free implementations of the mathematical building blocks of machine learning — derived, coded in pure NumPy, and visualized. This section isolates the primitives that every neural network depends on.

---

## Table of Contents

1. [Overview](#overview)
2. [Modules](#modules)
3. [Philosophy](#philosophy)
4. [Getting Started](#getting-started)

---

## Overview

Deep learning frameworks hide their most important ideas behind a single function call. **`ML-from-Scratch`** removes that abstraction, reconstructing the core components of learning systems from their mathematical definitions using nothing beyond NumPy. Each module pairs a formal derivation with an explicit implementation and a set of visualizations that make the behavior tangible.

Two foundational families are covered: the **activation functions** that give networks their non-linearity, and the **optimization algorithms** that drive learning.

---

## Modules

| Module | Description | Documentation |
| :--- | :--- | :--- |
| **[Activation-Functions](./Activation-Functions)** | Eight activation functions (Sigmoid, Softmax, TanH, ReLU, Leaky ReLU, ELU, SELU, SoftPlus) with forward passes and analytical gradients, implemented as callable classes. | [README](./Activation-Functions/README.md) |
| **[Optimizers](./Optimizers)** | Five gradient-based optimizers (BGD, SGD, Momentum, RMSProp, Adam) with update-rule derivations, convergence curves, and 2D optimization-path visualizations. | [README](./Optimizers/README.md) |

### Activation Functions
Non-linearity is what allows a deep network to represent complex functions; without it, any stack of layers collapses into a single linear map. This module implements each activation and its derivative — the exact quantities backpropagation requires — and compares them side by side over a common input range.

### Optimizers
The optimizer defines *how* a model learns from gradients. Starting from the derivation of the Mean Squared Error gradient, this module builds up from vanilla Batch Gradient Descent to Adam, showing at each step which limitation of the previous method the new one solves — and demonstrating the difference visually on both loss curves and contour trajectories.

---

## Philosophy

1. **Derive before implementing.** Every algorithm begins with its mathematics, so the code is a transcription of understanding rather than a black box.
2. **Depend only on NumPy.** No autograd, no framework — the mechanics are fully exposed.
3. **Visualize the behavior.** Plots turn abstract update rules into observable convergence dynamics.

---

## Getting Started

```bash
pip install numpy matplotlib jupyter
```

Open either module's notebook to run the derivations and reproduce the figures:

```bash
jupyter notebook Optimizers/Optimizers.ipynb
jupyter notebook Activation-Functions/Activation_functions.ipynb
```

Each subdirectory contains a detailed, self-contained README with full derivations, code sketches, and figure explanations.

---

*Part of the [Attention-Is-All-You-Code](https://github.com/ArpitKadam/Attention-Is-All-You-Code) project — implementing machine learning concepts from scratch.*
