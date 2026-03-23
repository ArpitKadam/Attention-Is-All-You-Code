# Optimization Algorithms from Scratch

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557C?style=flat)](https://matplotlib.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-ArpitKadam-181717?style=flat&logo=github)](https://github.com/ArpitKadam/Attention-Is-All-You-Code)

> A ground-up, educational implementation of the most important gradient-based optimization algorithms in machine learning — built with NumPy and visualized with Matplotlib.

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Background](#mathematical-background)
3. [Optimizers Implemented](#optimizers-implemented)
   - [Batch Gradient Descent (BGD)](#1-batch-gradient-descent-bgd)
   - [Stochastic Gradient Descent (SGD)](#2-stochastic-gradient-descent-sgd)
   - [Gradient Descent with Momentum](#3-gradient-descent-with-momentum)
   - [RMSProp](#4-rmsprop)
   - [Adam](#5-adam-adaptive-moment-estimation)
4. [Visualizations & Plots](#visualizations--plots)
5. [Implementation Details](#implementation-details)
6. [Comparison Summary](#comparison-summary)
7. [Key Takeaways](#key-takeaways)
8. [Usage](#usage)

---

## Overview

Optimization algorithms are the engine of machine learning. Every model learns by minimizing a **loss function** — a mathematical measure of how wrong its predictions are. The optimizer defines the iterative strategy used to update model parameters (weights and biases) to reduce this loss over time.

This notebook explores the most prominent **gradient-based optimizers**, implementing each from scratch without any deep learning framework. The goal is to build deep intuition about:

- **Why** each optimizer was designed the way it is
- **What problems** it solves over the previous generation
- **How** the update rules translate to convergence behavior in practice

Implementations are applied to two test beds:
1. **Linear regression on synthetic data** — minimizing Mean Squared Error (MSE)
2. **Quadratic bowl** loss surface ($f(x, y) = x^2 + 10y^2$) — ideal for visualizing optimization trajectories in 2D

---

## Mathematical Background

### The Loss Function: Mean Squared Error (MSE)

For a linear model $\hat{y} = mx + c$, the MSE over $n$ observations is:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (mx_i + c - y_i)^2$$

### Gradient Derivation via the Chain Rule

To update $m$ and $c$, we need the partial derivatives of MSE. Applying the **Power Rule** and **Chain Rule**:

$$\frac{\partial \text{MSE}}{\partial m} = \frac{2}{n} \sum_{i=1}^{n} x_i (\hat{y}_i - y_i)$$

$$\frac{\partial \text{MSE}}{\partial c} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$$

### General Parameter Update

All gradient-based optimizers follow this principle:

$$\theta \leftarrow \theta - \eta \cdot \Delta\theta$$

where $\eta$ is the **learning rate** and $\Delta\theta$ is an optimizer-specific update term derived from gradient information.

---

## Optimizers Implemented

### 1. Batch Gradient Descent (BGD)

**Intuition:**
Batch Gradient Descent (also called Vanilla Gradient Descent) computes the gradient of the loss function over the **entire training dataset** before making a single parameter update. This gives the most accurate gradient estimate at each step, but comes at the cost of computational efficiency on large datasets.

**Update Rule:**

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)$$

**Implementation sketch:**
```python
for epoch in range(epochs):
    predictions = X.dot(theta)
    errors = predictions - y
    gradients = (2 / m) * X.T.dot(errors)
    theta -= lr * gradients
```

**Advantages:**
- Produces stable, deterministic convergence
- Smooth, monotonically decreasing loss curve
- Conceptually simple baseline

**Limitations:**
- Prohibitively slow for large datasets (must process all samples per step)
- May converge to sharp local minima

**Typical Use Cases:** Small datasets, benchmarking, final fine-tuning

---

### 2. Stochastic Gradient Descent (SGD)

**Intuition:**
Instead of computing the gradient over the full dataset, SGD selects **one random training sample** per update. This introduces noise but dramatically reduces the cost per iteration, enabling learning at internet scale.

**Update Rule:**

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}\left(\theta_t;\, x^{(i)}, y^{(i)}\right)$$

where $(x^{(i)}, y^{(i)})$ is a randomly drawn training sample.

**Implementation sketch:**
```python
for epoch in range(epochs):
    idx = np.random.randint(m)
    xi, yi = X[idx:idx+1], y[idx:idx+1]
    prediction = xi.dot(theta)
    gradient = (2 / 1) * xi.T.dot(prediction - yi)
    theta -= lr * gradient
```

**Advantages:**
- Orders-of-magnitude faster per update on large datasets
- Gradient noise can help escape local minima
- Enables online/incremental learning

**Limitations:**
- Noisy, oscillating loss curve
- Requires careful learning rate tuning
- Less stable convergence than BGD

**Typical Use Cases:** Large-scale machine learning, online learning, modern deep learning (usually in mini-batch form)

---

### 3. Gradient Descent with Momentum

**Intuition:**
Imagine rolling a ball down a hilly landscape. The ball accelerates in directions where the slope is consistently downhill and slows where it fluctuates. Momentum mimics this by accumulating a **velocity vector** that averages past gradients — smoothing out oscillations on elongated or curved loss surfaces.

**Update Rule:**

$$v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot \nabla_\theta \mathcal{L}$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_t$$

where $\beta$ is the **momentum coefficient** (typically $0.9$), and $v_t$ is the velocity vector.

**Implementation sketch:**
```python
v = np.array([0.0, 0.0])
for _ in range(epochs):
    grad = grad_func(x, y)
    v = beta * v + (1 - beta) * grad
    x -= eta * v[0]
    y -= eta * v[1]
```

**Advantages:**
- Faster convergence than plain BGD on non-isotropic loss surfaces
- Suppresses oscillations transverse to the gradient direction
- Helps escape shallow local minima

**Limitations:**
- Introduces an additional hyperparameter $\beta$
- Can overshoot if momentum is too high

**Typical Use Cases:** Training deep neural networks, whenever gradient directions remain consistent over several steps

---

### 4. RMSProp

**Intuition:**
RMSProp (Root Mean Square Propagation, Hinton 2012) addresses a core limitation: all parameters share a single learning rate $\eta$. Some parameters may need large updates; others need small ones. RMSProp tracks a **per-parameter running average of squared gradients**, effectively normalizing the learning rate for each dimension individually.

**Update Rule:**

$$E[g^2]_t = \beta \cdot E[g^2]_{t-1} + (1 - \beta) \cdot g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t$$

where $\epsilon \approx 10^{-8}$ prevents division by zero.

**Implementation sketch:**
```python
eg2 = np.array([0.0, 0.0])
for _ in range(epochs):
    grad = grad_func(x, y)
    eg2 = beta * eg2 + (1 - beta) * (grad ** 2)
    x -= (lr / np.sqrt(eg2[0] + epsilon)) * grad[0]
    y -= (lr / np.sqrt(eg2[1] + epsilon)) * grad[1]
```

**Advantages:**
- Adaptive learning rate per parameter
- Effective on non-stationary objectives and sparse gradients
- Good out-of-the-box choice for recurrent networks

**Limitations:**
- Still requires manual tuning of the base learning rate $\eta$
- The effective learning rate is monotonically decreasing

**Typical Use Cases:** Recurrent neural networks (RNNs), non-stationary optimization problems

---

### 5. Adam (Adaptive Moment Estimation)

**Intuition:**
Adam (Kingma & Ba, 2015) is the synthesis: it combines **Momentum** (first-order moment) and **RMSProp** (second-order moment) and adds **bias correction** to account for the zero-initialization of both moment estimates. It is arguably the most widely used optimizer in modern deep learning.

**Update Rule:**

*First moment (momentum):*
$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

*Second moment (adaptive scaling):*
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

*Bias-corrected estimates:*
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

*Parameter update:*
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

**Default Hyperparameters:** $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

**Implementation sketch:**
```python
m = np.array([0.0, 0.0])
v = np.array([0.0, 0.0])
for i in range(1, epochs + 1):
    grad = grad_func(x, y)
    m = beta1 * m + (1 - beta1) * grad           # First moment
    v = beta2 * v + (1 - beta2) * (grad ** 2)    # Second moment
    m_hat = m / (1 - beta1 ** i)                 # Bias correction
    v_hat = v / (1 - beta2 ** i)
    x -= lr * m_hat[0] / (np.sqrt(v_hat[0]) + epsilon)
    y -= lr * m_hat[1] / (np.sqrt(v_hat[1]) + epsilon)
```

**Advantages:**
- Combines adaptive learning rates with momentum — best of both worlds
- Bias correction prevents poor updates in early training steps
- Works well with default hyperparameters across many architectures
- Handles sparse and noisy gradients effectively

**Limitations:**
- Higher memory overhead (two moment vectors per parameter)
- Can sometimes converge to sharp, less generalizable minima vs. SGD

**Typical Use Cases:** Default choice for most deep learning models, transformers, CNNs, NLP

---

## Visualizations & Plots

Six plots are generated and saved as `.png` files in this directory. Below is each plot with an explanation of what it represents.

---

### Loss vs. Iterations — BGD vs. SGD (Small & Large Datasets)

![Loss vs Iterations for Small and Large Datasets](Loss%20vs%20Iterations%20for%20Small%20and%20Large%20Datasets.png)

**What it shows:** The loss convergence of Batch Gradient Descent and Stochastic Gradient Descent on both a **1,000-sample** (small) and a **10,000,000-sample** (large) synthetic linear regression dataset.

**Key insight:** On the small dataset, BGD is smooth but slower per effective epoch. On the massive dataset, BGD becomes computationally infeasible while SGD remains tractable — demonstrating SGD's critical role in large-scale machine learning.

---

### Loss vs. Iterations — BGD vs. Momentum

![Loss vs Iterations in BGD vs Momentum](Loss%20vs%20Iterations%20in%20BGD%20vs%20Momentum.png)

**What it shows:** Loss curves for plain BGD and Momentum-based GD on the quadratic bowl $f(x, y) = x^2 + 10y^2$.

**Key insight:** Momentum converges significantly faster by accumulating gradient history. The loss for momentum drops steeply, especially in the early iterations, while BGD descends more gradually.

---

### Loss vs. Iterations — RMSProp vs. BGD

![Loss vs Iterations in RMSProp vs BGD](Loss%20vs%20Iterations%20in%20RMSProp%20vs%20BGD.png)

**What it shows:** Loss curves for RMSProp and BGD on the same quadratic bowl.

**Key insight:** RMSProp's per-parameter adaptive learning rates produce faster, smoother convergence. The normalization by the running average of squared gradients ensures that dimensions with high gradient magnitudes (like the $y$-axis with coefficient $10$) are properly scaled down.

---

### Loss vs. Iterations — Adam vs. BGD

![Loss vs Iterations in Adam vs BGD](Loss%20vs%20Iterations%20in%20Adam%20vs%20BGD.png)

**What it shows:** Loss curves comparing Adam and BGD.

**Key insight:** Adam achieves the fastest convergence among all optimizers shown, dropping the loss to near-zero within a fraction of the iterations required by BGD.

---

### Optimization Path — RMSProp vs. BGD

![Optimization Path in RMSProp vs BGD](Optimization%20Path%20in%20RMSProp%20vs%20BGD.png)

**What it shows:** A 2D contour plot of the quadratic loss surface, overlaid with the **step-by-step trajectory** each optimizer takes from a starting point of $(1.0, 1.0)$ toward the global minimum at $(0, 0)$.

**Key insight:** BGD zigzags down the elongated contours. RMSProp takes a more direct path by compensating for the asymmetric curvature — the adaptive scaling along each axis allows it to move more efficiently in both dimensions simultaneously.

---

### Optimization Path — Adam vs. BGD

![Optimization Path in Adam vs BGD](Optimization%20Path%20in%20Adam%20vs%20BGD.png)

**What it shows:** Optimization trajectories for Adam and BGD on the quadratic bowl.

**Key insight:** Adam's path is among the most direct seen across all algorithms. The combined effect of momentum (direction persistence) and adaptive scaling (dimension normalization) produces a trajectory that navigates the loss landscape with maximal efficiency.

---

## Implementation Details

All code lives in **`Optimizers.ipynb`**. The notebook is organized into clearly delineated sections, one per optimizer. Each section contains:

1. A **Markdown derivation** of the algorithm's update rule
2. **Python implementation** using only NumPy
3. **Visualization** of loss convergence and/or optimization paths

### Key Shared Utilities

```python
# Loss surface used for 2D trajectory comparisons
def quadratic_loss_function(x, y):
    return x**2 + 10*y**2

# Analytical gradient of the quadratic loss
def quadratic_grad(x, y):
    dx = 2 * x
    dy = 20 * y
    return np.array([dx, dy])

# MSE for linear regression
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
```

### Experimental Setup

| Parameter | Value |
|-----------|-------|
| Quadratic start point | $(1.0,\ 1.0)$ |
| Epochs (quadratic) | 100 |
| Epochs (linear regression) | 500–10,000 |
| Learning rate ($\eta$) | 0.001–0.1 |
| Momentum ($\beta$) | 0.9 |
| Adam $\beta_1$ | 0.9 |
| Adam $\beta_2$ | 0.999 |
| $\epsilon$ (numerical stability) | $10^{-8}$ |

---

## Comparison Summary

| Optimizer | Adaptive LR | Momentum | Bias Correction | Memory Cost | Best For |
|-----------|:-----------:|:--------:|:---------------:|:-----------:|----------|
| BGD | ✗ | ✗ | ✗ | Low | Small datasets, baselines |
| SGD | ✗ | ✗ | ✗ | Low | Large-scale, online learning |
| Momentum | ✗ | ✓ | ✗ | Low | Consistent gradient directions |
| RMSProp | ✓ | ✗ | ✗ | Medium | RNNs, non-stationary problems |
| Adam | ✓ | ✓ | ✓ | High | General-purpose deep learning |

---

## Key Takeaways

1. **Vanilla BGD** is the simplest optimizer and the conceptual foundation of all others. It is the correct baseline for understanding how gradient information drives learning.

2. **SGD** is not just a "faster BGD" — the gradient noise it introduces is a feature, not a bug. It enables convergence on datasets too large to fit in memory and can regularize training by preventing convergence to sharp minima.

3. **Momentum** breaks the zigzag behavior of gradient descent on anisotropic loss surfaces. The velocity vector acts as a low-pass filter over past gradients, smoothing the update direction and accelerating convergence.

4. **RMSProp** solves the "one learning rate to rule them all" problem. By normalizing each gradient component with its historical squared magnitude, it automatically calibrates the effective step size per dimension.

5. **Adam** is the synthesis of decades of optimization research. By combining first-moment (momentum) and second-moment (RMSProp-style) estimates with bias correction, it delivers robust, fast convergence with minimal hyperparameter tuning.

6. **Visualization reveals what equations cannot**: The contour plots demonstrate that adaptive methods (RMSProp, Adam) take dramatically more direct paths to the minimum compared to BGD — making the case for adaptive optimizers viscerally clear.

---

## Usage

### Prerequisites

```bash
pip install numpy matplotlib jupyter
```

### Running the Notebook

```bash
jupyter notebook Optimizers.ipynb
```

No external datasets are required. All data is generated synthetically within the notebook. Plots are automatically saved as `.png` files to the current directory.

---

*Part of the [Attention-Is-All-You-Code](https://github.com/ArpitKadam/Attention-Is-All-You-Code) project — implementing machine learning concepts from scratch.*
