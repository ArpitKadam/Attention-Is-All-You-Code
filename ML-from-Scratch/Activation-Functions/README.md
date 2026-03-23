# Activation Functions from Scratch

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Array_Computing-013243?logo=numpy&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-Fundamentals-red)
![Neural Networks](https://img.shields.io/badge/Neural_Networks-Activation_Functions-orange)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-From_Scratch-green)

A from-scratch implementation of the most widely used activation functions in modern neural networks, complete with **forward pass** and **gradient (derivative)** computations using only NumPy. Each function is implemented as a callable Python class, ready for integration into custom deep learning pipelines.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Implemented Activation Functions](#implemented-activation-functions)
   - [Sigmoid](#1-sigmoid)
   - [Softmax](#2-softmax)
   - [TanH](#3-tanh-hyperbolic-tangent)
   - [ReLU](#4-relu-rectified-linear-unit)
   - [Leaky ReLU](#5-leaky-relu)
   - [ELU](#6-elu-exponential-linear-unit)
   - [SELU](#7-selu-scaled-exponential-linear-unit)
   - [SoftPlus](#8-softplus)
3. [Comparative Analysis](#comparative-analysis)
4. [Implementation Design](#implementation-design)
5. [Usage](#usage)
6. [Key Takeaways](#key-takeaways)

---

## Motivation

Activation functions introduce **non-linearity** into neural networks, enabling them to learn complex, non-linear mappings from inputs to outputs. Without activation functions, a multi-layer network would collapse into a single linear transformation regardless of its depth. The choice of activation function directly affects gradient flow, convergence speed, and overall model performance.

This module implements eight foundational activation functions entirely in NumPy, with both the forward computation (`__call__`) and the analytical gradient (`gradient`), providing a clear view of the mathematics that underpins modern deep learning.

---

## Implemented Activation Functions

### 1. Sigmoid

The logistic sigmoid maps any real-valued input into the range $(0, 1)$, producing the characteristic "S"-shaped curve.

**Definition:**

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Gradient:**

$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

The derivative is expressed entirely in terms of its own output—an elegant property that simplifies backpropagation.

| Property | Detail |
| :--- | :--- |
| **Range** | $(0, 1)$ |
| **Use Cases** | Binary classification output layers, gating mechanisms (LSTMs, GRUs) |
| **Advantages** | Smooth, differentiable; output interpretable as probability |
| **Limitations** | Suffers from **vanishing gradients** for large $\|x\|$; output is not zero-centered |

```python path=null start=null
class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))
```

---

### 2. Softmax

Softmax generalizes the sigmoid to multi-class settings. It converts a vector of raw logits into a **probability distribution** where all outputs sum to 1.

**Definition (for element $i$):**

$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

**Gradient (element-wise diagonal approximation):**

$$\text{Softmax}'(x_i) = \text{Softmax}(x_i) \cdot (1 - \text{Softmax}(x_i))$$

> **Numerical Stability:** The implementation subtracts the maximum value from the input (`x - np.max(x)`) before exponentiation to prevent floating-point overflow—a standard practice known as the *max trick*.

| Property | Detail |
| :--- | :--- |
| **Range** | Each element in $(0, 1)$; vector sums to $1$ |
| **Use Cases** | Multi-class classification output layers |
| **Advantages** | Outputs a valid probability distribution; amplifies dominant logits |
| **Limitations** | Computationally heavier than element-wise functions; sensitive to outlier logits |

```python path=null start=null
class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)
```

---

### 3. TanH (Hyperbolic Tangent)

TanH is a rescaled sigmoid that maps inputs to $(-1, 1)$. Its **zero-centered** output avoids the systematic bias inherent in Sigmoid, leading to faster convergence during gradient descent.

**Definition:**

$$\text{TanH}(x) = \frac{2}{1 + e^{-2x}} - 1$$

**Gradient:**

$$\text{TanH}'(x) = 1 - \text{TanH}(x)^2$$

This follows from the identity $\text{sech}^2(x) = 1 - \tanh^2(x)$.

| Property | Detail |
| :--- | :--- |
| **Range** | $(-1, 1)$ |
| **Relationship to Sigmoid** | $\text{TanH}(x) = 2\sigma(2x) - 1$ |
| **Use Cases** | Hidden layers (especially in RNNs); anywhere zero-centered output is preferred |
| **Advantages** | Zero-centered output; stronger gradients than Sigmoid near $x = 0$ |
| **Limitations** | Still suffers from **vanishing gradients** at saturation regions |

```python path=null start=null
class TanH():
    def __call__(self, x):
        return 2 / (1 + np.exp(-2*x)) - 1

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)
```

---

### 4. ReLU (Rectified Linear Unit)

ReLU is the **de facto standard** activation for hidden layers in modern deep networks. Its piecewise linearity provides computational efficiency and mitigates the vanishing gradient problem.

**Definition:**

$$f(x) = \max(0, x)$$

**Gradient:**

$$f'(x) = \begin{cases} 1 & \text{if } x \geq 0 \\ 0 & \text{if } x < 0 \end{cases}$$

| Property | Detail |
| :--- | :--- |
| **Range** | $[0, \infty)$ |
| **Use Cases** | Default choice for hidden layers in CNNs, feedforward networks |
| **Advantages** | Computationally very fast (no exponentials); constant gradient of 1 prevents vanishing gradients; promotes sparsity |
| **Limitations** | **"Dying ReLU" problem**: neurons with consistently negative input have zero gradient and stop learning permanently |

```python path=null start=null
class ReLU():
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)
```

---

### 5. Leaky ReLU

Leaky ReLU addresses the dying ReLU problem by introducing a small, non-zero slope $\alpha$ for negative inputs, ensuring that neurons always have a gradient and can continue to learn.

**Definition:**

$$f(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha x & \text{if } x < 0 \end{cases}$$

**Gradient:**

$$f'(x) = \begin{cases} 1 & \text{if } x \geq 0 \\ \alpha & \text{if } x < 0 \end{cases}$$

| Property | Detail |
| :--- | :--- |
| **Range** | $(-\infty, \infty)$ |
| **Hyperparameter** | $\alpha$ (default: $0.2$) — controls the "leak" for negative inputs |
| **Use Cases** | Drop-in replacement for ReLU where dying neurons are a concern |
| **Advantages** | No dead neurons; retains the computational simplicity of ReLU |
| **Limitations** | Performance is sensitive to the choice of $\alpha$; not always superior to standard ReLU in practice |

```python path=null start=null
class LeakyReLU():
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)
```

---

### 6. ELU (Exponential Linear Unit)

ELU uses an **exponential curve** for negative inputs rather than a straight line, providing a smoother transition at $x = 0$. This smoothness can aid optimization, and the negative saturation pushes mean activations closer to zero.

**Definition:**

$$f(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha(e^x - 1) & \text{if } x < 0 \end{cases}$$

**Gradient:**

$$f'(x) = \begin{cases} 1 & \text{if } x \geq 0 \\ f(x) + \alpha & \text{if } x < 0 \end{cases}$$

> **Implementation Note:** The gradient for $x < 0$ is computed as $f(x) + \alpha$ rather than $\alpha e^x$, reusing the forward pass output for efficiency.

| Property | Detail |
| :--- | :--- |
| **Range** | $[-\alpha, \infty)$ |
| **Hyperparameter** | $\alpha$ (default: $0.1$) — controls the negative saturation value |
| **Use Cases** | Deep networks where near-zero mean activations are beneficial |
| **Advantages** | Smooth ($C^1$ continuous when $\alpha = 1$); pushes mean activations toward zero |
| **Limitations** | Slower to compute than ReLU due to exponentiation; $\alpha$ requires tuning |

```python path=null start=null
class ELU():
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, x):
        return np.where(x >= 0.0, 1, self.__call__(x) + self.alpha)
```

---

### 7. SELU (Scaled Exponential Linear Unit)

SELU enables **Self-Normalizing Neural Networks (SNNs)**, where activations automatically converge toward zero mean and unit variance across layers, potentially eliminating the need for Batch Normalization.

**Definition:**

$$f(x) = \lambda \begin{cases} x & \text{if } x \geq 0 \\ \alpha(e^x - 1) & \text{if } x < 0 \end{cases}$$

**Gradient:**

$$f'(x) = \lambda \begin{cases} 1 & \text{if } x \geq 0 \\ \alpha e^x & \text{if } x < 0 \end{cases}$$

The self-normalizing property requires two mathematically derived constants:

| Constant | Value |
| :--- | :--- |
| $\lambda$ (scale) | $\approx 1.0507$ |
| $\alpha$ | $\approx 1.6733$ |

| Property | Detail |
| :--- | :--- |
| **Use Cases** | Deep feedforward networks where Batch Normalization is undesirable |
| **Advantages** | Internal normalization; robust to vanishing/exploding gradients |
| **Requirements** | Weights must be initialized with **LeCun Normal Initialization** for the self-normalizing property to hold |

```python path=null start=null
class SELU():
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def __call__(self, x):
        return self.scale * np.where(x >= 0.0, x, self.alpha*(np.exp(x)-1))

    def gradient(self, x):
        return self.scale * np.where(x >= 0.0, 1, self.alpha * np.exp(x))
```

---

### 8. SoftPlus

SoftPlus is a **smooth approximation of ReLU**. Where ReLU has a non-differentiable kink at $x = 0$, SoftPlus provides a gradual, infinitely differentiable transition.

**Definition:**

$$f(x) = \ln(1 + e^x)$$

**Gradient:**

$$f'(x) = \frac{1}{1 + e^{-x}} = \sigma(x)$$

A notable mathematical relationship: the derivative of SoftPlus is exactly the **Sigmoid function**.

| Property | Detail |
| :--- | :--- |
| **Range** | $(0, \infty)$ |
| **Use Cases** | Situations where a smooth, strictly positive output is required (e.g., parameterizing variance in probabilistic models) |
| **Advantages** | Smooth and differentiable everywhere; asymptotically approaches ReLU for large $x$ |
| **Limitations** | Slightly more expensive than ReLU; near-zero gradient for very negative inputs |

```python path=null start=null
class SoftPlus():
    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        return 1 / (1 + np.exp(-x))
```

---

## Comparative Analysis

The figure below visualizes all eight activation functions and their gradients side by side over the range $[-5, 5]$. Blue solid lines represent the **forward pass output**, and red dashed lines represent the **gradient**.

![Comparative Analysis of Activation Functions](Comparative%20Analysis%20of%20Activation%20Functions.png)

Key observations from the comparison:

- **Sigmoid** and **Softmax** produce small, bounded outputs with gradients that vanish at the extremes.
- **TanH** exhibits stronger gradients than Sigmoid near $x = 0$ but still saturates.
- **ReLU** shows the distinctive hard cutoff at zero — constant gradient of 1 in the positive region, exactly 0 elsewhere.
- **Leaky ReLU** retains the small (0.2) gradient in the negative region, preventing dead neurons.
- **ELU** and **SELU** introduce smooth exponential curves below zero, with SELU's scaling producing visibly larger magnitudes.
- **SoftPlus** closely mirrors ReLU but with a smooth curve around the origin.

---

## Implementation Design

All activation functions follow a consistent interface:

| Method | Description |
| :--- | :--- |
| `__call__(x)` | Computes the forward pass: applies the activation function to input `x` |
| `gradient(x)` | Computes the analytical derivative with respect to `x`, used for backpropagation |

Where applicable, classes accept hyperparameters (e.g., `alpha` for Leaky ReLU and ELU) through the constructor. SELU uses fixed, mathematically derived constants and requires no user-specified parameters.

---

## Usage

```python path=null start=null
import numpy as np

# Instantiate
relu = ReLU()
leaky = LeakyReLU(alpha=0.2)
elu = ELU(alpha=1.0)

# Sample input
x = np.linspace(-5, 5, 100)

# Forward pass
y = relu(x)

# Gradient (for backpropagation)
dy = relu.gradient(x)
```

To reproduce the comparative visualization:

```python path=null start=null
import matplotlib.pyplot as plt

activations = {
    "Sigmoid": Sigmoid(), "Softmax": Softmax(),
    "TanH": TanH(), "ReLU": ReLU(),
    "LeakyReLU": LeakyReLU(alpha=0.2), "ELU": ELU(alpha=1.0),
    "SELU": SELU(), "SoftPlus": SoftPlus()
}

x = np.linspace(-5, 5, 400)
fig, axes = plt.subplots(4, 2, figsize=(15, 20))

for i, (name, func) in enumerate(activations.items()):
    ax = axes.flatten()[i]
    ax.plot(x, func(x), label=f'{name} Output', color='blue', linewidth=2)
    ax.plot(x, func.gradient(x), label=f'{name} Gradient', color='red', linestyle='--', linewidth=1.5)
    ax.set_title(f"{name} Activation & Gradient", fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.axvline(0, color='black', lw=1, alpha=0.5)
    ax.legend()
    ax.grid(True, which='both', linestyle=':', alpha=0.7)

plt.suptitle("Comparative Analysis of Activation Functions", fontsize=20, y=1.02)
plt.tight_layout()
plt.show()
```

---

## Key Takeaways

1. **Sigmoid and TanH** are historically important but suffer from vanishing gradients in deep networks. They remain useful in output layers (Sigmoid for binary classification, TanH for bounded outputs) and gating mechanisms.
2. **ReLU** dominates as the default hidden-layer activation due to its simplicity and effective gradient flow. Its primary weakness—dead neurons—is addressed by the Leaky ReLU, ELU, and SELU variants.
3. **SELU** is unique in providing a **self-normalizing** property, but it requires strict conditions (LeCun initialization, fully connected layers) to realize this benefit.
4. **SoftPlus** serves as a theoretically elegant, smooth alternative to ReLU, with its derivative being exactly the Sigmoid function.
5. **Softmax** occupies a distinct role as a **vector-to-probability** transformation, making it indispensable for multi-class classification output layers. The *max trick* for numerical stability is a critical implementation detail.
