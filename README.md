# Attention Is All You Code

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers/)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-ArpitKadam-181717?style=flat&logo=github)](https://github.com/ArpitKadam/Attention-Is-All-You-Code)

> A ground-up study of modern machine learning and large language models — from the mathematics of a single activation function to the reconstruction of billion-parameter transformers — implemented explicitly, derived rigorously, and documented as a coherent curriculum.

⚠️ **Work in Progress.** This repository is under active development. New modules are added regularly; existing ones may evolve.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [The Curriculum](#the-curriculum)
   - [Machine Learning From Scratch](#1-machine-learning-from-scratch)
   - [LLMs From Scratch](#2-llms-from-scratch)
   - [Fine-Tuning 101](#3-fine-tuning-101)
   - [Fine-Tuning Jobs](#4-fine-tuning-jobs)
   - [LLM Applications](#5-llm-applications)
   - [Utilities](#6-utilities)
4. [Diagnostics & Tooling](#diagnostics--tooling)
5. [Getting Started](#getting-started)
6. [Submodules](#submodules)
7. [License](#license)

---

## Overview

**Attention Is All You Code** is a monorepo that reconstructs machine learning from first principles. Its guiding conviction is that the most durable way to understand a system is to build it — so every core idea here is implemented explicitly rather than imported from a high-level abstraction.

The repository is organized as a progressive curriculum. It begins with the mathematical primitives that underpin all neural networks (activation functions and optimizers in pure NumPy), advances through the construction and training of decoder-only transformers (from a 124M-parameter GPT-2 to modern multi-billion-parameter architectures), and culminates in the practical arts of **fine-tuning** and **applied LLM engineering** — retrieval-augmented generation, agentic workflows, quantization, and generative modeling.

---

## Repository Structure

```text
Attention-Is-All-You-Code/
├── ML-from-Scratch/          # Activation functions & optimizers in pure NumPy
│   ├── Activation-Functions/
│   └── Optimizers/
├── LLMs-from-Scratch/        # Decoder-only transformers, built layer by layer
│   ├── GPT-2 (124M) Model/       # 14-chapter GPT-2 from scratch
│   ├── Gemma3-270M/              # (submodule) from-scratch pretraining
│   ├── Nano-GPT-OSS-580M/        # (submodule) MoE + efficient attention
│   ├── Olmo-3-7B/                # 7B reconstruction with weight loading
│   └── Qwen3_Coder_30B_A3B_Instruct/  # (submodule) sparse MoE, code-tuned
├── FineTuning-101/           # 8-chapter fine-tuning curriculum
├── FineTuning-Jobs/          # Applied end-to-end fine-tuning projects
├── LLM-Applications/         # RAG, decoding, diffusion, agentic app
│   └── Guidely.ai/               # (submodule) LangGraph travel agent
├── utils/                    # Cheatsheet & notebook tooling
├── full_diagnostics.py       # PyTorch / GPU benchmark & profiling suite
├── the-verdict.txt           # Training corpus for GPT-2 pretraining
└── requirements.txt
```

---

## The Curriculum

### 1. Machine Learning From Scratch

[`ML-from-Scratch/`](./ML-from-Scratch) — The mathematical foundations, in pure NumPy.

- **Activation Functions** — eight functions (Sigmoid, Softmax, TanH, ReLU, Leaky ReLU, ELU, SELU, SoftPlus) with forward passes and analytical gradients.
- **Optimizers** — five gradient-based optimizers (BGD, SGD, Momentum, RMSProp, Adam), derived and visualized on loss curves and 2D optimization paths.

### 2. LLMs From Scratch

[`LLMs-from-Scratch/`](./LLMs-from-Scratch) — Decoder-only transformers, from 124M to 30B parameters.

A progression from a token-by-token GPT-2 (124M) to faithful reconstructions of Gemma3-270M, Nano-GPT-OSS-580M, OLMo-3 7B, and Qwen3 Coder 30B — introducing RMSNorm, RoPE/YaRN, Grouped-Query Attention, SwiGLU, and Mixture-of-Experts along the way.

### 3. Fine-Tuning 101

[`FineTuning-101/`](./FineTuning-101) — An eight-chapter tour of model adaptation.

Classification and instruction fine-tuning, LoRA/QLoRA, knowledge distillation, and quantization (PTQ, QAT, GPTQ, AWQ, GGUF).

### 4. Fine-Tuning Jobs

[`FineTuning-Jobs/`](./FineTuning-Jobs) — Applied, end-to-end fine-tuning projects.

Llama 2 instruction tuning, DistilBERT emotion and multi-label classification, and BART summarization — each with Weights & Biases / Weave experiment tracking.

### 5. LLM Applications

[`LLM-Applications/`](./LLM-Applications) — Putting models to work.

Attention visualization (BertViz), retrieval-augmented generation over source code (LangChain + Chroma + Groq), controllable decoding (beam search, nucleus sampling), text-to-image diffusion, and **Guidely.ai** — a LangGraph agentic web application.

### 6. Utilities

[`utils/`](./utils) — Shared reference material and tooling.

A Hugging Face quick-reference cheatsheet and a notebook-metadata sanitizer wired into pre-commit.

---

## Diagnostics & Tooling

**`full_diagnostics.py`** is a standalone PyTorch diagnostics and benchmark suite for validating a workstation before training. It reports system and PyTorch build information, CUDA availability, and runs performance benchmarks — including attention profiling and an **FP16 vs. BF16** comparison for transformer workloads.

```bash
python full_diagnostics.py
```

---

## Getting Started

### Prerequisites

Install the full dependency set:

```bash
pip install -r requirements.txt
```

The stack centers on PyTorch, the Hugging Face ecosystem (`transformers`, `datasets`, `accelerate`, `peft`), and the standard scientific-Python tools (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`), with `tiktoken` for tokenization, `bitsandbytes` for quantization, and `wandb`/`weave` for experiment tracking.

### Recommended Path

1. Start with **[ML-from-Scratch](./ML-from-Scratch)** for the mathematical primitives.
2. Move to **[LLMs-from-Scratch/GPT-2 (124M) Model](./LLMs-from-Scratch/GPT-2%20%28124M%29%20Model)** to build a transformer end to end.
3. Continue through the modern architectures in **[LLMs-from-Scratch](./LLMs-from-Scratch)**.
4. Learn adaptation in **[FineTuning-101](./FineTuning-101)** and apply it in **[FineTuning-Jobs](./FineTuning-Jobs)**.
5. Build real systems in **[LLM-Applications](./LLM-Applications)**.

Every module contains its own detailed README with derivations, configurations, and usage instructions.

---

## Submodules

Several components are maintained as independent repositories and included here as Git submodules:

| Submodule | Repository |
| :--- | :--- |
| `LLM-Applications/Guidely.ai` | [ArpitKadam/Guidely.ai](https://github.com/ArpitKadam/Guidely.ai) |
| `LLMs-from-Scratch/Gemma3-270M` | [ArpitKadam/Gemma3-270M](https://github.com/ArpitKadam/Gemma3-270M) |
| `LLMs-from-Scratch/Nano-GPT-OSS-580M` | [ArpitKadam/Nano-GPT-OSS-580M](https://github.com/ArpitKadam/Nano-GPT-OSS-580M) |
| `LLMs-from-Scratch/Qwen3_Coder_30B_A3B_Instruct` | [ArpitKadam/Qwen3_Coder_30B_A3B_Instruct](https://github.com/ArpitKadam/Qwen3_Coder_30B_A3B_Instruct) |

Clone with submodules initialized:

```bash
git clone --recurse-submodules https://github.com/ArpitKadam/Attention-Is-All-You-Code
# or, in an existing clone:
git submodule update --init --recursive
```

---

## License

Released under the [MIT License](LICENSE).

---

*Built and maintained by [Arpit Kadam](https://github.com/ArpitKadam) — implementing machine learning concepts from scratch.*
