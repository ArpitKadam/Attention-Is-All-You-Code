# LLMs From Scratch

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Ecosystem-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Transformers](https://img.shields.io/badge/Architecture-Decoder--Only-purple?style=flat)](https://arxiv.org/abs/1706.03762)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-ArpitKadam-181717?style=flat&logo=github)](https://github.com/ArpitKadam/Attention-Is-All-You-Code)

> A progression of decoder-only transformer language models rebuilt from first principles — from a 124M-parameter GPT-2 constructed token by token, up to modern multi-billion-parameter architectures (Gemma, GPT-OSS, OLMo, Qwen) reconstructed layer by layer.

---

## Table of Contents

1. [Overview](#overview)
2. [The Model Progression](#the-model-progression)
3. [Model Details](#model-details)
4. [Architectural Themes](#architectural-themes)
5. [Hardware & Environment](#hardware--environment)
6. [Working With Submodules](#working-with-submodules)
7. [Getting Started](#getting-started)

---

## Overview

**`LLMs-from-Scratch`** traces the evolution of the decoder-only transformer through a series of complete, explicit implementations. The series begins with the canonical **GPT-2 (124M)** — built end to end from raw text — and advances to faithful reconstructions of contemporary open-weight models, each of which introduces the architectural innovations that define the current generation of language models.

The intent is pedagogical rigor: rather than calling a pre-built model, each project defines every layer as plain PyTorch, then (where applicable) loads official pretrained weights to validate the reconstruction against the reference.

---

## The Model Progression

| Model | Parameters | Key Innovations | Location |
| :--- | :--- | :--- | :--- |
| **[GPT-2 (124M)](./GPT-2%20%28124M%29%20Model)** | 124M | Learned absolute positions, causal MHA, GELU FFN — built from scratch and trained | In-repo |
| **[Gemma3-270M](./Gemma3-270M)** | 270M | From-scratch pretraining pipeline, TensorBoard tracking | Submodule |
| **[Nano-GPT-OSS-580M](./Nano-GPT-OSS-580M)** | 580M | MoE, sliding-window + full attention, RoPE + YaRN, GQA, SwiGLU | Submodule |
| **[OLMo-3 7B](./Olmo-3-7B)** | 7B | Interleaved sliding/full attention, YaRN RoPE, GQA, SwiGLU — weight-loaded | In-repo |
| **[Qwen3 Coder 30B A3B](./Qwen3_Coder_30B_A3B_Instruct)** | 30B (3B active) | Sparse Mixture-of-Experts, code-specialized instruction tuning | Submodule |

---

## Model Details

### GPT-2 (124M) — The Foundation

A fourteen-chapter, ground-up construction of OpenAI's GPT-2 Small: tokenization, embeddings, the attention mechanism (from weightless intuition to efficient multi-head), the full transformer, the training loop, sampling strategies, and OpenAI-compatible weight loading. **The recommended starting point for the entire series.**

### Gemma3-270M — Full Pretraining Pipeline

A 270M-parameter decoder-only model built and **trained from scratch**, following the architectural principles of Google's Gemma family. Includes an end-to-end pretraining workflow, a diagnostic/analysis suite, TensorBoard experiment tracking, and automated visualization of training dynamics.

### Nano-GPT-OSS-580M — Efficient Attention & Sparsity

A 580M-parameter model exploring efficient computation: alternating **sliding-window and full-context attention**, **Mixture-of-Experts** feed-forward layers (top-1 routing), **RoPE with YaRN** scaling, **Grouped-Query Attention**, **SwiGLU**, and **RMSNorm**. Trained on TinyStories as a research platform for efficient architectures.

### OLMo-3 7B — Modern 7B Reconstruction

A faithful, from-scratch reconstruction of AllenAI's fully open OLMo-3 7B, loaded with the official `safetensors` weights. Demonstrates interleaved sliding/full attention, YaRN-scaled RoPE, Grouped-Query Attention, and SwiGLU at 7-billion-parameter scale.

### Qwen3 Coder 30B A3B Instruct — Sparse MoE at Scale

An instruction-tuned, code-specialized model built on a **sparse Mixture-of-Experts** design: 30B total parameters with only ~3B active per token. Optimized for multi-language code generation with support for efficient quantized inference.

---

## Architectural Themes

Reading the series in order surfaces the trajectory of transformer design:

| Theme | Early (GPT-2) | Modern (OLMo-3 / Qwen3 / Nano-OSS) |
| :--- | :--- | :--- |
| **Normalization** | LayerNorm | RMSNorm |
| **Positional encoding** | Learned absolute | Rotary (RoPE), extended with YaRN |
| **Attention** | Dense multi-head | Grouped-Query + sliding-window/full interleaving |
| **Feed-forward** | GELU MLP | SwiGLU, and sparse Mixture-of-Experts |
| **Capacity scaling** | Dense (all params active) | Sparse activation (MoE) |

---

## Hardware & Environment

None of these models were trained or run on local/consumer hardware. All development, training, and inference were carried out on **rented cloud GPUs from [RunPod.io](https://www.runpod.io/)** — primarily an **NVIDIA A40 (48 GB)** or a comparable data-center accelerator (A100, L40S).

GPU memory requirements scale with parameter count. The following are practical guidelines for the reference `bfloat16` setup:

| Model | Parameters | Indicative GPU VRAM |
| :--- | :--- | :--- |
| GPT-2 (124M) | 124M | ~8 GB (trains comfortably on an A40) |
| Gemma3-270M | 270M | ~12 GB (pretraining) |
| Nano-GPT-OSS-580M | 580M | ~16 GB (training) |
| OLMo-3 7B | 7B | ~16 GB minimum, ≥24 GB comfortable (inference) |
| Qwen3 Coder 30B A3B | 30B (3B active) | Large / multi-GPU; quantization recommended |

**Reference environment:** NVIDIA A40 (48 GB) · CUDA 12.x · PyTorch 2.x · ≥32 GB system RAM · `bfloat16` precision.

> A RunPod A40 instance comfortably handles the entire series up to OLMo-3 7B. Each model's `Diagnostic.ipynb` (where present) reports the live GPU, driver, and CUDA details of the rented instance.

---

## Working With Submodules

Gemma3-270M, Nano-GPT-OSS-580M, and Qwen3 Coder 30B A3B are maintained as **Git submodules**, each in its own repository. Initialize them with:

```bash
git submodule update --init --recursive
```

Each submodule ships its own detailed `README.md`, model configuration, and training/inference notebooks.

---

## Getting Started

### Prerequisites

```bash
pip install torch tiktoken transformers safetensors tokenizers huggingface_hub numpy matplotlib
```

### Recommended Path

1. **Begin with [GPT-2 (124M)](./GPT-2%20%28124M%29%20Model)** to establish the transformer fundamentals.
2. Progress to **Gemma3-270M** for a complete from-scratch pretraining pipeline.
3. Study **Nano-GPT-OSS-580M** and **OLMo-3 7B** for modern efficient-attention and normalization techniques.
4. Explore **Qwen3 Coder 30B A3B** for sparse Mixture-of-Experts at scale.

---

*Part of the [Attention-Is-All-You-Code](https://github.com/ArpitKadam/Attention-Is-All-You-Code) project — implementing machine learning concepts from scratch.*
