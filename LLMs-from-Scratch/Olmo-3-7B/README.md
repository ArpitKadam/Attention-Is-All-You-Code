# OLMo-3 7B — Architecture From Scratch

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Weights-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/allenai/Olmo-3-7B-Instruct)
[![Model](https://img.shields.io/badge/Model-7B%20Parameters-green?style=flat)](https://huggingface.co/allenai/Olmo-3-7B-Instruct)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-ArpitKadam-181717?style=flat&logo=github)](https://github.com/ArpitKadam/Attention-Is-All-You-Code)

> A faithful, from-scratch PyTorch reconstruction of **AllenAI's OLMo-3 7B** decoder-only transformer. The model architecture is rebuilt component by component, then loaded with the official pretrained `safetensors` weights to reproduce inference — no `transformers` modeling classes involved.

---

## Table of Contents

1. [Overview](#overview)
2. [Model Configuration](#model-configuration)
3. [Architectural Components](#architectural-components)
   - [RMSNorm](#rmsnorm)
   - [SwiGLU Feed-Forward](#swiglu-feed-forward)
   - [Rotary Positional Embeddings with YaRN](#rotary-positional-embeddings-with-yarn)
   - [Grouped-Query Attention](#grouped-query-attention)
   - [Interleaved Sliding-Window & Full Attention](#interleaved-sliding-window--full-attention)
4. [Inference Pipeline](#inference-pipeline)
5. [Hardware & Environment](#hardware--environment)
6. [Repository Contents](#repository-contents)
7. [Getting Started](#getting-started)
8. [Key Takeaways](#key-takeaways)

---

## Overview

**OLMo-3** is AllenAI's fully open language-model family — open weights, open data, and open training recipe. This module reconstructs the **7B-parameter** variant (`allenai/Olmo-3-7B-Instruct`) entirely in explicit PyTorch, defining each layer type from first principles.

The workflow is deliberately transparent:

1. **Reconstruct** the exact OLMo-3 7B architecture as plain `nn.Module` classes.
2. **Download** the official pretrained weights from the Hugging Face Hub (`snapshot_download`, sharded `safetensors`).
3. **Map** every published tensor name (e.g. `model.embed_tokens.weight`) onto the from-scratch modules via a dedicated weight-loading routine.
4. **Generate** text with a minimal greedy, streaming decoder.

This makes the model's internals — attention masking, positional encoding, normalization placement — fully inspectable, while still producing outputs identical to the reference implementation.

---

## Model Configuration

The reconstruction targets the published OLMo-3 7B configuration:

| Hyperparameter | Value |
| :--- | :--- |
| Vocabulary size | 100,278 |
| Context length | 65,536 |
| Embedding dimension ($d_{model}$) | 4,096 |
| Transformer layers | 32 |
| Attention heads | 32 |
| Key-value heads | 32 |
| Head dimension | 128 |
| Feed-forward hidden dimension | 11,008 |
| Sliding-window size | 4,096 |
| Attention bias | None |
| Normalization | RMSNorm ($\epsilon = 10^{-6}$) |
| Positional encoding | RoPE (with optional YaRN scaling) |

---

## Architectural Components

### RMSNorm

Root Mean Square Layer Normalization normalizes activations by their root-mean-square without subtracting the mean, and rescales with a learned weight. Computation is performed in `float32` for numerical stability before casting back to the input dtype. OLMo-3 places normalization **after** the attention and feed-forward sub-blocks (`post_attn_layernorm`, `post_ff_layernorm`).

### SwiGLU Feed-Forward

The position-wise feed-forward network uses a **gated linear unit with a SiLU (Swish) gate**:

$$\text{FFN}(x) = W_3 \left( \text{SiLU}(W_1 x) \odot W_2 x \right)$$

Three bias-free linear projections (`fc1`, `fc2`, `fc3`) implement the gating, expanding to an 11,008-dimensional hidden space.

### Rotary Positional Embeddings with YaRN

Positions are encoded by rotating query and key vectors (RoPE) rather than adding positional vectors. The implementation supports **YaRN** (Yet another RoPE extrapolatioN) scaling — via `find_correction_dim` / `find_correction_range` — enabling the extension of the effective context window well beyond the base training length.

### Grouped-Query Attention

Attention is implemented as **Grouped-Query Attention (GQA)**, in which query heads are partitioned into groups that share key/value projections, reducing the key-value cache footprint during inference. Separate bias-free projections produce queries, keys, and values, with the group size derived as `n_heads / n_kv_heads`.

### Interleaved Sliding-Window & Full Attention

OLMo-3 alternates two attention regimes across its 32 layers:

- **Sliding-window attention** — each token attends only to the previous 4,096 tokens (local context, lower cost).
- **Full attention** — unrestricted causal attention, applied at every fourth layer.

Each `TransformerBlock` selects a **local mask** or a **global mask** according to its assigned `attn_type`, so long-range dependencies are periodically re-integrated while most layers remain computationally efficient.

---

## Inference Pipeline

| Stage | Mechanism |
| :--- | :--- |
| **Weight acquisition** | `snapshot_download` fetches sharded `safetensors`; shards are merged from `model.safetensors.index.json` |
| **Weight assignment** | `load_weights_into_olmo` copies each named tensor into the matching module, with shape validation |
| **Tokenization** | `tokenizers.Tokenizer` loaded from the model's `tokenizer.json`, wrapped by `OlmoTokenizer` with chat-template support |
| **Generation** | `generate_text_basic_stream` performs greedy (argmax) autoregressive decoding, yielding one token at a time and halting on the EOS token |

---

## Hardware & Environment

This model was **not** run on local/consumer hardware. All development and inference were performed on a **rented cloud GPU from [RunPod.io](https://www.runpod.io/)** — an **NVIDIA A40** (or a comparable data-center accelerator).

A 7-billion-parameter model in half precision occupies roughly **14–16 GB** for weights alone, before activations and the key-value cache. The recommended environment is therefore:

| Requirement | Recommended |
| :--- | :--- |
| GPU | NVIDIA A40 (48 GB) — or A100 (40/80 GB), L40S, or ≥24 GB equivalent |
| VRAM (bf16 inference) | ≥ 24 GB comfortably; ~16 GB minimum |
| CUDA | 12.x |
| PyTorch | 2.x with CUDA support |
| System RAM | ≥ 32 GB |
| Disk | ≥ 30 GB free (for the downloaded `safetensors` shards) |
| Precision | `bfloat16` |

> Run `Diagnostic.ipynb` first to confirm the GPU, driver, CUDA version, and available memory of the rented instance before loading weights.

---

## Repository Contents

```text
Olmo-3-7B/
├── Olmo3_7B.ipynb      # Architecture reconstruction, weight loading, and generation
└── Diagnostic.ipynb    # System / GPU diagnostic (device, memory, capability checks)
```

---

## Getting Started

### Prerequisites

```bash
pip install torch safetensors tokenizers huggingface_hub prompt_toolkit
```

Loading a 7B-parameter model requires a data-center GPU — see [Hardware & Environment](#hardware--environment). Run `Diagnostic.ipynb` first to verify the rented instance.

### Running

```bash
jupyter notebook Olmo3_7B.ipynb
```

The notebook will download the official weights from `allenai/Olmo-3-7B-Instruct` on first execution.

---

## Key Takeaways

1. **Modern 7B-scale models are assemblies of well-defined primitives** — RMSNorm, SwiGLU, RoPE, and GQA — each of which fits in a few dozen lines of PyTorch.
2. **Attention need not be uniform across depth.** Interleaving sliding-window and full attention trades a small amount of global reach for a large reduction in compute, recovered periodically by full-attention layers.
3. **Weight compatibility validates correctness.** Because the from-scratch modules mirror the published tensor layout exactly, the official `safetensors` weights load without adaptation — and the model generates identically to the reference.
4. **Context extension is an architectural choice.** YaRN-scaled RoPE demonstrates how the effective context length can be extended without retraining positional parameters.

---

*Part of the [Attention-Is-All-You-Code](https://github.com/ArpitKadam/Attention-Is-All-You-Code) project — implementing machine learning concepts from scratch.*
