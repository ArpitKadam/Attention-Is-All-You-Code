# Building GPT-2 (124M) From Scratch

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![tiktoken](https://img.shields.io/badge/tiktoken-BPE-4B8BBE?style=flat)](https://github.com/openai/tiktoken)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-ArpitKadam-181717?style=flat&logo=github)](https://github.com/ArpitKadam/Attention-Is-All-You-Code)

> A fourteen-chapter, ground-up reconstruction of OpenAI's **GPT-2 (124M parameters)** — from raw text tokenization to a fully trained autoregressive transformer with OpenAI-compatible weight loading. Built with PyTorch, using no high-level modeling abstractions.

---

## Table of Contents

1. [Overview](#overview)
2. [Architectural Summary](#architectural-summary)
3. [The Curriculum](#the-curriculum)
   - [Part I — Data Foundations (CHP 01–02)](#part-i--data-foundations)
   - [Part II — The Attention Mechanism (CHP 03–06)](#part-ii--the-attention-mechanism)
   - [Part III — Model Construction (CHP 07–09)](#part-iii--model-construction)
   - [Part IV — Training & Evaluation (CHP 10–12)](#part-iv--training--evaluation)
   - [Part V — Generation & Weights (CHP 13–14)](#part-v--generation--weights)
4. [Model Configuration](#model-configuration)
5. [Repository Contents](#repository-contents)
6. [Getting Started](#getting-started)
7. [Key Takeaways](#key-takeaways)

---

## Overview

This module documents the complete construction of a decoder-only transformer language model that is architecturally identical to OpenAI's **GPT-2 Small (124M)**. Rather than importing a pre-built model, every component — the byte-pair tokenizer interface, the self-attention mechanism, the transformer block, the training loop, and the sampling strategies — is implemented explicitly and derived step by step.

The curriculum is organized as fourteen sequential notebooks (`CHP_01` … `CHP_14`). Each chapter builds directly on its predecessor, progressing from individual tokens to a model capable of generating coherent text and interoperating with OpenAI's officially released pretrained weights.

The training corpus is **`the-verdict.txt`** (a public-domain short story by Edith Wharton), which serves as a compact, self-contained dataset for demonstrating the full pretraining loop on modest hardware.

---

## Architectural Summary

GPT-2 is a **decoder-only, autoregressive transformer**. Its forward pass composes the following elements, each constructed in this module:

- **Byte-Pair Encoding (BPE)** tokenization via `tiktoken`
- **Token embeddings** combined additively with **learned absolute positional embeddings**
- A stack of **transformer blocks**, each containing:
  - **Causal (masked) multi-head self-attention**
  - A **position-wise feed-forward network** with **GELU** activation
  - **Layer normalization** (pre-norm) and **residual (shortcut) connections**
- A final **layer norm** and a **linear output head** projecting to the vocabulary

---

## The Curriculum

### Part I — Data Foundations

| Chapter | Title | Concepts |
| :--- | :--- | :--- |
| **CHP 01** | GPT Tokenizer & Dataloaders | Regex tokenization, token-ID vocabularies, a custom tokenizer class, special context tokens (`<\|endoftext\|>`, `<\|unk\|>`), byte-pair encoding, sliding-window data sampling, PyTorch `Dataset`/`DataLoader` |
| **CHP 02** | Vector Embeddings | Token embedding layers, the semantics of embedding lookups, learned absolute positional embeddings, combining token and positional signals |

### Part II — The Attention Mechanism

| Chapter | Title | Concepts |
| :--- | :--- | :--- |
| **CHP 03** | Simplified Self-Attention (no weights) | Attention scores as dot products, softmax normalization, context vectors — the intuition before trainable parameters |
| **CHP 04** | Self-Attention & Multi-Head Attention (trainable weights) | Query/Key/Value projections ($W_q, W_k, W_v$), scaled dot-product attention ($1/\sqrt{d_k}$), context-vector computation, extension to multiple heads |
| **CHP 05** | Causal Attention Mechanism | Masking future tokens, the causal (lower-triangular) mask, dropout on attention weights, a reusable `CausalAttention` class |
| **CHP 06** | Multi-Head Attention | Efficient multi-head attention via tensor weight splits (single projection reshaped into heads) |

### Part III — Model Construction

| Chapter | Title | Concepts |
| :--- | :--- | :--- |
| **CHP 07** | GPT Architecture (Bird's-Eye View) | Configuration dictionary, dummy model scaffold, layer normalization, GELU feed-forward network, shortcut connections, the assembled transformer block |
| **CHP 08** | Coding the Entire GPT Model | Full assembly of the 124M-parameter `GPTModel` class end to end |
| **CHP 09** | Predicting the Next Token | Converting model logits into token predictions and iterative greedy text generation |

### Part IV — Training & Evaluation

| Chapter | Title | Concepts |
| :--- | :--- | :--- |
| **CHP 10** | Measuring the Loss Function | Cross-entropy loss for language modeling and its relationship to **perplexity** |
| **CHP 11** | Performance on a Real Dataset | Train/validation dataloaders over `the-verdict.txt`, a reusable loss-calculation utility, a single forward/backward pass |
| **CHP 12** | The Entire Pre-training Loop | The complete training loop, periodic evaluation, and visualization of training/validation loss curves |

### Part V — Generation & Weights

| Chapter | Title | Concepts |
| :--- | :--- | :--- |
| **CHP 13** | Temperature Scaling & Top-K Sampling | Probabilistic decoding — temperature scaling of logits, top-K truncation, and the combination of both for controllable generation |
| **CHP 14** | Saving & Loading Pretrained Weights | Checkpointing model and optimizer state, and loading OpenAI's official GPT-2 weights into the from-scratch architecture via `gpt_download3.py` |

---

## Model Configuration

The reference configuration matches GPT-2 Small:

| Hyperparameter | Value |
| :--- | :--- |
| Vocabulary size | 50,257 |
| Context length | 1,024 |
| Embedding dimension ($d_{model}$) | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Feed-forward expansion | 4× ($d_{model} \rightarrow 3072$) |
| Dropout | 0.1 |
| Parameter count | ~124 million |

---

## Repository Contents

```text
GPT-2 (124M) Model/
├── CHP_01_GPT_Tokenizer_and_Dataloaders.ipynb
├── CHP_02_Vector_Embedding.ipynb
├── CHP_03_Simplified_Self_Attention_without_weights.ipynb
├── CHP_04_Self_Attention_&_Multi_Head_Attention_with_Trainable_Weights.ipynb
├── CHP_05_Causal_Attention_Mechanism.ipynb
├── CHP_06_Multi_Head_Attention.ipynb
├── CHP_07_GPT_Model_Architecture_Bird_Eye_View.ipynb
├── CHP_08_Coding_Entire_GPT_Model.ipynb
├── CHP_09_Coding_GPT_to_Predict_the_Next_Token.ipynb
├── CHP_10_Measuring_the_LLM_loss_function.ipynb
├── CHP_11_LLM_performance_on_real_dataset.ipynb
├── CHP_12_Entire_LLM_Pre_training_Loop.ipynb
├── CHP_13_Temperature_Scaling_and_Top_K_Sampling_in_LLM.ipynb
├── CHP_14_Saving_and_loading_Pre_Trained_Weights.ipynb
└── gpt_download3.py          # Utility for downloading OpenAI's official GPT-2 weights
```

The training text `the-verdict.txt` is located at the repository root.

---

## Getting Started

### Prerequisites

```bash
pip install torch tiktoken numpy matplotlib tensorflow tqdm jupyter
```

> `tensorflow` is required only by `gpt_download3.py` (CHP 14) to parse OpenAI's original TensorFlow checkpoints.

### Recommended Path

The notebooks are designed to be studied **in order**. Begin with `CHP_01` and proceed sequentially; each chapter assumes the concepts and code from the previous ones.

```bash
jupyter notebook "CHP_01_GPT_Tokenizer_and_Dataloaders.ipynb"
```

---

## Key Takeaways

1. **A large language model is a composition of simple parts.** GPT-2's capability emerges from the repeated application of attention and feed-forward layers — each individually tractable.
2. **Attention is built incrementally.** Understanding self-attention is easiest when approached first without trainable weights (CHP 03), then with weights (CHP 04), then with causality (CHP 05), and finally at multi-head efficiency (CHP 06).
3. **The training objective is next-token prediction.** Cross-entropy loss and its exponential, perplexity, are the sole signals guiding pretraining.
4. **Decoding strategy shapes output.** Temperature and top-K sampling convert a deterministic argmax into controllable, diverse generation.
5. **From-scratch code is weight-compatible.** Because the architecture faithfully mirrors GPT-2, OpenAI's released weights load directly into it — the ultimate validation of a correct implementation.

---

*Part of the [Attention-Is-All-You-Code](https://github.com/ArpitKadam/Attention-Is-All-You-Code) project — implementing machine learning concepts from scratch.*
