# Fine-Tuning 101

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%2FQLoRA-purple?style=flat)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-ArpitKadam-181717?style=flat&logo=github)](https://github.com/ArpitKadam/Attention-Is-All-You-Code)

> A structured, eight-chapter curriculum on adapting pretrained language models to downstream tasks — spanning classification and instruction fine-tuning, parameter-efficient methods, knowledge distillation, and quantization.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [The Curriculum](#the-curriculum)
4. [Chapter Details](#chapter-details)
5. [Techniques Covered](#techniques-covered)
6. [Getting Started](#getting-started)
7. [Key Takeaways](#key-takeaways)

---

## Overview

Pretraining teaches a language model the general structure of language; **fine-tuning** specializes that general knowledge for a concrete task or behavior. This module surveys the modern fine-tuning toolkit through eight progressive notebooks, moving from the fundamentals of supervised adaptation to the compression techniques that make large models deployable.

The curriculum treats fine-tuning not as a single technique but as a family of methods, each addressing a distinct constraint — task specialization, instruction following, memory efficiency, model size, or inference latency.

---

## Prerequisites

The classification and instruction-tuning chapters build directly on the GPT-2 architecture developed in [`LLMs-from-Scratch/GPT-2 (124M) Model`](../LLMs-from-Scratch/GPT-2%20%28124M%29%20Model). Familiarity with transformer internals, tokenization, and the training loop is assumed. Later chapters rely on the Hugging Face ecosystem (`transformers`, `datasets`, `peft`, `trl`).

---

## The Curriculum

| Chapter | Title | Focus |
| :--- | :--- | :--- |
| **CHP 01** | Classification Fine-Tuning | Adapting GPT-2 for supervised classification |
| **CHP 02** | Instruction Fine-Tuning | Teaching a model to follow natural-language instructions |
| **CHP 03** | Fine-Tuning Gemma 2B | QLoRA fine-tuning of a modern instruction model |
| **CHP 04** | BERT Text Classification | Encoder fine-tuning with the `Trainer` API |
| **CHP 05** | Knowledge Distillation (Neural Networks) | Teacher-student compression on image data |
| **CHP 06** | Knowledge Distillation (BERT) | Distilling a transformer classifier |
| **CHP 07** | LLM Quantization | Reducing model precision for efficient inference |
| **CHP 08** | Unsloth Tutorial | Accelerated, memory-efficient fine-tuning |

---

## Chapter Details

### CHP 01 — Classification Fine-Tuning
Loads GPT-2 with pretrained weights and replaces its language-modeling head with a **classification head**. Covers dataset download, dataloader construction, the modified architecture, loss and accuracy computation, supervised fine-tuning on labeled data, and evaluation of the resulting classifier.

### CHP 02 — Instruction Fine-Tuning
Prepares an instruction dataset in the **Alpaca prompt format**, organizes examples into training batches with padding, applies **target-token masking** (so loss is computed only over the response), builds dataloaders, and fine-tunes the LLM to follow instructions. Concludes with qualitative evaluation of the instruction-tuned model.

### CHP 03 — Fine-Tuning Gemma 2B
Applies **QLoRA** — Low-Rank Adaptation (`peft.LoraConfig`) over a 4-bit quantized base model (`BitsAndBytesConfig`) — to fine-tune Google's Gemma 2B using the `trl.SFTTrainer`. Demonstrates parameter-efficient adaptation of a multi-billion-parameter model on commodity hardware.

### CHP 04 — BERT Text Classification
Fine-tunes `BertForSequenceClassification` with the Hugging Face `Trainer` and `TrainingArguments`, then serves predictions through the `pipeline` API — the canonical encoder-based classification workflow.

### CHP 05 — Knowledge Distillation in Neural Networks
Introduces the **teacher-student paradigm** on a vision dataset (via `torchvision`): a large teacher network transfers its "dark knowledge" (softened output distributions) to a compact student trained on temperature-scaled soft targets.

### CHP 06 — Knowledge Distillation in BERT
Extends distillation to transformers, compressing a fine-tuned BERT classifier into a smaller student with a learning-rate scheduler and combined hard/soft-label objectives, evaluated by accuracy against the teacher.

### CHP 07 — LLM Quantization
A comprehensive tour of precision reduction:
- **Post-Training Quantization (PTQ)** — both **dynamic** and **static**
- **Quantization-Aware Training (QAT)**
- **GPTQ** (Gradient Post-Training Quantization)
- **AWQ** (Activation-Aware Weight Quantization)
- **GGML / GGUF** formats for CPU-efficient inference

### CHP 08 — Unsloth Tutorial
Uses the **Unsloth** library (`FastLanguageModel`) with `trl.SFTTrainer`/`SFTConfig` to fine-tune LLMs with substantially reduced memory and increased throughput, including PEFT adapter merging.

---

## Techniques Covered

| Technique | Category | Chapters |
| :--- | :--- | :--- |
| Supervised classification head | Full fine-tuning | 01, 04 |
| Instruction / SFT | Behavioral alignment | 02, 03, 08 |
| LoRA / QLoRA | Parameter-efficient FT | 03, 08 |
| Knowledge distillation | Model compression | 05, 06 |
| Quantization (PTQ/QAT/GPTQ/AWQ/GGUF) | Precision reduction | 07 |

---

## Getting Started

### Prerequisites

```bash
pip install torch transformers datasets peft trl bitsandbytes accelerate tiktoken scikit-learn matplotlib
```

> Chapters 03 and 08 target GPU environments (e.g. Google Colab). Chapters 01–02 reuse `gpt_download3.py` (included in this directory) to obtain GPT-2 weights.

### Running

Study the notebooks in order:

```bash
jupyter notebook CHP_01_Classification_FineTuning.ipynb
```

---

## Key Takeaways

1. **Fine-tuning is task-shaped.** Classification swaps the output head; instruction tuning reshapes behavior with formatted prompts and response masking.
2. **Parameter-efficient methods make scale accessible.** LoRA and QLoRA adapt billion-parameter models by training a tiny fraction of the weights over a quantized base.
3. **Distillation trades size for a small accuracy cost.** A student model can inherit most of a teacher's competence at a fraction of the parameters.
4. **Quantization is the deployment bridge.** PTQ, QAT, GPTQ, AWQ, and GGUF each occupy a different point on the accuracy-versus-efficiency frontier.

---

*Part of the [Attention-Is-All-You-Code](https://github.com/ArpitKadam/Attention-Is-All-You-Code) project — implementing machine learning concepts from scratch.*
