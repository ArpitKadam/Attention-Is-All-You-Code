# Fine-Tuning Jobs

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers/)
[![Weights & Biases](https://img.shields.io/badge/W%26B-Tracking-FFBE00?style=flat&logo=weightsandbiases&logoColor=black)](https://wandb.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-ArpitKadam-181717?style=flat&logo=github)](https://github.com/ArpitKadam/Attention-Is-All-You-Code)

> A collection of applied, end-to-end fine-tuning projects on real datasets — covering causal LLM instruction tuning, encoder classification (single- and multi-label), and sequence-to-sequence summarization, each with experiment tracking.

---

## Table of Contents

1. [Overview](#overview)
2. [Projects](#projects)
3. [Project Details](#project-details)
4. [Common Workflow](#common-workflow)
5. [Getting Started](#getting-started)
6. [Key Takeaways](#key-takeaways)

---

## Overview

Where [`FineTuning-101`](../FineTuning-101) teaches the *techniques* of model adaptation, **`FineTuning-Jobs`** applies them to complete, self-contained tasks. Each notebook is a full pipeline: dataset acquisition, preprocessing, model selection, training, evaluation, and experiment tracking via **Weights & Biases** and **Weave**.

The four projects deliberately span the three canonical transformer task families — **causal (decoder-only)**, **encoder (classification)**, and **sequence-to-sequence** — providing a practical cross-section of applied NLP fine-tuning.

---

## Projects

| Project | Base Model | Task | Paradigm |
| :--- | :--- | :--- | :--- |
| Llama 2 Fine-Tuning | Llama-2-7B-Chat | Instruction / chat | Causal LM + QLoRA |
| DistilBERT — Emotion Recognition | DistilBERT | Single-label classification | Encoder |
| DistilBERT — Multi-Label Classification | DistilBERT | Multi-label classification | Encoder |
| BART Summarization | BART | Abstractive summarization | Seq2Seq |

---

## Project Details

### `Fine_Tune_Llama2_model.ipynb`
Fine-tunes **Meta's Llama-2-7B-Chat** using **LoRA/QLoRA** (`peft.LoraConfig`, `PeftModel`) and the `trl.SFTTrainer`/`SFTConfig`. Includes gated-model authentication via the Hugging Face Hub, dataset loading, memory management (`gc`, locale handling for large-scale training), and generation from the adapted model. Demonstrates parameter-efficient instruction tuning of a 7B causal LLM.

### `Fine_Tuning_DistilBERT_for_Emotion_Recognition.ipynb`
Fine-tunes **DistilBERT** (`AutoModelForSequenceClassification`) for single-label emotion classification. Uses the `Trainer` API, reports accuracy, F1, recall, and precision with a full `classification_report`, and visualizes results with Matplotlib and Seaborn. Tracked with W&B / Weave.

### `FineTune_DistilBERT_for_Multi_Label_Classification.ipynb`
Adapts DistilBERT for **multi-label** classification, where each example may carry several labels simultaneously. Employs `MultiLabelBinarizer` for target encoding, a custom `Dataset`, sigmoid-based multi-label evaluation (`f1_score`, `accuracy_score`, `precision_score`), and Hub authentication for model publishing.

### `Summarization_Fine_Tuning_using_BART.ipynb`
Fine-tunes **BART** (`AutoModelForSeq2SeqLM`) for **abstractive summarization** using `DataCollatorForSeq2Seq` and the `Seq2Seq`-style `Trainer`. Generates summaries through the `pipeline` API and tracks the run with W&B / Weave.

---

## Common Workflow

Each project follows the same disciplined structure:

1. **Authenticate** — log in to the Hugging Face Hub (and Google Drive / Colab secrets where applicable).
2. **Load & preprocess** — acquire the dataset via `datasets.load_dataset`, tokenize, and format.
3. **Configure** — select the base model, define `TrainingArguments`, and (for LLMs) attach LoRA adapters.
4. **Train** — run the `Trainer` / `SFTTrainer`, logging metrics to W&B and Weave.
5. **Evaluate** — compute task-appropriate metrics and inspect qualitative outputs.

---

## Getting Started

### Prerequisites

```bash
pip install torch transformers datasets peft trl accelerate bitsandbytes wandb weave scikit-learn pandas matplotlib seaborn
```

> These notebooks are designed for GPU environments (e.g. Google Colab) and require a Hugging Face access token. The Llama 2 project additionally requires accepting Meta's gated-model license.

### Running

```bash
jupyter notebook Fine_Tuning_DistilBERT_for_Emotion_Recognition.ipynb
```

---

## Key Takeaways

1. **One workflow, three paradigms.** The same fine-tuning discipline applies whether the backbone is a decoder (Llama 2), an encoder (DistilBERT), or an encoder-decoder (BART).
2. **Task shape dictates the head and the metric.** Single-label uses softmax and accuracy/F1; multi-label uses sigmoid with per-label thresholds; summarization uses seq2seq generation.
3. **QLoRA scales down the cost of scaling up.** A 7B chat model is fine-tuned on modest hardware by training only low-rank adapters over a quantized base.
4. **Tracking is part of the pipeline.** W&B and Weave make runs reproducible, comparable, and auditable.

---

*Part of the [Attention-Is-All-You-Code](https://github.com/ArpitKadam/Attention-Is-All-You-Code) project — implementing machine learning concepts from scratch.*
