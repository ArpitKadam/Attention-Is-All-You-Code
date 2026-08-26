# Utilities

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![nbformat](https://img.shields.io/badge/nbformat-Notebook%20Tools-F37626?style=flat&logo=jupyter&logoColor=white)](https://nbformat.readthedocs.io/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-ArpitKadam-181717?style=flat&logo=github)](https://github.com/ArpitKadam/Attention-Is-All-You-Code)

> Shared reference material and tooling that support the rest of the repository — a Hugging Face quick-reference cheatsheet and a notebook-hygiene utility.

---

## Table of Contents

1. [Overview](#overview)
2. [Contents](#contents)
3. [`cheatsheet.ipynb`](#cheatsheetipynb)
4. [`nb_converter.py`](#nb_converterpy)
5. [Usage](#usage)

---

## Overview

This directory collects cross-cutting resources that are not specific to any single model or experiment: a concise decision-oriented cheatsheet for the Hugging Face ecosystem, and a maintenance script that keeps notebooks rendering cleanly on GitHub.

---

## Contents

| File | Purpose |
| :--- | :--- |
| `cheatsheet.ipynb` | Quick-reference guide to Hugging Face `AutoModel` heads, pipelines, and evaluation metrics |
| `nb_converter.py` | Repairs invalid notebook widget metadata that breaks GitHub rendering |

---

## `cheatsheet.ipynb`

A compact reference covering the practical decisions that recur throughout the repository:

- **`AutoModel` classes with different heads** — mapping a task (classification, token classification, question answering, causal/seq2seq generation) to the correct `AutoModelFor…` class.
- **The built-in `pipeline` method** — the fastest path from a pretrained model to a prediction.
- **BLEU vs. ROUGE** — a comparison of the two dominant text-generation evaluation metrics, and when each applies (translation vs. summarization).
- **A decision tree** — "which `AutoModel` should I pick?" — for choosing the right architecture head for a given task.

---

## `nb_converter.py`

Jupyter notebooks produced by some environments (notably Google Colab) embed a `widgets` block in their metadata that omits the required `state` key. GitHub's notebook renderer rejects such notebooks with an *"Invalid Notebook"* error.

This script detects the malformed `widgets` metadata and removes it, restoring clean rendering **without altering any cell content**.

```python
# Core logic
widgets = nb.metadata.get("widgets")
if isinstance(widgets, dict) and "state" not in widgets:
    nb.metadata.pop("widgets")     # strip the invalid block
    nbformat.write(nb, f)          # rewrite the notebook in place
```

It is wired into the repository's `.pre-commit-config.yaml`, so notebooks are sanitized automatically on every commit.

---

## Usage

### Cheatsheet

```bash
jupyter notebook cheatsheet.ipynb
```

### Notebook Sanitizer

Run manually on one or more notebooks:

```bash
pip install nbformat
python nb_converter.py path/to/notebook.ipynb [more.ipynb ...]
```

Or let it run automatically via pre-commit:

```bash
pip install pre-commit
pre-commit install
```

---

*Part of the [Attention-Is-All-You-Code](https://github.com/ArpitKadam/Attention-Is-All-You-Code) project — implementing machine learning concepts from scratch.*
