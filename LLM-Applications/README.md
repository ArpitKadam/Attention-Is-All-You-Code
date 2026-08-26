# LLM Applications

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Diffusers](https://img.shields.io/badge/Diffusers-Stable%20Diffusion-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/docs/diffusers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-ArpitKadam-181717?style=flat&logo=github)](https://github.com/ArpitKadam/Attention-Is-All-You-Code)

> Applied projects that put large language models and generative models to work — attention visualization, retrieval-augmented generation over source code, controllable text decoding, text-to-image diffusion, and a full agentic web application.

---

## Table of Contents

1. [Overview](#overview)
2. [Contents](#contents)
3. [Notebook Details](#notebook-details)
4. [Featured Application — Guidely.ai](#featured-application--guidelyai)
5. [Getting Started](#getting-started)
6. [Key Takeaways](#key-takeaways)

---

## Overview

This module shifts focus from *building* models to *applying* them. It gathers a set of self-contained projects that demonstrate how pretrained language and generative models are used in practice — from interpreting a model's internal attention, to grounding an LLM in external documents, to steering generation, to producing images from text.

Alongside the notebooks, the directory hosts **Guidely.ai**, a complete agentic web application maintained as a Git submodule.

---

## Contents

| Item | Type | Domain |
| :--- | :--- | :--- |
| `bertviz_tutorial.ipynb` | Notebook | Attention interpretability |
| `Source_Code_Analysis.ipynb` | Notebook | Retrieval-Augmented Generation (RAG) |
| `Text_Generation_with_GPT2.ipynb` | Notebook | Decoding strategies |
| `Text_To_Image_Diffusion_Model.ipynb` | Notebook | Generative diffusion |
| `Guidely.ai/` | Submodule | Agentic LangGraph application |

---

## Notebook Details

### `bertviz_tutorial.ipynb` — Attention Visualization
Uses the **BertViz** library to render transformer attention interactively across three complementary views:
- **Head View** — attention patterns for individual heads in a layer
- **Model View** — a bird's-eye grid of all heads across all layers
- **Neuron View** — how individual query/key neurons contribute to attention scores

A hands-on window into *what* a transformer attends to, and *where* in its depth.

### `Source_Code_Analysis.ipynb` — RAG over a Codebase
A **retrieval-augmented generation** pipeline that answers questions about a software repository. The notebook clones a Git repo (`GitPython`), splits source files with language-aware text splitters (`RecursiveCharacterTextSplitter`, `Language`), embeds the chunks with **Ollama** embeddings, and indexes them in a **Chroma** vector store. Queries are answered by a **Groq**-hosted chat model (`ChatGroq`), with conversation memory maintained through LangChain's `RunnableWithMessageHistory`.

### `Text_Generation_with_GPT2.ipynb` — Decoding Strategies
Explores how the choice of **decoding algorithm** shapes generated text with a pretrained GPT-2, contrasting:
- **Beam Search** — maintaining multiple candidate sequences for higher-likelihood output
- **Nucleus (Top-p) Sampling** — sampling from the smallest probability mass exceeding a threshold for diverse, natural generation

### `Text_To_Image_Diffusion_Model.ipynb` — Text-to-Image Generation
Runs a **Stable Diffusion** text-to-image pipeline via Hugging Face `diffusers` (`StableDiffusionPipeline`), generating images from natural-language prompts and visualizing the results.

---

## Featured Application — Guidely.ai

**Guidely.ai** is an AI-powered travel-planning assistant built as an **agentic workflow** with **LangGraph**. It is included here as a Git submodule and maintained in its own repository.

Highlights:
- A LangGraph agent that orchestrates multiple **tools** — place search, weather information, currency conversion, expense calculation, and arithmetic
- A **Flask** web frontend (HTML/CSS/JS templates)
- Structured configuration, logging, and exception handling under a clean `src/` layout
- Extensive architecture diagrams (system, data-flow, sequence, deployment)

> The submodule points to [`ArpitKadam/Guidely.ai`](https://github.com/ArpitKadam/Guidely.ai). Initialize it with `git submodule update --init --recursive`, and consult its own `README.md` and `project_documentation.md` for full details.

---

## Getting Started

### Prerequisites

```bash
pip install torch transformers bertviz diffusers accelerate \
            langchain langchain-community langchain-chroma langchain-groq langchain-ollama \
            gitpython chromadb
```

> `Source_Code_Analysis.ipynb` requires a running **Ollama** instance for embeddings and a **Groq** API key. `Text_To_Image_Diffusion_Model.ipynb` benefits from a GPU.

### Running

```bash
jupyter notebook bertviz_tutorial.ipynb
```

To work with the featured application:

```bash
git submodule update --init --recursive
cd Guidely.ai
```

---

## Key Takeaways

1. **Interpretability is accessible.** Attention visualization turns an opaque model into an inspectable one, revealing head specialization and layer-wise focus.
2. **RAG grounds LLMs in truth.** Retrieval over an embedded corpus lets a general model answer precise, source-specific questions without retraining.
3. **Decoding is a design decision.** Beam search and nucleus sampling produce markedly different text from the *same* weights.
4. **Generative modeling spans modalities.** The same transformer foundations extend from text to images via diffusion.
5. **Agents compose capabilities.** Guidely.ai shows how tool-calling and graph-structured control turn a language model into a functional application.

---

*Part of the [Attention-Is-All-You-Code](https://github.com/ArpitKadam/Attention-Is-All-You-Code) project — implementing machine learning concepts from scratch.*
