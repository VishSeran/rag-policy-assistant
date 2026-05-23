# Enhance LLMs using RAG and Hugging Face

> Build an intelligent question-answering system that combines Dense Passage Retrieval (DPR) with GPT-2 to ground language model outputs in real documents — no fine-tuning required.

---

## Overview

This project demonstrates **Retrieval-Augmented Generation (RAG)**, a technique that improves large language model (LLM) responses by first retrieving relevant passages from a document corpus and then conditioning text generation on those passages. The motivating use case is an HR policy assistant: employees ask natural-language questions such as *"What is our vacation policy?"* and receive accurate, grounded answers drawn directly from company policy documents.

By the end of this lab you will have built a complete NLP pipeline — from raw text ingestion all the way to context-aware answer generation — using only open-source tools and pretrained models.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebook](#running-the-notebook)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [1. Data Loading and Preprocessing](#1-data-loading-and-preprocessing)
  - [2. Context Encoding with DPR](#2-context-encoding-with-dpr)
  - [3. FAISS Indexing](#3-faiss-indexing)
  - [4. Question Encoding and Retrieval](#4-question-encoding-and-retrieval)
  - [5. Answer Generation with GPT-2](#5-answer-generation-with-gpt-2)
- [Key Results and Observations](#key-results-and-observations)


---

## How It Works

RAG separates the knowledge problem from the generation problem:

```
User Question
     │
     ▼
DPR Question Encoder  ──► Question Embedding
                                   │
                                   ▼
                          FAISS Index Search
                          (over DPR Context Embeddings)
                                   │
                                   ▼
                         Top-k Relevant Passages
                                   │
                    ┌──────────────┘
                    ▼
         Question + Passages → GPT-2 Generator → Final Answer
```

Rather than asking GPT-2 to answer from memorised training data alone, we first fetch the passages most similar to the query and prepend them to the prompt. This dramatically improves factual accuracy for domain-specific documents.

---

## Tech Stack

| Library | Role |
|---|---|
| `transformers` (Hugging Face) | DPR encoders, GPT-2 model and tokenizer |
| `faiss-cpu` | Fast similarity search over dense embeddings |
| `torch` (PyTorch) | Model inference and tensor operations |
| `numpy` | Numerical array manipulation |
| `scikit-learn` | t-SNE dimensionality reduction for visualisation |
| `matplotlib` | 3-D embedding visualisation |
| `wget` | Downloading the policy text corpus |

Pretrained models used:
- **`facebook/dpr-ctx_encoder-single-nq-base`** — encodes document passages
- **`facebook/dpr-question_encoder-single-nq-base`** — encodes user questions
- **`openai-community/gpt2`** — generates natural-language answers

---

## Project Structure

```
.
├── RAG-v1.ipynb            # Main notebook — full end-to-end pipeline
└── companyPolicies.txt     # Downloaded at runtime; HR policy corpus
```

---

## Getting Started

### Prerequisites

- Python 3.8 or later
- A CUDA-capable GPU is optional but recommended for faster DPR encoding; the notebook runs on CPU with `torch` CPU wheels

### Installation

Install all dependencies in one step:

```bash
pip install wget transformers datasets faiss-cpu matplotlib scikit-learn
pip install torch==2.8.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
```

> **Note:** After installation, restart your Jupyter kernel before executing any notebook cells.

### Running the Notebook

```bash
jupyter notebook RAG-v1.ipynb
```

Execute the cells in order from top to bottom. The notebook will automatically download the `companyPolicies.txt` corpus on first run.

---

## Pipeline Walkthrough

### 1. Data Loading and Preprocessing

The `companyPolicies.txt` file is fetched from IBM Cloud Object Storage and split into individual paragraphs. Each paragraph becomes one retrievable unit:

```python
def read_and_split_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    paragraphs = text.split('\n')
    paragraphs = [para.strip() for para in paragraphs if len(para.strip()) > 0]
    return paragraphs
```

You can substitute any plain-text file relevant to your domain.

### 2. Context Encoding with DPR

Each paragraph is tokenised by `DPRContextEncoderTokenizer` (a BERT-based tokeniser with segment embeddings and attention masks) and then encoded by `DPRContextEncoder` into a 768-dimensional dense vector:

```python
def encode_contexts(text_list):
    embeddings = []
    for text in text_list:
        inputs = context_tokenizer(text, return_tensors='pt',
                                   padding=True, truncation=True, max_length=256)
        outputs = context_encoder(**inputs)
        embeddings.append(outputs.pooler_output)
    return torch.cat(embeddings).detach().numpy()
```

A 3-D t-SNE visualisation is included to confirm that semantically similar paragraphs (e.g. two passages both discussing workplace diversity) cluster together in embedding space.

### 3. FAISS Indexing

All context embeddings are loaded into a `faiss.IndexFlatL2` index — a flat, exact-nearest-neighbour index using Euclidean distance. This enables sub-millisecond lookup at query time:

```python
import faiss

embedding_dim = 768
index = faiss.IndexFlatL2(embedding_dim)
index.add(context_embeddings_np)
```

### 4. Question Encoding and Retrieval

At query time, the user's question is encoded by the separate `DPRQuestionEncoder` (same BERT architecture, but trained with different contrastive objectives) and the top-k most similar paragraphs are fetched from the FAISS index:

```python
def search_relevant_contexts(question, question_tokenizer, question_encoder, index, k=5):
    question_inputs = question_tokenizer(question, return_tensors='pt')
    question_embedding = question_encoder(**question_inputs).pooler_output.detach().numpy()
    D, I = index.search(question_embedding, k)
    return D, I
```

The returned `D` (distances) and `I` (paragraph indices) give both a relevance score and the actual text to be passed to the generator.

### 5. Answer Generation with GPT-2

Two generation modes are compared side-by-side:

**Without RAG context** — GPT-2 answers from its pretrained weights alone:

```python
def generate_answer_without_context(question):
    inputs = tokenizer(question, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=150, num_beams=4,
                                 early_stopping=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

**With RAG context** — retrieved passages are prepended to the prompt:

```python
def generate_answer(question, contexts):
    input_text = question + ' ' + ' '.join(contexts)
    inputs = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_new_tokens=50, num_beams=4,
                                 early_stopping=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

---

## Key Results and Observations

- **Direct GPT-2 generation** produces generic, often unfocused answers because the model has no access to the company-specific document corpus.
- **RAG-augmented generation** produces answers that are visibly more specific and factually grounded, because the most relevant policy paragraphs are supplied directly in the prompt context.
- t-SNE visualisation of DPR embeddings confirms that the model's geometric similarity corresponds to genuine topical similarity between paragraphs.

---



