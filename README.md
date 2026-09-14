# Complete Local PyTorch Transformer Retrieval-Augmented Generation (RAG) System

A complete, production-grade, local **Retrieval-Augmented Generation (RAG)** system built entirely from scratch using **PyTorch** and **Transformer Architecture** on Windows.

This project runs 100% locally on your computer with **zero external cloud APIs** (no OpenAI, no Anthropic, no Google Cloud). It includes local document ingestion (TXT, PDF, DOCX), subword BPE tokenization, dense Transformer retriever training, vector index search, seq2seq Transformer answer generation, evaluation metrics (Recall@K, MRR, BLEU, Exact Match), independent checkpointing with resume functionality, and an interactive CLI chat interface.

---

## 🛠️ Key Technical Features

1. **Native PyTorch Transformer Components**:
   - Scaled Dot-Product Attention: $\text{softmax}(QK^T / \sqrt{d_k})V$
   - Multi-Head Attention ($H$ heads)
   - Sinusoidal Positional Encoding
   - Transformer Encoder & Decoder blocks with Residual Connections and Layer Normalization.

2. **Dense Retriever**:
   - Transformer Encoder with Mean Pooling / CLS token pooling.
   - InfoNCE / Contrastive Loss for question-document similarity.

3. **Local Vector Store**:
   - Matrix multiplication and Cosine Similarity search over PyTorch / NumPy arrays.
   - Saves `.npy` embeddings and `.json` metadata locally.

4. **Seq2Seq Answer Generator**:
   - Encoder-Decoder Transformer model trained with Teacher Forcing and Cross-Entropy Loss.
   - Greedy, Beam, and Top-K/Top-P Nucleus sampling decoding algorithms.

5. **Interruption Resilience & Checkpointing**:
   - Independent checkpoint saving (`latest.pt`, `best_model.pt`) for Retriever and Generator.
   - Resume training seamlessly after system restarts (`--resume`).

6. **Environment Awareness & Presets**:
   - Auto-detects OS, PyTorch version, CPU cores, RAM, and CUDA GPU status (`check_environment.py`).
   - Adapts model size using `SMALL`, `MEDIUM`, or `LARGE` presets.

---

## 🚀 Quickstart Step-by-Step Workflow

### Step 1: Activate Environment & Check Hardware

**PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

**Check Environment & Hardware Preset:**
```bash
python check_environment.py
```

---

### Step 2: Prepare Data & Ingest Local Documents

```bash
python prepare_data.py
python ingest_documents.py
```
This initializes sample documents in `data/documents/` and ingests/chunks them into `data/processed/chunks.json`.

---

### Step 3: Train Subword Tokenizer

```bash
python -m src.tokenizer.train_tokenizer --vocab_size 8000
```
Trains a local subword BPE tokenizer on your corpus and saves artifacts to `models/tokenizer/`.

---

### Step 4: Retriever Debug & Overfitting Test

Before long training, verify tensor shapes and loss convergence:

**Debug Test:**
```bash
python train_retriever.py --debug
```

**Overfitting Verification:**
```bash
python train_retriever.py --overfit-test
```

---

### Step 5: Train Dense Retriever (with Resume Support)

**Full Training:**
```bash
python train_retriever.py --preset SMALL
```

**Resume Interrupted Training:**
```bash
python train_retriever.py --resume
```

---

### Step 6: Build Vector Database Index

```bash
python build_index.py --rebuild
```
Generates dense embeddings for all chunked documents and saves `indexes/document_embeddings.npy`.

---

### Step 7: Generator Debug & Overfitting Test

**Debug Test:**
```bash
python train_generator.py --debug
```

**Overfitting Verification:**
```bash
python train_generator.py --overfit-test
```

---

### Step 8: Train Seq2Seq Answer Generator

**Full Training:**
```bash
python train_generator.py --preset SMALL
```

**Resume Interrupted Training:**
```bash
python train_generator.py --resume
```

---

### Step 9: Run Evaluation & Plot Metrics

**Run Evaluation:**
```bash
python evaluate.py
```
Computes Recall@1, Recall@3, MRR, Generator Exact Match, and BLEU scores.

**Plot Training Curves:**
```bash
python plot_metrics.py
```
Saves loss curves to `logs/training_loss_plot.png`.

---

### Step 10: Interactive Local RAG Chat

Start asking questions about your local documents:

```bash
python chat.py
```

**With Context Display:**
```bash
python chat.py --show-context
```

Example Session:
```text
==================================================
      LOCAL TRANSFORMER RAG SYSTEM CHAT          
==================================================

Question: What is Artificial Intelligence?

Retrieving relevant documents...

Top Retrieved Sources:
  1. ai_notes.txt -- Page 1 (Similarity Score: 0.9421)
  2. machine_learning_overview.txt -- Page 1 (Similarity Score: 0.8105)

Generating answer...

Answer:
  Artificial Intelligence refers to computer systems designed to simulate human intelligence processes.

Sources:
  [1] ai_notes.txt -- Page 1
  [2] machine_learning_overview.txt -- Page 1
```

---

## 📁 Repository Layout

```text
RAG/
├── README.md
├── requirements.txt
├── config.py
├── check_environment.py
├── prepare_data.py
├── ingest_documents.py
├── train_retriever.py
├── train_generator.py
├── build_index.py
├── evaluate.py
├── plot_metrics.py
├── chat.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── training/
│   └── documents/
│
├── models/
│   ├── tokenizer/
│   ├── retriever/
│   └── generator/
│
├── checkpoints/
│   ├── retriever/
│   └── generator/
│
├── indexes/
├── logs/
│
└── src/
    ├── data/
    │   ├── document_loader.py
    │   ├── document_cleaner.py
    │   ├── chunker.py
    │   └── dataset.py
    ├── tokenizer/
    │   ├── tokenizer.py
    │   └── train_tokenizer.py
    ├── transformer/
    │   ├── attention.py
    │   ├── positional_encoding.py
    │   ├── encoder.py
    │   ├── decoder.py
    │   └── transformer.py
    ├── retriever/
    │   ├── embedding_model.py
    │   ├── retriever.py
    │   ├── train_retriever.py
    │   └── evaluate_retriever.py
    ├── vectorstore/
    │   ├── vector_index.py
    │   └── similarity.py
    ├── generator/
    │   ├── generator_model.py
    │   ├── train_generator.py
    │   └── decoding.py
    ├── rag/
    │   ├── context_builder.py
    │   ├── rag_pipeline.py
    │   └── evaluator.py
    └── utils/
        ├── checkpoint.py
        ├── metrics.py
        └── seed.py
```
