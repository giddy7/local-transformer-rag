import os
import json
import time
import threading
import traceback
from typing import Dict, Any, List, Optional
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from config import (
    DOCUMENTS_DIR, PROCESSED_DATA_DIR, TRAINING_DATA_DIR,
    RETRIEVER_CHECKPOINT_DIR, GENERATOR_CHECKPOINT_DIR,
    TOKENIZER_DIR, INDEXES_DIR, PRESETS
)
from src.rag.rag_pipeline import RAGPipeline
from src.tokenizer.train_tokenizer import train_tokenizer
from src.retriever.train_retriever import run_retriever_training
from src.generator.train_generator import run_generator_training
from build_index import build_index
from ingest_documents import ingest_documents
from prepare_data import prepare_sample_data
from evaluate import run_evaluation

app = FastAPI(title="Local Transformer RAG Studio")

# Global pipeline instance and lock
pipeline_lock = threading.Lock()
current_pipeline: Optional[RAGPipeline] = None
current_preset = "SMALL"

# Training state
training_state = {
    "is_training": False,
    "current_task": "Idle",
    "progress": 0,
    "logs": []
}

def log_message(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    training_state["logs"].append(f"[{timestamp}] {msg}")
    if len(training_state["logs"]) > 200:
        training_state["logs"].pop(0)

def get_or_create_pipeline(preset: str = "SMALL", force_reload: bool = False) -> RAGPipeline:
    global current_pipeline, current_preset
    with pipeline_lock:
        if current_pipeline is None or current_preset != preset or force_reload:
            current_preset = preset
            current_pipeline = RAGPipeline(preset=preset)
        return current_pipeline

class QueryRequest(BaseModel):
    question: str
    preset: str = "SMALL"
    top_k: int = 3
    show_context: bool = True
    decoding: str = "greedy"

class IngestTextRequest(BaseModel):
    filename: str
    content: str

class TrainRequest(BaseModel):
    preset: str = "SMALL"
    epochs_retriever: int = 20
    epochs_generator: int = 20
    train_tokenizer_flag: bool = True
    train_retriever_flag: bool = True
    train_generator_flag: bool = True

@app.get("/api/status")
def get_status():
    tokenizer_ready = os.path.exists(os.path.join(TOKENIZER_DIR, "tokenizer.json"))
    retriever_ready = os.path.exists(os.path.join(RETRIEVER_CHECKPOINT_DIR, "best_model.pt"))
    generator_ready = os.path.exists(os.path.join(GENERATOR_CHECKPOINT_DIR, "best_model.pt"))
    index_ready = os.path.exists(os.path.join(INDEXES_DIR, "document_embeddings.npy"))

    chunks_count = 0
    chunks_file = os.path.join(PROCESSED_DATA_DIR, "chunks.json")
    if os.path.exists(chunks_file):
        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                chunks_count = len(chunks)
        except Exception:
            pass

    return {
        "tokenizer_ready": tokenizer_ready,
        "retriever_ready": retriever_ready,
        "generator_ready": generator_ready,
        "index_ready": index_ready,
        "chunks_count": chunks_count,
        "training": training_state,
        "preset": current_preset
    }

@app.post("/api/query")
def handle_query(req: QueryRequest):
    try:
        pipeline = get_or_create_pipeline(preset=req.preset)
        response = pipeline.answer_question(
            question=req.question,
            top_k=req.top_k,
            show_context=req.show_context,
            decoding=req.decoding
        )
        return {"success": True, "data": response}
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

@app.get("/api/documents")
def list_documents():
    docs = []
    if os.path.exists(DOCUMENTS_DIR):
        for fname in os.listdir(DOCUMENTS_DIR):
            fpath = os.path.join(DOCUMENTS_DIR, fname)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                docs.append({
                    "filename": fname,
                    "size_bytes": size,
                    "modified": time.ctime(os.path.getmtime(fpath))
                })
    return {"documents": docs}

@app.post("/api/documents/add")
def add_document(req: IngestTextRequest):
    try:
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        safe_name = "".join(c for c in req.filename if c.isalnum() or c in "._- ")
        if not safe_name.endswith(".txt"):
            safe_name += ".txt"
        file_path = os.path.join(DOCUMENTS_DIR, safe_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(req.content)
        
        # Auto-reingest & update index
        chunks = ingest_documents()
        build_index(rebuild=True, preset=current_preset)
        get_or_create_pipeline(preset=current_preset, force_reload=True)
        return {"success": True, "message": f"Saved {safe_name} and rebuilt index with {len(chunks)} chunks."}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/index/rebuild")
def rebuild_vector_index():
    try:
        ingest_documents()
        build_index(rebuild=True, preset=current_preset)
        get_or_create_pipeline(preset=current_preset, force_reload=True)
        return {"success": True, "message": "Vector index successfully rebuilt."}
    except Exception as e:
        return {"success": False, "error": str(e)}

def _train_worker(req: TrainRequest):
    global training_state
    training_state["is_training"] = True
    training_state["progress"] = 5
    training_state["logs"] = []
    log_message("Starting training pipeline...")

    try:
        # 1. Prepare data
        log_message("Preparing training datasets and sample documents...")
        prepare_sample_data()
        ingest_documents()
        training_state["progress"] = 20

        # 2. Tokenizer
        if req.train_tokenizer_flag:
            log_message("Training Subword BPE Tokenizer...")
            training_state["current_task"] = "Training Tokenizer"
            train_tokenizer(vocab_size=8000)
            training_state["progress"] = 35
            log_message("Tokenizer training finished.")

        # 3. Retriever
        if req.train_retriever_flag:
            log_message(f"Training Dense Retriever with preset {req.preset}...")
            training_state["current_task"] = "Training Retriever"
            run_retriever_training(preset=req.preset)
            training_state["progress"] = 65
            log_message("Retriever training finished.")

        # 4. Build Index
        log_message("Building document vector index...")
        training_state["current_task"] = "Building Vector Index"
        build_index(rebuild=True, preset=req.preset)
        training_state["progress"] = 75
        log_message("Vector index built successfully.")

        # 5. Generator
        if req.train_generator_flag:
            log_message(f"Training Seq2Seq Transformer Generator with preset {req.preset}...")
            training_state["current_task"] = "Training Generator"
            run_generator_training(preset=req.preset)
            training_state["progress"] = 95
            log_message("Generator training finished.")

        # Reload pipeline
        log_message("Reloading active RAG Pipeline...")
        get_or_create_pipeline(preset=req.preset, force_reload=True)
        training_state["progress"] = 100
        training_state["current_task"] = "Completed"
        log_message("All training tasks completed successfully!")

    except Exception as e:
        log_message(f"ERROR: {str(e)}")
        log_message(traceback.format_exc())
        training_state["current_task"] = f"Failed: {str(e)}"
    finally:
        training_state["is_training"] = False

@app.post("/api/train")
def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    if training_state["is_training"]:
        return {"success": False, "message": "Training is already in progress."}
    
    background_tasks.add_task(_train_worker, req)
    return {"success": True, "message": "Training job launched in background."}

@app.get("/api/evaluate")
def evaluate_models():
    try:
        results = run_evaluation(preset=current_preset)
        return {"success": True, "metrics": results}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTML_CONTENT

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local Transformer RAG Studio</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #07090e;
            --bg-card: rgba(18, 22, 34, 0.75);
            --bg-card-hover: rgba(26, 32, 50, 0.85);
            --bg-input: rgba(10, 14, 24, 0.8);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-accent: rgba(99, 102, 241, 0.35);
            --accent-primary: #6366f1;
            --accent-secondary: #a855f7;
            --accent-cyan: #06b6d4;
            --accent-emerald: #10b981;
            --accent-rose: #f43f5e;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --shadow-glow: 0 0 25px rgba(99, 102, 241, 0.25);
            --radius-lg: 16px;
            --radius-md: 10px;
            --radius-sm: 6px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(168, 85, 247, 0.10) 0px, transparent 50%),
                radial-gradient(at 50% 100%, rgba(6, 182, 212, 0.08) 0px, transparent 50%);
            background-attachment: fixed;
        }

        header {
            border-bottom: 1px solid var(--border-subtle);
            background: rgba(7, 9, 14, 0.85);
            backdrop-filter: blur(16px);
            position: sticky;
            top: 0;
            z-index: 100;
            padding: 14px 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
        }

        .brand-icon svg {
            width: 20px;
            height: 20px;
            color: #fff;
        }

        .brand-title {
            font-size: 1.15rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 40%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }

        .brand-subtitle {
            font-size: 0.72rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(16, 185, 129, 0.1);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.25);
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 8px #10b981;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.85); }
        }

        .main-layout {
            display: grid;
            grid-template-columns: 320px 1fr 340px;
            gap: 20px;
            padding: 20px 28px;
            flex: 1;
            max-width: 1700px;
            width: 100%;
            margin: 0 auto;
        }

        .glass-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            backdrop-filter: blur(12px);
            padding: 18px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            transition: border-color 0.2s ease;
        }

        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.12);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .card-title {
            font-size: 0.92rem;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .card-title svg {
            width: 16px;
            height: 16px;
            color: var(--accent-primary);
        }

        /* Controls / Form */
        label {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 6px;
            display: block;
        }

        .form-group {
            margin-bottom: 14px;
        }

        select, input[type="text"], input[type="number"], textarea {
            width: 100%;
            background: var(--bg-input);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            color: var(--text-primary);
            padding: 9px 12px;
            font-size: 0.85rem;
            font-family: inherit;
            transition: all 0.2s ease;
            outline: none;
        }

        select:focus, input:focus, textarea:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
        }

        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 9px 16px;
            border-radius: var(--radius-sm);
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            border: none;
            transition: all 0.2s ease;
            font-family: inherit;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            color: #ffffff;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
        }

        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.06);
            color: var(--text-primary);
            border: 1px solid var(--border-subtle);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.15);
        }

        .btn-sm {
            padding: 6px 10px;
            font-size: 0.75rem;
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }

        /* Chat Section */
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 100%;
            overflow: hidden;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
            padding-right: 6px;
            margin-bottom: 14px;
            max-height: calc(100vh - 240px);
        }

        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }
        .chat-messages::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }

        .message-row {
            display: flex;
            gap: 12px;
            width: 100%;
        }

        .message-row.user {
            flex-direction: row-reverse;
        }

        .avatar {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: 700;
            flex-shrink: 0;
        }

        .avatar.bot {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-cyan));
            color: #fff;
        }

        .avatar.user {
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: #fff;
        }

        .message-bubble {
            max-width: 82%;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 12px 16px;
            font-size: 0.88rem;
            line-height: 1.55;
        }

        .message-row.user .message-bubble {
            background: rgba(99, 102, 241, 0.15);
            border-color: rgba(99, 102, 241, 0.3);
            color: #ffffff;
        }

        .sources-block {
            margin-top: 12px;
            padding-top: 10px;
            border-top: 1px dashed rgba(255, 255, 255, 0.1);
            font-size: 0.78rem;
        }

        .source-item {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 6px;
            padding: 8px 10px;
            margin-top: 6px;
        }

        .source-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: var(--accent-cyan);
            font-weight: 600;
            margin-bottom: 4px;
        }

        .score-bar-bg {
            height: 4px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 2px;
            overflow: hidden;
            margin: 4px 0 6px 0;
        }

        .score-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-primary));
        }

        .source-snippet {
            color: var(--text-secondary);
            font-size: 0.75rem;
            font-style: italic;
        }

        .chips-container {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 12px;
        }

        .chip {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 5px 12px;
            font-size: 0.74rem;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .chip:hover {
            background: rgba(99, 102, 241, 0.15);
            border-color: var(--accent-primary);
            color: #ffffff;
            transform: translateY(-1px);
        }

        .chat-input-row {
            display: flex;
            gap: 8px;
            position: relative;
        }

        .chat-input-row input {
            padding-right: 42px;
        }

        .send-btn {
            position: absolute;
            right: 6px;
            top: 50%;
            transform: translateY(-50%);
            width: 32px;
            height: 32px;
            border-radius: 6px;
            background: var(--accent-primary);
            border: none;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: background 0.2s ease;
        }

        .send-btn:hover {
            background: var(--accent-secondary);
        }

        /* Sidebar Tabs & Lists */
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-bottom: 14px;
        }

        .stat-box {
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            padding: 8px 10px;
        }

        .stat-label {
            font-size: 0.68rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
        }

        .stat-val {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-top: 2px;
        }

        .terminal-box {
            background: #040508;
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: var(--radius-sm);
            padding: 10px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: #38bdf8;
            height: 180px;
            overflow-y: auto;
            white-space: pre-wrap;
            line-height: 1.4;
            margin-top: 10px;
        }

        .progress-bar-container {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 8px;
        }

        .progress-bar-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, var(--accent-primary), var(--accent-cyan));
            transition: width 0.3s ease;
        }

        .doc-list {
            max-height: 150px;
            overflow-y: auto;
            margin-bottom: 12px;
        }

        .doc-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 8px;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.03);
            margin-bottom: 4px;
            font-size: 0.75rem;
        }

        .badge-pill {
            padding: 2px 7px;
            border-radius: 10px;
            font-size: 0.68rem;
            font-weight: 600;
            background: rgba(99, 102, 241, 0.15);
            color: #818cf8;
        }

        @media (max-width: 1200px) {
            .main-layout {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="brand-icon">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                </svg>
            </div>
            <div>
                <div class="brand-title">PyTorch Transformer RAG Studio</div>
                <div class="brand-subtitle">100% Local Offline Neural Search & Generation</div>
            </div>
        </div>
        <div class="header-actions">
            <div class="status-badge" id="app-status">
                <span class="status-dot"></span>
                <span>System Ready</span>
            </div>
        </div>
    </header>

    <main class="main-layout">
        <!-- Left Column: Pipeline Config & Training -->
        <div style="display: flex; flex-direction: column; gap: 16px;">
            <div class="glass-card">
                <div class="card-header">
                    <div class="card-title">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"/></svg>
                        Model Settings
                    </div>
                </div>

                <div class="form-group">
                    <label>Architecture Preset</label>
                    <select id="preset-select">
                        <option value="SMALL" selected>SMALL (d_model=128, heads=4, layers=3)</option>
                        <option value="MEDIUM">MEDIUM (d_model=256, heads=8, layers=4)</option>
                        <option value="LARGE">LARGE (d_model=512, heads=8, layers=6)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Retriever Top-K Passages (<span id="topk-val">3</span>)</label>
                    <input type="range" id="topk-range" min="1" max="5" value="3" style="width: 100%; accent-color: var(--accent-primary);">
                </div>

                <div class="form-group">
                    <label>Generation Decoding</label>
                    <select id="decoding-select">
                        <option value="greedy" selected>Greedy Decoding (Deterministic)</option>
                        <option value="sample">Nucleus Sampling (Creative)</option>
                    </select>
                </div>
            </div>

            <!-- Training Card -->
            <div class="glass-card">
                <div class="card-header">
                    <div class="card-title">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"/></svg>
                        Train Models
                    </div>
                    <span class="badge-pill" id="training-badge">Idle</span>
                </div>

                <p style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 12px;">
                    Trigger end-to-end retraining for Tokenizer, Dense Retriever, and Seq2Seq Generator.
                </p>

                <div style="display: flex; gap: 8px;">
                    <button class="btn btn-primary" id="btn-train" style="flex: 1;" onclick="triggerTraining()">
                        <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                        Start Training
                    </button>
                    <button class="btn btn-secondary btn-sm" onclick="runEval()">
                        Evaluate
                    </button>
                </div>

                <div class="progress-bar-container">
                    <div class="progress-bar-fill" id="train-progress"></div>
                </div>

                <div class="terminal-box" id="train-terminal">> Ready to train or evaluate.</div>
            </div>
        </div>

        <!-- Center Column: RAG Chat Interface -->
        <div class="glass-card chat-container">
            <div class="card-header">
                <div class="card-title">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/></svg>
                    Interactive RAG Chat
                </div>
                <button class="btn btn-secondary btn-sm" onclick="clearChat()">Clear Chat</button>
            </div>

            <div class="chips-container">
                <span class="chip" onclick="askPreset('What is Artificial Intelligence?')">🤖 What is Artificial Intelligence?</span>
                <span class="chip" onclick="askPreset('What is Machine Learning?')">🧠 What is Machine Learning?</span>
                <span class="chip" onclick="askPreset('Who introduced Transformer architecture?')">⚡ Who introduced Transformers?</span>
                <span class="chip" onclick="askPreset('What are the core components of RAG?')">📚 What is RAG?</span>
            </div>

            <div class="chat-messages" id="chat-messages">
                <div class="message-row">
                    <div class="avatar bot">RAG</div>
                    <div class="message-bubble">
                        Hello! I am your local Transformer Retrieval-Augmented Generation assistant. Ask any question grounded in your ingested documents.
                    </div>
                </div>
            </div>

            <div class="chat-input-row">
                <input type="text" id="query-input" placeholder="Type your question here (e.g. What is Artificial Intelligence?)..." onkeydown="if(event.key==='Enter') sendQuery()">
                <button class="send-btn" onclick="sendQuery()">
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14M12 5l7 7-7 7"/></svg>
                </button>
            </div>
        </div>

        <!-- Right Column: Knowledge Base & Ingestion -->
        <div style="display: flex; flex-direction: column; gap: 16px;">
            <div class="glass-card">
                <div class="card-header">
                    <div class="card-title">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
                        Knowledge Documents
                    </div>
                    <button class="btn btn-secondary btn-sm" onclick="loadDocs()">Refresh</button>
                </div>

                <div class="doc-list" id="doc-list">
                    <div style="font-size: 0.75rem; color: var(--text-muted);">Loading documents...</div>
                </div>

                <button class="btn btn-secondary btn-sm" style="width: 100%; margin-top: 4px;" onclick="rebuildIndex()">
                    🔄 Rebuild Vector Index
                </button>
            </div>

            <!-- Ingest New Knowledge -->
            <div class="glass-card">
                <div class="card-header">
                    <div class="card-title">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/></svg>
                        Add Knowledge
                    </div>
                </div>

                <div class="form-group">
                    <label>Document Title</label>
                    <input type="text" id="new-doc-name" placeholder="e.g. quantum_computing.txt">
                </div>

                <div class="form-group">
                    <label>Document Content</label>
                    <textarea id="new-doc-content" rows="4" placeholder="Paste your raw text notes here..."></textarea>
                </div>

                <button class="btn btn-primary btn-sm" style="width: 100%;" onclick="ingestNewDoc()">
                    Save & Re-Index
                </button>
            </div>

            <!-- Status Card -->
            <div class="glass-card">
                <div class="card-header">
                    <div class="card-title">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
                        System Check
                    </div>
                </div>

                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-label">Retriever</div>
                        <div class="stat-val" id="stat-retriever" style="color: #34d399;">Ready</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Generator</div>
                        <div class="stat-val" id="stat-generator" style="color: #34d399;">Ready</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Indexed Chunks</div>
                        <div class="stat-val" id="stat-chunks">4</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Device</div>
                        <div class="stat-val" style="color: #818cf8;">CPU</div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        const topkRange = document.getElementById('topk-range');
        const topkVal = document.getElementById('topk-val');
        topkRange.addEventListener('input', () => topkVal.innerText = topkRange.value);

        async function updateStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();

                document.getElementById('stat-retriever').innerText = data.retriever_ready ? 'Ready' : 'Not Found';
                document.getElementById('stat-retriever').style.color = data.retriever_ready ? '#34d399' : '#f43f5e';

                document.getElementById('stat-generator').innerText = data.generator_ready ? 'Ready' : 'Not Found';
                document.getElementById('stat-generator').style.color = data.generator_ready ? '#34d399' : '#f43f5e';

                document.getElementById('stat-chunks').innerText = data.chunks_count;

                const trainBadge = document.getElementById('training-badge');
                const btnTrain = document.getElementById('btn-train');
                const progressFill = document.getElementById('train-progress');
                const terminal = document.getElementById('train-terminal');

                if (data.training.is_training) {
                    trainBadge.innerText = data.training.current_task;
                    trainBadge.style.background = 'rgba(245, 158, 11, 0.2)';
                    trainBadge.style.color = '#fbbf24';
                    btnTrain.disabled = true;
                    progressFill.style.width = data.training.progress + '%';
                } else {
                    trainBadge.innerText = data.training.current_task || 'Idle';
                    trainBadge.style.background = 'rgba(99, 102, 241, 0.15)';
                    trainBadge.style.color = '#818cf8';
                    btnTrain.disabled = false;
                    progressFill.style.width = data.training.progress + '%';
                }

                if (data.training.logs && data.training.logs.length > 0) {
                    terminal.innerText = data.training.logs.join('\\n');
                    terminal.scrollTop = terminal.scrollHeight;
                }
            } catch (e) {
                console.error(e);
            }
        }

        setInterval(updateStatus, 1500);
        updateStatus();

        async function loadDocs() {
            try {
                const res = await fetch('/api/documents');
                const data = await res.json();
                const container = document.getElementById('doc-list');
                container.innerHTML = '';
                if (!data.documents || data.documents.length === 0) {
                    container.innerHTML = '<div style="color: var(--text-muted); font-size: 0.75rem;">No documents yet.</div>';
                    return;
                }
                data.documents.forEach(d => {
                    const item = document.createElement('div');
                    item.className = 'doc-item';
                    item.innerHTML = `
                        <span>📄 <b>${d.filename}</b></span>
                        <span style="color: var(--text-muted);">${(d.size_bytes / 1024).toFixed(1)} KB</span>
                    `;
                    container.appendChild(item);
                });
            } catch (e) {
                console.error(e);
            }
        }
        loadDocs();

        function askPreset(q) {
            document.getElementById('query-input').value = q;
            sendQuery();
        }

        function clearChat() {
            document.getElementById('chat-messages').innerHTML = `
                <div class="message-row">
                    <div class="avatar bot">RAG</div>
                    <div class="message-bubble">
                        Chat reset. Ask a question to begin!
                    </div>
                </div>
            `;
        }

        async function sendQuery() {
            const input = document.getElementById('query-input');
            const question = input.value.trim();
            if (!question) return;

            input.value = '';

            const chatMessages = document.getElementById('chat-messages');

            // Append user message
            const userMsg = document.createElement('div');
            userMsg.className = 'message-row user';
            userMsg.innerHTML = `
                <div class="avatar user">You</div>
                <div class="message-bubble">${escapeHtml(question)}</div>
            `;
            chatMessages.appendChild(userMsg);

            // Append loading bot message
            const botMsg = document.createElement('div');
            botMsg.className = 'message-row';
            const msgId = 'bot-' + Date.now();
            botMsg.innerHTML = `
                <div class="avatar bot">RAG</div>
                <div class="message-bubble" id="${msgId}">
                    <span style="color: var(--text-muted); font-style: italic;">Retrieving passages & generating grounded response...</span>
                </div>
            `;
            chatMessages.appendChild(botMsg);
            chatMessages.scrollTop = chatMessages.scrollHeight;

            const preset = document.getElementById('preset-select').value;
            const topK = parseInt(document.getElementById('topk-range').value);
            const decoding = document.getElementById('decoding-select').value;

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        question: question,
                        preset: preset,
                        top_k: topK,
                        show_context: true,
                        decoding: decoding
                    })
                });
                const result = await res.json();
                const target = document.getElementById(msgId);

                if (result.success) {
                    const data = result.data;
                    let sourcesHtml = '';
                    if (data.sources && data.sources.length > 0) {
                        sourcesHtml = `
                            <div class="sources-block">
                                <div style="font-weight: 700; margin-bottom: 4px; color: var(--text-secondary);">Top Retrieved Context Passages:</div>
                                ${data.sources.map(s => `
                                    <div class="source-item">
                                        <div class="source-header">
                                            <span>#${s.rank} ${escapeHtml(s.document_name)} (Page ${s.page})</span>
                                            <span>Sim: ${(s.score * 100).toFixed(1)}%</span>
                                        </div>
                                        <div class="score-bar-bg">
                                            <div class="score-bar-fill" style="width: ${Math.max(0, Math.min(100, s.score * 100))}%;"></div>
                                        </div>
                                        <div class="source-snippet">"${escapeHtml(s.text)}"</div>
                                    </div>
                                `).join('')}
                            </div>
                        `;
                    }

                    target.innerHTML = `
                        <div style="font-weight: 500;">${escapeHtml(data.answer)}</div>
                        ${sourcesHtml}
                    `;
                } else {
                    target.innerHTML = `<span style="color: var(--accent-rose);">Error: ${escapeHtml(result.error)}</span>`;
                }
            } catch (e) {
                document.getElementById(msgId).innerHTML = `<span style="color: var(--accent-rose);">Network error: ${escapeHtml(e.message)}</span>`;
            }

            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        async function triggerTraining() {
            const preset = document.getElementById('preset-select').value;
            const res = await fetch('/api/train', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    preset: preset,
                    train_tokenizer_flag: true,
                    train_retriever_flag: true,
                    train_generator_flag: true
                })
            });
            const data = await res.json();
            alert(data.message);
        }

        async function runEval() {
            const terminal = document.getElementById('train-terminal');
            terminal.innerText += '\\n[*] Running full evaluation...';
            const res = await fetch('/api/evaluate');
            const data = await res.json();
            if (data.success) {
                terminal.innerText += '\\n[OK] Evaluation Results:\\n' + JSON.stringify(data.metrics, null, 2);
            } else {
                terminal.innerText += '\\n[!] Evaluation failed: ' + data.error;
            }
            terminal.scrollTop = terminal.scrollHeight;
        }

        async function ingestNewDoc() {
            const name = document.getElementById('new-doc-name').value.trim();
            const content = document.getElementById('new-doc-content').value.trim();
            if (!name || !content) {
                alert('Please provide both document title and content.');
                return;
            }
            const res = await fetch('/api/documents/add', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename: name, content: content})
            });
            const data = await res.json();
            if (data.success) {
                alert(data.message);
                document.getElementById('new-doc-name').value = '';
                document.getElementById('new-doc-content').value = '';
                loadDocs();
                updateStatus();
            } else {
                alert('Error: ' + data.error);
            }
        }

        async function rebuildIndex() {
            const res = await fetch('/api/index/rebuild', {method: 'POST'});
            const data = await res.json();
            alert(data.message);
            updateStatus();
        }

        function escapeHtml(text) {
            if (!text) return '';
            return text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }
    </script>
</body>
</html>
"""

def start_server(host="127.0.0.1", port=4881):
    print(f"[*] Starting Local Transformer RAG Server on http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    start_server()
