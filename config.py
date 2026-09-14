import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Any

# Root Project Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Standard Directory Layout
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
TRAINING_DATA_DIR = os.path.join(DATA_DIR, "training")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")

MODELS_DIR = os.path.join(BASE_DIR, "models")
TOKENIZER_DIR = os.path.join(MODELS_DIR, "tokenizer")
RETRIEVER_MODEL_DIR = os.path.join(MODELS_DIR, "retriever")
GENERATOR_MODEL_DIR = os.path.join(MODELS_DIR, "generator")

CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
RETRIEVER_CHECKPOINT_DIR = os.path.join(CHECKPOINTS_DIR, "retriever")
GENERATOR_CHECKPOINT_DIR = os.path.join(CHECKPOINTS_DIR, "generator")

INDEXES_DIR = os.path.join(BASE_DIR, "indexes")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Auto-create required directories
for d in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAINING_DATA_DIR, DOCUMENTS_DIR,
    TOKENIZER_DIR, RETRIEVER_MODEL_DIR, GENERATOR_MODEL_DIR,
    RETRIEVER_CHECKPOINT_DIR, GENERATOR_CHECKPOINT_DIR,
    INDEXES_DIR, LOGS_DIR
]:
    os.makedirs(d, exist_ok=True)

@dataclass
class TokenizerConfig:
    vocab_size: int = 8000
    min_frequency: int = 2
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    sep_token: str = "<SEP>"

@dataclass
class ChunkingConfig:
    strategy: str = "sentence"  # "fixed", "sentence", "token"
    chunk_size: int = 256
    chunk_overlap: int = 50

@dataclass
class ModelConfig:
    d_model: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 512

PRESETS: Dict[str, Dict[str, Any]] = {
    "SMALL": {
        "d_model": 128,
        "num_heads": 4,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "d_ff": 512,
        "batch_size": 8,
        "grad_accum_steps": 2,
    },
    "MEDIUM": {
        "d_model": 256,
        "num_heads": 8,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "d_ff": 1024,
        "batch_size": 16,
        "grad_accum_steps": 4,
    },
    "LARGE": {
        "d_model": 512,
        "num_heads": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "d_ff": 2048,
        "batch_size": 32,
        "grad_accum_steps": 4,
    }
}

@dataclass
class TrainingConfig:
    preset_name: str = "SMALL"
    learning_rate: float = 5e-4
    batch_size: int = 8
    epochs: int = 20
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0
    use_amp: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0  # 0 for Windows compatibility
    eval_every: int = 1
    save_every_epoch: bool = True

@dataclass
class RAGConfig:
    top_k: int = 3
    max_context_length: int = 512
    generation_max_tokens: int = 128
    decoding_strategy: str = "greedy"  # "greedy", "top_k", "top_p"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k_sampling: int = 50
