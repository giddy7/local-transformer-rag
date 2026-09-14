import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any

from config import (
    TOKENIZER_DIR, RETRIEVER_CHECKPOINT_DIR, LOGS_DIR,
    TRAINING_DATA_DIR, PRESETS, TrainingConfig, ModelConfig
)
from src.tokenizer.tokenizer import LocalTokenizer
from src.data.dataset import RetrieverDataset
from src.retriever.embedding_model import DualEncoderRetrieverModel
from src.utils.checkpoint import CheckpointManager
from src.utils.seed import set_seed

def info_nce_loss(q_emb: torch.Tensor, doc_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    # q_emb, doc_emb shape: (batch_size, d_model) normalized
    sim_matrix = torch.matmul(q_emb, doc_emb.T) / temperature
    labels = torch.arange(q_emb.size(0), device=q_emb.device, dtype=torch.long)
    return F.cross_entropy(sim_matrix, labels)

def run_retriever_training(debug: bool = False, overfit_test: bool = False, resume: bool = False, preset: str = "SMALL", epochs: int = None):
    set_seed(42)
    p_cfg = PRESETS.get(preset, PRESETS["SMALL"])
    m_cfg = ModelConfig(
        d_model=p_cfg["d_model"],
        num_heads=p_cfg["num_heads"],
        num_encoder_layers=p_cfg["num_encoder_layers"],
        d_ff=p_cfg["d_ff"]
    )
    t_cfg = TrainingConfig(
        preset_name=preset,
        batch_size=p_cfg["batch_size"],
        grad_accum_steps=p_cfg["grad_accum_steps"]
    )

    print(f"[*] Initializing Retriever Training (Preset: {preset}, Device: {t_cfg.device})")

    # Load Tokenizer
    tokenizer = LocalTokenizer()
    if not os.path.exists(os.path.join(TOKENIZER_DIR, "tokenizer.json")):
        from src.tokenizer.train_tokenizer import train_tokenizer
        train_tokenizer()
    tokenizer.load(TOKENIZER_DIR)

    # Load Data
    data_file = os.path.join(TRAINING_DATA_DIR, "retriever_data.json")
    if not os.path.exists(data_file):
        from prepare_data import prepare_sample_data
        prepare_sample_data()

    with open(data_file, "r", encoding="utf-8") as f:
        retriever_data = json.load(f)

    if debug:
        print("[!] RUNNING IN DEBUG MODE (Small subset & 2 epochs)")
        retriever_data = retriever_data[:8]
        total_epochs = 2
    elif overfit_test:
        print("[!] RUNNING OVERFITTING TEST (10 examples, 50 epochs)")
        retriever_data = retriever_data[:10]
        total_epochs = 50
    else:
        total_epochs = epochs if epochs is not None else t_cfg.epochs

    dataset = RetrieverDataset(retriever_data, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=min(t_cfg.batch_size, len(dataset)), shuffle=True, num_workers=t_cfg.num_workers)

    model = DualEncoderRetrieverModel(
        vocab_size=len(tokenizer),
        d_model=m_cfg.d_model,
        num_heads=m_cfg.num_heads,
        num_layers=m_cfg.num_encoder_layers,
        d_ff=m_cfg.d_ff,
        pad_idx=tokenizer.pad_id
    ).to(t_cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=t_cfg.learning_rate, weight_decay=0.01)
    ckpt_manager = CheckpointManager(RETRIEVER_CHECKPOINT_DIR)

    start_epoch = 1
    loss_history = []

    if resume and ckpt_manager.latest_checkpoint_exists():
        ckpt = ckpt_manager.load_checkpoint(model, optimizer, device=t_cfg.device)
        start_epoch = ckpt.get("epoch", 1) + 1
        loss_history = ckpt.get("loss_history", [])

    print(f"[*] Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[*] Total Batches per Epoch: {len(dataloader)}")

    best_loss = float("inf")
    model.train()

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader, 1):
            q_ids = batch["question_input_ids"].to(t_cfg.device)
            pos_ids = batch["pos_doc_input_ids"].to(t_cfg.device)

            q_emb, doc_emb = model(q_ids, pos_ids)
            loss = info_nce_loss(q_emb, doc_emb)
            loss = loss / t_cfg.grad_accum_steps
            loss.backward()

            if step % t_cfg.grad_accum_steps == 0 or step == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * t_cfg.grad_accum_steps

        avg_loss = running_loss / len(dataloader)
        loss_history.append({"epoch": epoch, "loss": avg_loss})
        elapsed = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{total_epochs:02d}] | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_manager.save_checkpoint(
                model=model, optimizer=optimizer, scheduler=None,
                epoch=epoch, step=epoch * len(dataloader),
                loss_history=loss_history, val_metrics={"best_loss": best_loss},
                config=m_cfg.__dict__, filename="best_model.pt"
            )

        if epoch % 5 == 0 or epoch == total_epochs or t_cfg.save_every_epoch:
            ckpt_manager.save_checkpoint(
                model=model, optimizer=optimizer, scheduler=None,
                epoch=epoch, step=epoch * len(dataloader),
                loss_history=loss_history, val_metrics={"loss": avg_loss},
                config=m_cfg.__dict__, filename="latest.pt"
            )

    # Save training log JSON
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(os.path.join(LOGS_DIR, "retriever_history.json"), "w") as f:
        json.dump(loss_history, f, indent=2)

    if overfit_test:
        if best_loss < 0.5:
            print(f"[OK] OVERFITTING TEST PASSED! Final Loss: {best_loss:.4f}")
        else:
            print(f"[WARNING] Overfitting test loss was higher than expected: {best_loss:.4f}")

    print("[OK] Retriever Training Finished Successfully!")
    return model
