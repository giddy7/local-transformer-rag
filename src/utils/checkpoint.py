import os
import torch
from typing import Dict, Any, Optional

class CheckpointManager:
    """
    Manages saving and restoring training state checkpoints for Retriever and Generator models.
    """
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, 
                        model: torch.nn.Module, 
                        optimizer: torch.optim.Optimizer, 
                        scheduler: Optional[Any], 
                        epoch: int, 
                        step: int, 
                        loss_history: list, 
                        val_metrics: dict, 
                        config: dict, 
                        filename: str = "latest.pt") -> str:
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "loss_history": loss_history,
            "val_metrics": val_metrics,
            "config": config,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        torch.save(checkpoint, filepath)
        print(f"[OK] Saved checkpoint to: {filepath}")
        return filepath

    def load_checkpoint(self, 
                        model: torch.nn.Module, 
                        optimizer: Optional[torch.optim.Optimizer] = None, 
                        scheduler: Optional[Any] = None, 
                        filename: str = "latest.pt", 
                        device: str = "cpu") -> Dict[str, Any]:
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint file found at: {filepath}")

        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "rng_state" in checkpoint and checkpoint["rng_state"] is not None:
            torch.set_rng_state(checkpoint["rng_state"])

        print(f"[OK] Successfully loaded checkpoint from {filepath} (Epoch {checkpoint.get('epoch', 0)}, Step {checkpoint.get('step', 0)})")
        return checkpoint

    def latest_checkpoint_exists(self, filename: str = "latest.pt") -> bool:
        return os.path.exists(os.path.join(self.checkpoint_dir, filename))
