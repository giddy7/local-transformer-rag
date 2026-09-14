import torch
import numpy as np
from typing import List, Dict, Any

class DenseRetriever:
    """
    Encodes text queries and document chunks into normalized vector embeddings.
    """
    def __init__(self, model: torch.nn.Module, tokenizer, device: str = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def encode_text(self, text: str, max_length: int = 256) -> np.ndarray:
        ids = self.tokenizer.encode(text, max_length=max_length, pad=True)
        tensor_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            emb = self.model.question_encoder(tensor_ids) if hasattr(self.model, 'question_encoder') else self.model(tensor_ids)
        return emb.cpu().numpy()[0]

    def encode_batch(self, texts: List[str], max_length: int = 256, is_doc: bool = False) -> np.ndarray:
        batch_ids = [self.tokenizer.encode(t, max_length=max_length, pad=True) for t in texts]
        tensor_ids = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            if hasattr(self.model, 'doc_encoder') and is_doc:
                emb = self.model.doc_encoder(tensor_ids)
            elif hasattr(self.model, 'question_encoder'):
                emb = self.model.question_encoder(tensor_ids)
            else:
                emb = self.model(tensor_ids)
        return emb.cpu().numpy()
