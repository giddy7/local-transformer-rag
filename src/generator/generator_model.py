import torch
import torch.nn as nn
from src.transformer.transformer import TransformerSeq2Seq

class TransformerGeneratorModel(nn.Module):
    """
    Wrapper around TransformerSeq2Seq for sequence-to-sequence answer generation.
    """
    def __init__(self, vocab_size: int, 
                 d_model: int = 128, 
                 num_heads: int = 4, 
                 num_encoder_layers: int = 3, 
                 num_decoder_layers: int = 3, 
                 d_ff: int = 512, 
                 dropout: float = 0.1, 
                 pad_idx: int = 0):
        super().__init__()
        
        self.transformer = TransformerSeq2Seq(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            pad_idx=pad_idx
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return self.transformer(src, tgt)
