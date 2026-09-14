import torch
import torch.nn.functional as F

def greedy_decode(model, src: torch.Tensor, bos_idx: int, eos_idx: int, max_len: int = 128, repetition_penalty: float = 1.3) -> torch.Tensor:
    """
    Greedy autoregressive decoding with repetition penalty for sequence generation.
    src shape: (1, src_len)
    """
    model.eval()
    device = src.device
    
    with torch.no_grad():
        memory = model.transformer.encode(src)
        ys = torch.ones(1, 1, dtype=torch.long, device=device).fill_(bos_idx)

        for _ in range(max_len - 1):
            out = model.transformer.decode(ys, memory)
            logits = out[:, -1, :].clone()

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(ys[0].tolist()):
                    if token_id != bos_idx and token_id != eos_idx:
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= repetition_penalty
                        else:
                            logits[0, token_id] *= repetition_penalty

            _, next_word = torch.max(logits, dim=1)
            next_word_idx = next_word.item()

            if next_word_idx == eos_idx:
                break

            ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long, device=device).fill_(next_word_idx)], dim=1)

    return ys[0]

def sample_decode(model, src: torch.Tensor, bos_idx: int, eos_idx: int, max_len: int = 128,
                  temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9, repetition_penalty: float = 1.25) -> torch.Tensor:
    """
    Temperature, Top-K, and Top-P (Nucleus) sampling autoregressive decoding.
    """
    model.eval()
    device = src.device
    
    with torch.no_grad():
        memory = model.transformer.encode(src)
        ys = torch.ones(1, 1, dtype=torch.long, device=device).fill_(bos_idx)

        for _ in range(max_len - 1):
            out = model.transformer.decode(ys, memory)
            logits = out[:, -1, :].clone() / max(temperature, 1e-5)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(ys[0].tolist()):
                    if token_id != bos_idx and token_id != eos_idx:
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= repetition_penalty
                        else:
                            logits[0, token_id] *= repetition_penalty

            # Top-K filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')

            # Top-P (Nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_word = torch.multinomial(probs, num_samples=1)
            next_word_idx = next_word.item()

            if next_word_idx == eos_idx:
                break

            ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long, device=device).fill_(next_word_idx)], dim=1)

    return ys[0]
