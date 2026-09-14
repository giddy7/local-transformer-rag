import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any

class RetrieverDataset(Dataset):
    """
    PyTorch Dataset for Contrastive / In-Batch Negative Retriever Training.
    Expects items with 'question', 'positive_doc', and optional 'negative_doc'.
    """
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        pos_doc = item["positive_doc"]
        neg_doc = item.get("negative_doc", "")

        q_ids = self.tokenizer.encode(question, max_length=self.max_length, pad=True)
        pos_ids = self.tokenizer.encode(pos_doc, max_length=self.max_length, pad=True)

        res = {
            "question_input_ids": torch.tensor(q_ids, dtype=torch.long),
            "pos_doc_input_ids": torch.tensor(pos_ids, dtype=torch.long),
        }

        if neg_doc:
            neg_ids = self.tokenizer.encode(neg_doc, max_length=self.max_length, pad=True)
            res["neg_doc_input_ids"] = torch.tensor(neg_ids, dtype=torch.long)

        return res

class GeneratorDataset(Dataset):
    """
    PyTorch Dataset for Seq2Seq Generator Training.
    Format:
      Encoder Input: <CONTEXT> chunk1 <SEP> chunk2 <SEP> Question
      Decoder Target: Answer
    """
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_enc_len: int = 512, max_dec_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item.get("context", "")
        question = item["question"]
        answer = item["answer"]

        # Prompt format: context + SEP + question
        if context:
            enc_text = f"{context} {self.tokenizer.sep_token} {question}"
        else:
            enc_text = question

        enc_ids = self.tokenizer.encode(enc_text, max_length=self.max_enc_len, pad=True)
        
        # Decoder target tokenization with BOS and EOS
        dec_input_text = f"{self.tokenizer.bos_token} {answer}"
        dec_target_text = f"{answer} {self.tokenizer.eos_token}"

        dec_in_ids = self.tokenizer.encode(dec_input_text, max_length=self.max_dec_len, pad=True)
        dec_target_ids = self.tokenizer.encode(dec_target_text, max_length=self.max_dec_len, pad=True)

        return {
            "encoder_input_ids": torch.tensor(enc_ids, dtype=torch.long),
            "decoder_input_ids": torch.tensor(dec_in_ids, dtype=torch.long),
            "decoder_target_ids": torch.tensor(dec_target_ids, dtype=torch.long),
        }
