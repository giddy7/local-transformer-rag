import os
import json
from typing import List, Union
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

class LocalTokenizer:
    """
    Subword BPE Tokenizer for Transformer models.
    Supports special tokens: <PAD>, <UNK>, <BOS>, <EOS>, <SEP>.
    """
    def __init__(self, vocab_size: int = 8000, 
                 pad_token: str = "<PAD>", 
                 unk_token: str = "<UNK>", 
                 bos_token: str = "<BOS>", 
                 eos_token: str = "<EOS>", 
                 sep_token: str = "<SEP>"):
        
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token, sep_token]
        self.tokenizer = None
        self._init_tokenizer()

    def _init_tokenizer(self):
        # Initialize BPE model
        self.tokenizer = Tokenizer(models.BPE(unk_token=self.unk_token))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()

    @property
    def pad_id(self) -> int:
        return self.tokenizer.token_to_id(self.pad_token)

    @property
    def unk_id(self) -> int:
        return self.tokenizer.token_to_id(self.unk_token)

    @property
    def bos_id(self) -> int:
        return self.tokenizer.token_to_id(self.bos_token)

    @property
    def eos_id(self) -> int:
        return self.tokenizer.token_to_id(self.eos_token)

    @property
    def sep_id(self) -> int:
        return self.tokenizer.token_to_id(self.sep_token)

    def train_from_files(self, files: List[str], save_path: str = None):
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        self.tokenizer.train(files, trainer)
        if save_path:
            self.save(save_path)

    def train_from_iterator(self, iterator, save_path: str = None):
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        self.tokenizer.train_from_iterator(iterator, trainer)
        if save_path:
            self.save(save_path)

    def encode(self, text: str, max_length: int = None, pad: bool = False) -> List[int]:
        if not self.tokenizer:
            raise ValueError("Tokenizer is not initialized or loaded.")
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids

        if max_length:
            if len(ids) > max_length:
                ids = ids[:max_length]
            elif pad and len(ids) < max_length:
                ids = ids + [self.pad_id] * (max_length - len(ids))
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if not self.tokenizer:
            raise ValueError("Tokenizer is not initialized or loaded.")
        # Filter padding/special tokens if requested
        if skip_special_tokens:
            special_ids = {self.pad_id, self.bos_id, self.eos_id, self.sep_id}
            ids = [i for i in ids if i not in special_ids and i is not None]
        return self.tokenizer.decode(ids)

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, "tokenizer.json")
        self.tokenizer.save(file_path)
        meta_path = os.path.join(directory, "meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "special_tokens": self.special_tokens,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "sep_token": self.sep_token,
            }, f, indent=2)

    def load(self, directory: str):
        file_path = os.path.join(directory, "tokenizer.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No tokenizer file found at {file_path}")
        self.tokenizer = Tokenizer.from_file(file_path)
        meta_path = os.path.join(directory, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.vocab_size = meta.get("vocab_size", self.vocab_size)
                self.pad_token = meta.get("pad_token", self.pad_token)
                self.unk_token = meta.get("unk_token", self.unk_token)
                self.bos_token = meta.get("bos_token", self.bos_token)
                self.eos_token = meta.get("eos_token", self.eos_token)
                self.sep_token = meta.get("sep_token", self.sep_token)

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size() if self.tokenizer else 0
