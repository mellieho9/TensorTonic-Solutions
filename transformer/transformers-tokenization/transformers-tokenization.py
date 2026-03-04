import numpy as np
from typing import List, Dict
import string

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        self.translator = str.maketrans('', '', string.punctuation)
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"


    def add_to_vocab(self, text: str) -> None:
        if text not in self.word_to_id:
            self.word_to_id[text] = self.vocab_size
            self.id_to_word[self.vocab_size] = text
            self.vocab_size += 1

    def normalize(self, text: str) -> str:
        return text.lower().translate(self.translator)
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        print(texts)
        for token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            self.add_to_vocab(token)
        for text in texts: 
            for word in text.split():
                normalized = self.normalize(word)
                self.add_to_vocab(normalized)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        texts = text.split(" ")
        encodings = []
        for text in texts:
            normalized = self.normalize(text)
            if normalized in self.word_to_id:
                encodings.append(self.word_to_id[normalized])
            else:
                encodings.append(self.word_to_id[self.unk_token])
        return encodings
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        return " ".join(self.id_to_word.get(id, self.unk_token) for id in ids)
