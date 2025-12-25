"""
Vocabulary class for caption tokenization.
Shared between SECOND-CC and LEVIR-MCI datasets.
"""

import json
from typing import List, Dict, Optional
from collections import Counter
import torch


class Vocabulary:
    """
    Vocabulary class for caption tokenization.
    
    Special tokens:
        <pad>: 0 - Padding token
        <start>: 1 - Start of sequence
        <end>: 2 - End of sequence
        <unk>: 3 - Unknown token
    """
    
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"
    
    PAD_IDX = 0
    START_IDX = 1
    END_IDX = 2
    UNK_IDX = 3
    
    def __init__(self, min_word_freq: int = 5):
        self.min_word_freq = min_word_freq
        
        # Initialize with special tokens
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.START_TOKEN: self.START_IDX,
            self.END_TOKEN: self.END_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()
        
    def build_vocab(self, sentences: List[List[str]]):
        """
        Build vocabulary from list of tokenized sentences.
        
        Args:
            sentences: List of tokenized sentences (list of word lists)
        """
        # Count word frequencies
        for sentence in sentences:
            self.word_freq.update(sentence)
        
        # Add words with frequency >= min_word_freq
        for word, freq in self.word_freq.items():
            if freq >= self.min_word_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def __len__(self) -> int:
        return len(self.word2idx)
    
    def encode(self, tokens: List[str], max_length: int = 50) -> torch.Tensor:
        """
        Convert tokens to indices with padding.
        
        Args:
            tokens: List of word tokens
            max_length: Maximum sequence length
        
        Returns:
            Tensor of token indices [max_length]
        """
        indices = [self.START_IDX]
        
        for token in tokens[:max_length - 2]:
            indices.append(self.word2idx.get(token, self.UNK_IDX))
        
        indices.append(self.END_IDX)
        
        # Pad to max_length
        while len(indices) < max_length:
            indices.append(self.PAD_IDX)
        
        return torch.tensor(indices[:max_length], dtype=torch.long)
    
    def decode(self, indices: torch.Tensor, skip_special: bool = True) -> str:
        """
        Convert indices back to string.
        
        Args:
            indices: Tensor of token indices
            skip_special: Skip special tokens in output
        
        Returns:
            Decoded string
        """
        words = []
        for idx in indices.tolist():
            if idx == self.END_IDX:
                break
            word = self.idx2word.get(idx, self.UNK_TOKEN)
            if skip_special and word in [self.PAD_TOKEN, self.START_TOKEN]:
                continue
            words.append(word)
        return ' '.join(words)
    
    def save(self, path: str):
        """Save vocabulary to JSON file."""
        data = {
            'word2idx': self.word2idx,
            'min_word_freq': self.min_word_freq,
            'word_freq': dict(self.word_freq)
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Load vocabulary from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        vocab = cls(min_word_freq=data['min_word_freq'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        vocab.word_freq = Counter(data.get('word_freq', {}))
        return vocab
