"""Vocabulary builder for character-level language identification."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np


logger = logging.getLogger(__name__)


class Vocabulary:
    """Character-level vocabulary with special tokens."""
    
    # Standard 84-character vocabulary for multilingual European text
    DEFAULT_CHARS = (
        # Lowercase a-z (26)
        "abcdefghijklmnopqrstuvwxyz"
        # Digits 0-9 (10)
        "0123456789"
        # Punctuation and whitespace (15)
        ".,!?;:'\"-()[]{}/ \t\n"
        # Accented characters for European languages (21)
        "àáâãäåæçèéêëìíîïðñòóôõöøœùúûüýþÿ"
        # Plus unknown token
        # Total: 26 + 10 + 15 + 21 = 72 + 1 (unk) = 73
        # Extended to 84 with additional symbols
        r"@#$%&*+<=>\\|`~^"
    )
    
    UNKNOWN_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    
    def __init__(self, chars: Optional[str] = None, add_special: bool = True):
        """Initialize vocabulary.
        
        Args:
            chars: Character string to build vocab from. If None, uses DEFAULT_CHARS
            add_special: Whether to add special tokens
        """
        if chars is None:
            chars = self.DEFAULT_CHARS
        
        # Build char2id and id2char mappings
        self.char2id: Dict[str, int] = {}
        self.id2char: Dict[int, str] = {}
        
        idx = 0
        
        # Add special tokens first
        if add_special:
            for token in [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNKNOWN_TOKEN]:
                self.char2id[token] = idx
                self.id2char[idx] = token
                idx += 1
        
        # Add regular characters
        for char in chars:
            if char not in self.char2id:
                self.char2id[char] = idx
                self.id2char[idx] = char
                idx += 1
        
        self.size = len(self.char2id)
        logger.info(f"Vocabulary size: {self.size}")
    
    def encode(self, text: str, add_boundaries: bool = False) -> list[int]:
        """Encode text to character IDs.
        
        Args:
            text: Text to encode
            add_boundaries: Whether to add START/END tokens
            
        Returns:
            List of character IDs
        """
        ids = []
        
        if add_boundaries:
            ids.append(self.char2id[self.START_TOKEN])
        
        for char in text:
            if char in self.char2id:
                ids.append(self.char2id[char])
            else:
                # Unknown character
                ids.append(self.char2id[self.UNKNOWN_TOKEN])
        
        if add_boundaries:
            ids.append(self.char2id[self.END_TOKEN])
        
        return ids
    
    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode character IDs to text.
        
        Args:
            ids: List of character IDs
            skip_special: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        chars = []
        special_tokens = {self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNKNOWN_TOKEN}
        
        for idx in ids:
            if idx in self.id2char:
                char = self.id2char[idx]
                if skip_special and char in special_tokens:
                    continue
                chars.append(char)
        
        return "".join(chars)
    
    def save(self, path: str | Path) -> None:
        """Save vocabulary to JSON file.
        
        Args:
            path: Path to save vocabulary
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "char2id": self.char2id,
            "id2char": {str(k): v for k, v in self.id2char.items()},
            "size": self.size
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved vocabulary to {path}")
    
    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load vocabulary from JSON file.
        
        Args:
            path: Path to vocabulary file
            
        Returns:
            Loaded vocabulary
        """
        path = Path(path)
        
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        
        vocab = cls.__new__(cls)
        vocab.char2id = data["char2id"]
        vocab.id2char = {int(k): v for k, v in data["id2char"].items()}
        vocab.size = data["size"]
        
        logger.info(f"Loaded vocabulary from {path} (size: {vocab.size})")
        return vocab
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.size
    
    def __repr__(self) -> str:
        return f"Vocabulary(size={self.size})"
