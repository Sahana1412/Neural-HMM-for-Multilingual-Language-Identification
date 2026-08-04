"""Noise generators for robustness testing (NoisyEval)."""

import logging
import random
import string
from typing import Callable, Dict, Optional

from matplotlib import text

from .normalizer import TextNormalizer


logger = logging.getLogger(__name__)


class NoiseGenerator:
    """Generate various types of text noise for evaluation."""
    
    # Keyboard neighbors for realistic substitution errors
    KEYBOARD_NEIGHBORS = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wrdsf',
        'f': 'drtgcv', 'g': 'fytdhvb', 'h': 'gyujnbv', 'i': 'uojkl',
        'j': 'huiklm', 'k': 'ijolm', 'l': 'kop', 'm': 'njkl',
        'n': 'bhjkm', 'o': 'ipklm', 'p': 'olkm', 'q': 'awe', 'r': 'wetgf',
        's': 'aedxzw', 't': 'ergfyh', 'u': 'yijkh', 'v': 'cfgbn',
        'w': 'qeasd', 'x': 'sdfcz', 'y': 'truhgj', 'z': 'axsd',
        '0': '9pl', '1': '2qw', '2': '13wqe', '3': '24ewqr', '4': '35rewrt',
        '5': '46tert', '6': '57rtyf', '7': '68ytfu', '8': '79uifu',
        '9': '80oiu'
    }
    
    def __init__(self, seed: int = 42):
        """Initialize noise generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        self.normalizer = TextNormalizer()
    
    def add_noise(
        self,
        text: str,
        noise_config: Dict[str, float]
    ) -> str:
    # Convert Config object to dict if needed
        if hasattr(noise_config, '_data'):
            noise_config = noise_config._data
    
        noisy = text
    # Accent stripping
        if noise_config.get("accent_strip_prob") and self.rng.random() < noise_config.get("accent_strip_prob", 0):
            noisy = self.normalizer.strip_accents_map(noisy)
    
    # Random deletion
        if noise_config.get("random_delete_prob") and self.rng.random() < noise_config.get("random_delete_prob", 0):
            noisy = self._random_delete(noisy, rate=0.1)
    
    # Random insertion
        if noise_config.get("random_insert_prob") and self.rng.random() < noise_config.get("random_insert_prob", 0):
            noisy = self._random_insert(noisy, rate=0.05)
    
    # Character substitution
        if noise_config.get("char_substitute_prob") and self.rng.random() < noise_config.get("char_substitute_prob", 0):
            noisy = self._char_substitute(noisy, rate=0.05, use_keyboard=True)
    
    # Truncation
        if noise_config.get("truncate_prob") and self.rng.random() < noise_config.get("truncate_prob", 0):
            noisy = self._truncate(noisy, factor=0.8)
    
    # Lowercasing
        if noise_config.get("lowercase_prob") and self.rng.random() < noise_config.get("lowercase_prob", 0):
            noisy = noisy.lower()
    
        return noisy
    
    def _random_delete(self, text: str, rate: float = 0.05) -> str:
        """Randomly delete characters.
        
        Args:
            text: Input text
            rate: Deletion rate (fraction of chars to delete)
            
        Returns:
            Text with random deletions
        """
        n_delete = max(1, int(len(text) * rate))
        indices = self.rng.sample(range(len(text)), min(n_delete, len(text)))
        result = "".join(char for i, char in enumerate(text) if i not in indices)
        return result if result else text[0]  # Keep at least one char
    
    def _random_insert(self, text: str, rate: float = 0.05) -> str:
        """Randomly insert characters.
        
        Args:
            text: Input text
            rate: Insertion rate
            
        Returns:
            Text with random insertions
        """
        n_insert = max(1, int(len(text) * rate))
        chars = list(text)
        
        for _ in range(n_insert):
            idx = self.rng.randint(0, len(chars))
            random_char = self.rng.choice(string.ascii_lowercase + string.digits + " ")
            chars.insert(idx, random_char)
        
        return "".join(chars)
    
    def _char_substitute(self, text: str, rate: float = 0.05, use_keyboard: bool = True) -> str:
        """Substitute characters with similar ones.
        
        Args:
            text: Input text
            rate: Substitution rate
            use_keyboard: Use keyboard neighbors if True, else random chars
            
        Returns:
            Text with substitutions
        """
        n_substitute = max(1, int(len(text) * rate))
        indices = self.rng.sample(range(len(text)), min(n_substitute, len(text)))
        
        chars = list(text)
        for idx in indices:
            char = chars[idx]
            if use_keyboard and char in self.KEYBOARD_NEIGHBORS:
                replacement = self.rng.choice(self.KEYBOARD_NEIGHBORS[char])
            else:
                replacement = self.rng.choice(string.ascii_lowercase + string.digits)
            chars[idx] = replacement
        
        return "".join(chars)
    
    def _truncate(self, text: str, factor: float = 0.8) -> str:
        """Truncate text to a fraction of its length.
        
        Args:
            text: Input text
            factor: Keep this fraction of text
            
        Returns:
            Truncated text
        """
        new_length = max(1, int(len(text) * factor))
        return text[:new_length]
