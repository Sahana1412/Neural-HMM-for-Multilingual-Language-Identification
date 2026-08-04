"""Unicode normalization for multilingual text."""

import logging
import re
import unicodedata
from typing import Optional


logger = logging.getLogger(__name__)


class TextNormalizer:
    """Normalize Unicode text with accent handling options."""
    
    # Accent mapping for advanced accent stripping
    ACCENT_MAP = {
        'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a', 'æ': 'ae',
        'ç': 'c', 'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
        'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i', 'ð': 'd',
        'ñ': 'n', 'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ø': 'o', 'œ': 'oe',
        'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u', 'ý': 'y', 'þ': 'th', 'ÿ': 'y'
    }
    
    def __init__(self, normalize_unicode: bool = True, preserve_accents: bool = False):
        """Initialize normalizer.
        
        Args:
            normalize_unicode: Apply NFD normalization
            preserve_accents: Keep accents (for CleanEval)
        """
        self.normalize_unicode = normalize_unicode
        self.preserve_accents = preserve_accents
    
    def normalize(self, text: str) -> str:
        """Normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Apply Unicode normalization (NFD = decomposed form)
        if self.normalize_unicode:
            text = unicodedata.normalize("NFD", text)
        
        if not self.preserve_accents:
            text = self._strip_accents(text)
        
        # Remove control characters
        text = "".join(char for char in text if unicodedata.category(char)[0] != "C")
        
        return text
    
    @staticmethod
    def _strip_accents(text: str) -> str:
        """Strip accents from text using decomposition.
        
        Args:
            text: Input text
            
        Returns:
            Text with accents removed
        """
        # Decompose into base + combining marks
        text = unicodedata.normalize("NFD", text)
        
        # Remove combining marks (accents)
        output = []
        for char in text:
            if unicodedata.category(char) != "Mn":  # Mn = Mark, Nonspacing
                output.append(char)
        
        # Recompose
        return unicodedata.normalize("NFC", "".join(output))
    
    @staticmethod
    def strip_accents_map(text: str) -> str:
        """Strip accents using explicit character mapping.
        
        Args:
            text: Input text
            
        Returns:
            Text with accents removed
        """
        result = []
        for char in text.lower():
            if char in TextNormalizer.ACCENT_MAP:
                result.append(TextNormalizer.ACCENT_MAP[char])
            else:
                result.append(char)
        return "".join(result)
    
    @staticmethod
    def remove_control_chars(text: str) -> str:
        """Remove Unicode control characters."""
        return "".join(char for char in text if unicodedata.category(char)[0] != "C")
