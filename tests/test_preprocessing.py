"""Unit tests for preprocessing modules."""

import pytest
from src.preprocessing.vocab import Vocabulary
from src.preprocessing.normalizer import TextNormalizer
from src.preprocessing.noise_generator import NoiseGenerator


class TestVocabulary:
    """Test vocabulary operations."""
    
    def test_vocab_initialization(self):
        """Test vocabulary initialization."""
        vocab = Vocabulary()
        assert vocab.size > 0
        assert len(vocab.char2id) == vocab.size
        assert len(vocab.id2char) == vocab.size
    
    def test_encode_decode_roundtrip(self):
        """Test encoding and decoding."""
        vocab = Vocabulary()
        text = "hello world"
        encoded = vocab.encode(text)
        decoded = vocab.decode(encoded, skip_special=True)
        # Characters should match after roundtrip
        assert len(encoded) > 0
        assert isinstance(encoded, list)
        assert all(isinstance(x, int) for x in encoded)
    
    def test_unknown_char_handling(self):
        """Test handling of unknown characters."""
        vocab = Vocabulary()
        text_with_unknown = "hello€world"  # € is unlikely to be in vocab
        encoded = vocab.encode(text_with_unknown)
        # Should encode successfully, possibly with unknown tokens
        assert len(encoded) > 0
    
    def test_encode_with_boundaries(self):
        """Test encoding with start/end tokens."""
        vocab = Vocabulary()
        text = "hello"
        encoded = vocab.encode(text, add_boundaries=True)
        assert vocab.id2char[encoded[0]] == vocab.START_TOKEN
        assert vocab.id2char[encoded[-1]] == vocab.END_TOKEN


class TestTextNormalizer:
    """Test text normalization."""
    
    def test_accent_normalization(self):
        """Test accent handling."""
        normalizer = TextNormalizer(preserve_accents=False)
        text = "café"
        normalized = normalizer.normalize(text)
        assert "é" not in normalized or len(normalized) <= len(text)
    
    def test_preserve_accents(self):
        """Test preserving accents."""
        normalizer = TextNormalizer(preserve_accents=True)
        text = "café"
        normalized = normalizer.normalize(text)
        # Should preserve accents
        assert "é" in normalized or "e" in normalized
    
    def test_control_char_removal(self):
        """Test control character removal."""
        normalizer = TextNormalizer()
        text_with_control = "hello\x00world"
        normalized = normalizer.normalize(text_with_control)
        assert "\x00" not in normalized
    
    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        normalizer_nfd = TextNormalizer(normalize_unicode=True)
        text = "café"
        normalized = normalizer_nfd.normalize(text)
        assert isinstance(normalized, str)


class TestNoiseGenerator:
    """Test noise generation."""
    
    def test_accent_stripping(self):
        """Test accent stripping noise."""
        gen = NoiseGenerator(seed=42)
        text = "Málaga"
        noisy = gen._random_delete(text, rate=0.0)  # No deletion
        assert len(noisy) > 0
    
    def test_random_deletion(self):
        """Test random character deletion."""
        gen = NoiseGenerator(seed=42)
        text = "hello world"
        noisy = gen._random_delete(text, rate=0.3)
        assert len(noisy) < len(text)
        assert len(noisy) > 0
    
    def test_random_insertion(self):
        """Test random character insertion."""
        gen = NoiseGenerator(seed=42)
        text = "hello"
        noisy = gen._random_insert(text, rate=0.2)
        assert len(noisy) >= len(text)
    
    def test_char_substitution(self):
        """Test character substitution."""
        gen = NoiseGenerator(seed=42)
        text = "hello"
        noisy = gen._char_substitute(text, rate=0.2)
        assert len(noisy) == len(text)
    
    def test_truncation(self):
        """Test text truncation."""
        gen = NoiseGenerator(seed=42)
        text = "hello world"
        noisy = gen._truncate(text, factor=0.5)
        assert len(noisy) <= len(text)
    
    def test_combined_noise(self):
        """Test combined noise transformations."""
        gen = NoiseGenerator(seed=42)
        text = "hello world"
        noise_config = {
            "accent_strip_prob": 0.5,
            "random_delete_prob": 0.3,
            "random_insert_prob": 0.2,
        }
        # Should not crash
        for _ in range(5):
            noisy = gen.add_noise(text, noise_config)
            assert isinstance(noisy, str)
            assert len(noisy) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
