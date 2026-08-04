"""Unit tests for model implementations."""

import pytest
import numpy as np
from src.models.markov_chain import MarkovChain, MarkovChainLangID
from src.models.hmm import HiddenMarkovModel, HiddenMarkovModelLangID


class TestMarkovChain:
    """Test Markov Chain implementation."""
    
    def test_initialization(self):
        """Test MC initialization."""
        mc = MarkovChain(n_gram_order=2, vocab_size=84)
        assert mc.n_gram_order == 2
        assert mc.vocab_size == 84
    
    def test_training(self):
        """Test MC training."""
        mc = MarkovChain(n_gram_order=2, vocab_size=84)
        sequence = [1, 2, 3, 4, 5]
        mc.train(sequence)
        # Should have learned n-gram statistics
        assert len(mc.counts) > 0
    
    def test_scoring(self):
        """Test MC scoring."""
        mc = MarkovChain(n_gram_order=2, vocab_size=84)
        sequence = [1, 2, 3, 4, 5]
        mc.train(sequence)
        score = mc.score(sequence)
        # Score should be a finite number
        assert isinstance(score, (int, float, np.ndarray))
        assert np.isfinite(score)
    
    def test_probability_computation(self):
        """Test probability computation."""
        mc = MarkovChain(n_gram_order=1)
        sequence = [1, 2, 1, 2, 1, 2]
        mc.train(sequence)
        
        # Test probability for observed context
        prob = mc.prob(char=1, context=(2,))
        assert np.isfinite(prob)
        assert prob <= 0  # Log probability


class TestHMM:
    """Test Hidden Markov Model."""
    
    def test_initialization(self):
        """Test HMM initialization."""
        hmm = HiddenMarkovModel(n_states=8, n_symbols=84)
        assert hmm.n_states == 8
        assert hmm.n_symbols == 84
        assert hmm.log_pi.shape == (8,)
        assert hmm.log_A.shape == (8, 8)
        assert hmm.log_B.shape == (8, 84)
    
    def test_forward_algorithm(self):
        """Test forward algorithm."""
        hmm = HiddenMarkovModel(n_states=4, n_symbols=10)
        observations = np.array([0, 1, 2, 3, 1])
        alpha, likelihood = hmm.forward(observations)
        
        assert alpha.shape == (5, 4)
        assert np.isfinite(likelihood)
    
    def test_backward_algorithm(self):
        """Test backward algorithm."""
        hmm = HiddenMarkovModel(n_states=4, n_symbols=10)
        observations = np.array([0, 1, 2, 3, 1])
        alpha, _ = hmm.forward(observations)
        beta = hmm.backward(observations, alpha)
        
        assert beta.shape == (5, 4)
        assert np.all(np.isfinite(beta))
    
    def test_viterbi_algorithm(self):
        """Test Viterbi decoding."""
        hmm = HiddenMarkovModel(n_states=4, n_symbols=10)
        observations = np.array([0, 1, 2, 3, 1])
        path, likelihood = hmm.viterbi(observations)
        
        assert len(path) == 5
        assert all(0 <= s < 4 for s in path)
        assert np.isfinite(likelihood)
    
    def test_baum_welch_training(self):
        """Test Baum-Welch training."""
        hmm = HiddenMarkovModel(n_states=4, n_symbols=10)
        observations = np.array([0, 1, 2, 3, 1, 0, 2, 3, 1])
        
        initial_likelihood = hmm.forward(observations)[1]
        trained_likelihood = hmm.baum_welch(observations, n_iterations=5)
        
        assert np.isfinite(trained_likelihood)
        # Likelihood should improve (or stay similar)
        assert trained_likelihood >= initial_likelihood - 1e-3
    
    def test_scoring(self):
        """Test HMM scoring."""
        hmm = HiddenMarkovModel(n_states=4, n_symbols=10)
        observations = np.array([0, 1, 2, 3, 1])
        score = hmm.score(observations)
        
        assert np.isfinite(score)


class TestMarkovChainLangID:
    """Test multilingual Markov Chain model."""
    
    def test_initialization(self):
        """Test initialization."""
        languages = ["en", "es", "fr"]
        model = MarkovChainLangID(languages, n_gram_order=2)
        assert len(model.models) == 3
        assert all(lang in model.models for lang in languages)
    
    def test_training_and_prediction(self):
        """Test training and prediction."""
        languages = ["en", "es", "fr"]
        model = MarkovChainLangID(languages, n_gram_order=2)
        
        # Create fake training data
        train_examples = [
            {"char_ids": [1, 2, 3, 4], "language": "en"},
            {"char_ids": [5, 6, 7, 8], "language": "es"},
            {"char_ids": [9, 10, 11, 12], "language": "fr"},
        ]
        
        model.train(train_examples)
        
        # Test prediction
        pred_lang, confidence = model.predict([1, 2, 3])
        assert pred_lang in languages
        assert 0 <= confidence <= 1


class TestHMMLangID:
    """Test multilingual HMM model."""
    
    def test_initialization(self):
        """Test initialization."""
        languages = ["en", "es", "fr"]
        model = HiddenMarkovModelLangID(languages, n_states=4, n_symbols=10)
        assert len(model.models) == 3
    
    def test_training_and_prediction(self):
        """Test training and prediction."""
        languages = ["en", "es"]
        model = HiddenMarkovModelLangID(languages, n_states=4, n_symbols=10)
        
        # Create fake training data
        train_examples = [
            {"char_ids": [1, 2, 3, 4, 5], "language": "en"},
            {"char_ids": [1, 2, 3, 4, 5], "language": "en"},
            {"char_ids": [6, 7, 8, 9, 0], "language": "es"},
            {"char_ids": [6, 7, 8, 9, 0], "language": "es"},
        ]
        
        model.train(train_examples, n_iterations=2)
        
        # Test prediction
        pred_lang, confidence = model.predict([1, 2, 3])
        assert pred_lang in languages
        assert 0 <= confidence <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
