"""Character-level Markov Chain language model."""

import logging
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


class MarkovChain:
    """N-gram Markov Chain language model with Laplace/Kneser-Ney smoothing."""
    
    def __init__(
        self,
        n_gram_order: int = 2,
        smoothing: str = "laplace",
        alpha: float = 1.0,
        vocab_size: int = 84
    ):
        """Initialize Markov Chain model.
        
        Args:
            n_gram_order: N-gram order (1-4)
            smoothing: "laplace" or "kneser_ney"
            alpha: Laplace smoothing parameter
            vocab_size: Size of vocabulary
        """
        assert 1 <= n_gram_order <= 4, "n_gram_order must be 1-4"
        assert smoothing in ["laplace", "kneser_ney"], "Unknown smoothing method"
        
        self.n_gram_order = n_gram_order
        self.smoothing = smoothing
        self.alpha = alpha
        self.vocab_size = vocab_size
        
        # Counts: {context: {char: count}}
        self.counts: Dict[Tuple[int, ...], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.context_totals: Dict[Tuple[int, ...], int] = defaultdict(int)
        
        # For Kneser-Ney
        self.continuation_counts: Dict[int, int] = defaultdict(int)
        self.unique_contexts: Dict[Tuple[int, ...], set] = defaultdict(set)
    
    def train(self, char_ids: list[int]) -> None:
        """Train language model on character sequence.
        
        Args:
            char_ids: List of character IDs
        """
        for i in range(len(char_ids)):
            for order in range(1, self.n_gram_order + 1):
                if i >= order - 1:
                    context = tuple(char_ids[i - order + 1:i])
                    char = char_ids[i]
                    
                    self.counts[context][char] += 1
                    self.context_totals[context] += 1
                    
                    if self.smoothing == "kneser_ney":
                        self.continuation_counts[char] += 1
                        self.unique_contexts[context[1:] if context else ()].add(char)
    
    def prob(self, char: int, context: Tuple[int, ...]) -> float:
        """Compute probability of character given context.
        
        Args:
            char: Character ID
            context: Context tuple
            
        Returns:
            Log-space probability
        """
        context = tuple(context[-(self.n_gram_order - 1):])  # Trim to order
        
        if self.smoothing == "laplace":
            return self._laplace_prob(char, context)
        else:
            return self._kneser_ney_prob(char, context)
    
    def _laplace_prob(self, char: int, context: Tuple[int, ...]) -> float:
        """Laplace (add-k) smoothing probability."""
        count = self.counts[context].get(char, 0)
        total = self.context_totals[context]
        
        # Laplace: (count + alpha) / (total + alpha * vocab_size)
        prob = (count + self.alpha) / (total + self.alpha * self.vocab_size)
        
        return np.log(prob) if prob > 0 else -np.inf
    
    def _kneser_ney_prob(self, char: int, context: Tuple[int, ...]) -> float:
        """Kneser-Ney smoothing probability."""
        # Simplified Kneser-Ney: use continuation probabilities
        # P_KN(w|context) ~ uniqueness of w across contexts
        
        count = self.counts[context].get(char, 0)
        total = self.context_totals[context]
        continuation = self.continuation_counts[char]
        
        # Discount and backoff
        discount = 0.75
        if total > 0:
            prob = max(count - discount, 0) / total
        else:
            prob = 0
        
        # Backoff term
        if context:
            backoff_context = context[1:]
            backoff_prob = self._kneser_ney_prob(char, backoff_context)
            lambda_weight = (discount / total) * len(self.unique_contexts[context])
            prob += lambda_weight * np.exp(backoff_prob)
        
        return np.log(max(prob, 1e-10))
    
    def score(self, char_ids: list[int]) -> float:
        """Score sequence with log-likelihood.
        
        Args:
            char_ids: List of character IDs
            
        Returns:
            Log-likelihood score (higher is better)
        """
        log_likelihood = 0.0
        
        for i in range(1, len(char_ids)):
            context = tuple(char_ids[max(0, i - self.n_gram_order + 1):i])
            char = char_ids[i]
            log_likelihood += self.prob(char, context)
        
        # Normalize by sequence length for fair comparison
        return log_likelihood / max(len(char_ids) - 1, 1)
    
    def __repr__(self) -> str:
        n_contexts = len(self.counts)
        return f"MarkovChain(order={self.n_gram_order}, contexts={n_contexts}, smoothing={self.smoothing})"


class MarkovChainLangID:
    """Multilingual language identifier using Markov Chains."""
    
    def __init__(
        self,
        languages: list[str],
        n_gram_order: int = 2,
        smoothing: str = "laplace",
        vocab_size: int = 84
    ):
        """Initialize multilingual identifier.
        
        Args:
            languages: List of language codes
            n_gram_order: N-gram order
            smoothing: Smoothing method
            vocab_size: Vocabulary size
        """
        self.languages = languages
        self.models = {
            lang: MarkovChain(n_gram_order, smoothing, vocab_size=vocab_size)
            for lang in languages
        }
    
    def train(self, examples: list[Dict[str, any]], language_key: str = "language") -> None:
        """Train per-language models.
        
        Args:
            examples: List of dicts with "char_ids" and language info
            language_key: Key in dict containing language label
        """
        for example in examples:
            lang = example[language_key]
            if lang in self.models:
                self.models[lang].train(example["char_ids"])
    
    def predict(self, char_ids: list[int]) -> Tuple[str, float]:
        """Predict language for sequence.
        
        Args:
            char_ids: Character ID sequence
            
        Returns:
            Tuple of (predicted_language, confidence)
        """
        scores = {}
        for lang, model in self.models.items():
            scores[lang] = model.score(char_ids)
        
        best_lang = max(scores, key=scores.get)
        confidence = self._scores_to_probs(scores)
        
        return best_lang, confidence[best_lang]
    
    def predict_batch(self, batch: list[list[int]]) -> Tuple[list[str], list[float]]:
        """Predict languages for batch.
        
        Args:
            batch: List of character ID sequences
            
        Returns:
            Tuple of (languages, confidences)
        """
        predictions = []
        confidences = []
        
        for char_ids in batch:
            lang, conf = self.predict(char_ids)
            predictions.append(lang)
            confidences.append(conf)
        
        return predictions, confidences
    
    @staticmethod
    def _scores_to_probs(scores: Dict[str, float]) -> Dict[str, float]:
        """Convert scores to probabilities via softmax."""
        # Shift for numerical stability
        max_score = max(scores.values())
        exp_scores = {lang: np.exp(score - max_score) for lang, score in scores.items()}
        total = sum(exp_scores.values())
        
        return {lang: exp / total for lang, exp in exp_scores.items()}
    
    def __repr__(self) -> str:
        return f"MarkovChainLangID(languages={len(self.languages)}, models={list(self.models.values())[0]})"
