"""Optimized Hidden Markov Model with Baum-Welch training and Viterbi decoding."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
# from numba import jit, prange


logger = logging.getLogger(__name__)


# Numba-compiled functions for critical loops
def forward_log_space_numba(
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
    observations: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Numba-optimized forward algorithm in log-space."""
    T = len(observations)
    n_states = log_A.shape[0]
    alpha = np.zeros((T, n_states))
    
    # Initial step
    alpha[0] = log_pi + log_B[:, observations[0]]
    
    # Forward pass
    for t in range(1, T):
        for j in range(n_states):
            vals = alpha[t-1] + log_A[:, j]
            max_val = np.max(vals)
            log_sum = max_val + np.log(np.sum(np.exp(vals - max_val)))
            alpha[t, j] = log_sum + log_B[j, observations[t]]
    
    # Compute likelihood
    max_val = np.max(alpha[T-1])
    log_likelihood = max_val + np.log(np.sum(np.exp(alpha[T-1] - max_val)))
    
    return alpha, log_likelihood


def backward_log_space_numba(
    log_A: np.ndarray,
    log_B: np.ndarray,
    observations: np.ndarray
) -> np.ndarray:
    """Numba-optimized backward algorithm in log-space."""
    T = len(observations)
    n_states = log_A.shape[0]
    beta = np.zeros((T, n_states))
    
    # Last step
    beta[T-1] = 0.0
    
    # Backward pass
    for t in range(T-2, -1, -1):
        for i in range(n_states):
            vals = log_A[i, :] + log_B[:, observations[t+1]] + beta[t+1]
            max_val = np.max(vals)
            beta[t, i] = max_val + np.log(np.sum(np.exp(vals - max_val)))
    
    return beta


def viterbi_log_space_numba(
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
    observations: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Numba-optimized Viterbi algorithm in log-space."""
    T = len(observations)
    n_states = log_A.shape[0]
    delta = np.zeros((T, n_states))
    psi = np.zeros((T, n_states), dtype=np.int32)
    
    # Initial step
    delta[0] = log_pi + log_B[:, observations[0]]
    
    # Forward pass with max tracking
    for t in range(1, T):
        for j in range(n_states):
            temp = delta[t-1] + log_A[:, j]
            psi[t, j] = np.argmax(temp)
            delta[t, j] = np.max(temp) + log_B[j, observations[t]]
    
    # Backtrack
    log_likelihood = np.max(delta[T-1])
    
    return delta, psi, log_likelihood

def compute_gamma(
    alpha: np.ndarray,
    beta: np.ndarray
) -> np.ndarray:
    """Compute posterior state probabilities (gamma) with numerical stability."""
    T = alpha.shape[0]
    n_states = alpha.shape[1]
    gamma_log = alpha + beta
    
    # Normalize in log space
    gamma_log_shifted = gamma_log - np.max(gamma_log, axis=1, keepdims=True)
    gamma = np.exp(gamma_log_shifted)
    gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
    
    return gamma

def update_emission_fast(
    gamma: np.ndarray,
    observations: np.ndarray,
    n_symbols: int,
    smoothing: float
) -> np.ndarray:
    """Vectorized emission matrix update."""
    n_states = gamma.shape[1]
    log_B = np.zeros((n_states, n_symbols))
    
    for j in range(n_states):
        for k in range(n_symbols):
            # Find all positions where symbol k appears
            count = 0.0
            total = 0.0
            
            for t in range(len(observations)):
                if observations[t] == k:
                    count += gamma[t, j]
                total += gamma[t, j]
            
            numer = count + smoothing
            denom = total + smoothing * n_symbols
            log_B[j, k] = np.log(max(numer / denom, 1e-10))
    
    return log_B


class HiddenMarkovModel:
    """Optimized HMM for sequence modeling with n-gram language identification."""
    
    def __init__(
        self,
        n_states: int = 8,
        n_symbols: int = 84,
        smoothing: float = 1.0,
        use_log_space: bool = True,
        batch_size: int = 1000,
        cache_emissions: bool = True
    ):
        """Initialize HMM.
        
        Args:
            n_states: Number of hidden states
            n_symbols: Number of observable symbols (vocabulary size)
            smoothing: Laplace smoothing parameter
            use_log_space: Use log-space for numerical stability
            batch_size: Batch size for processing long sequences
            cache_emissions: Cache emission lookups (memory tradeoff)
        """
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.smoothing = smoothing
        self.use_log_space = use_log_space
        self.batch_size = batch_size
        self.cache_emissions = cache_emissions
        
        # Model parameters (always in log-space)
        self.log_pi = np.zeros(n_states)
        self.log_A = np.zeros((n_states, n_states))
        self.log_B = np.zeros((n_states, n_symbols))
        
        # Initialize uniformly
        self._initialize_uniform()
    
    def _initialize_uniform(self) -> None:
        """Initialize parameters uniformly."""
        self.log_pi = np.log(np.ones(self.n_states) / self.n_states)
        self.log_A = np.log(np.ones((self.n_states, self.n_states)) / self.n_states)
        self.log_B = np.log(np.ones((self.n_states, self.n_symbols)) / self.n_symbols)
    
    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward algorithm using Numba-optimized implementation."""
        observations = np.asarray(observations, dtype=np.int32)
        return forward_log_space_numba(self.log_pi, self.log_A, self.log_B, observations)
    
    def backward(self, observations: np.ndarray) -> np.ndarray:
        """Backward algorithm using Numba-optimized implementation."""
        observations = np.asarray(observations, dtype=np.int32)
        return backward_log_space_numba(self.log_A, self.log_B, observations)
    
    def viterbi(self, observations: np.ndarray) -> Tuple[list, float]:
        """Viterbi algorithm using Numba-optimized implementation."""
        observations = np.asarray(observations, dtype=np.int32)
        delta, psi, log_likelihood = viterbi_log_space_numba(
            self.log_pi, self.log_A, self.log_B, observations
        )
        
        # Backtrack to find path
        T = len(observations)
        path = [int(np.argmax(delta[T-1]))]
        for t in range(T-1, 0, -1):
            path.append(int(psi[t, path[-1]]))
        path.reverse()
        
        return path, log_likelihood
    
    def baum_welch(
        self,
        observations: np.ndarray,
        n_iterations: int = 5,
        convergence_threshold: float = 1e-4
    ) -> float:
        """Baum-Welch EM algorithm for training.
        
        Args:
            observations: Sequence of observation indices [T]
            n_iterations: Maximum number of iterations
            convergence_threshold: Convergence threshold
            
        Returns:
            Final log-likelihood
        """
        observations = np.asarray(observations, dtype=np.int32)
        T = len(observations)
        prev_likelihood = -np.inf
        
        for iteration in range(n_iterations):
            # E-step: compute forward and backward probabilities
            alpha, likelihood = self.forward(observations)
            beta = self.backward(observations)
            
            logger.debug(f"Iteration {iteration}: likelihood={likelihood:.4f}")
            
            # Check convergence
            if abs(likelihood - prev_likelihood) < convergence_threshold:
                logger.info(f"Converged at iteration {iteration}")
                break
            
            prev_likelihood = likelihood
            
            # M-step: update parameters
            self._update_parameters_fast(observations, alpha, beta)
        
        return likelihood
    
    def _update_parameters_fast(
        self,
        observations: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray
    ) -> None:
        """Fast parameter update using Numba."""
        # Compute gamma
        gamma = compute_gamma(alpha, beta)
        
        # Update pi
        self.log_pi = np.log(np.maximum(gamma[0], 1e-10))
        
        # Update A (transition matrix)
        # Using aggregated statistics instead of per-pair computation
        xi_sum = np.zeros((self.n_states, self.n_states))
        for t in range(len(observations) - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi_sum[i, j] += (
                        gamma[t, i] * 
                        self.log_A[i, j] * 
                        self.log_B[j, observations[t+1]] * 
                        gamma[t+1, j]
                    )
        
        # Normalize and add smoothing
        for i in range(self.n_states):
            row_sum = np.sum(xi_sum[i]) + self.smoothing * self.n_states
            for j in range(self.n_states):
                self.log_A[i, j] = np.log(
                    np.maximum((xi_sum[i, j] + self.smoothing) / row_sum, 1e-10)
                )
        
        # Update B (emission matrix) - vectorized
        self.log_B = update_emission_fast(
            gamma, observations, self.n_symbols, self.smoothing
        )
    
    def score(self, observations: np.ndarray) -> float:
        """Score sequence by normalized log-likelihood."""
        observations = np.asarray(observations, dtype=np.int32)
        _, likelihood = self.forward(observations)
        return likelihood / max(len(observations), 1)


class HiddenMarkovModelLangID:
    """Optimized multilingual language identifier using HMMs."""
    
    def __init__(
        self,
        languages: list,
        n_states: int = 8,
        n_symbols: int = 84,
        smoothing: float = 1.0,
        n_jobs: int = -1
    ):
        """Initialize multilingual HMM.
        
        Args:
            languages: List of language codes
            n_states: Number of hidden states per model
            n_symbols: Vocabulary size
            smoothing: Laplace smoothing parameter
            n_jobs: Number of parallel jobs (-1 = all cores)
        """
        self.languages = languages
        self.n_jobs = n_jobs
        self.models = {
            lang: HiddenMarkovModel(n_states, n_symbols, smoothing)
            for lang in languages
        }
    
    def train(
        self,
        examples: list,
        language_key: str = "language",
        n_iterations: int = 5
    ) -> None:
        """Train per-language HMM models.
        
        Args:
            examples: List of training examples
            language_key: Key containing language label
            n_iterations: Number of EM iterations
        """
        # Group examples by language
        by_lang = {}
        for example in examples:
            lang = example[language_key]
            if lang not in by_lang:
                by_lang[lang] = []
            by_lang[lang].append(example)
        
        # Train each model
        for lang, lang_examples in by_lang.items():
            if lang not in self.models:
                continue
            
            logger.info(f"Training HMM for {lang} ({len(lang_examples)} examples)")
            
            # Concatenate all sequences for this language
            all_chars = []
            for example in lang_examples:
                all_chars.extend(example["char_ids"])
            
            # Train on concatenated sequence
            observations = np.array(all_chars, dtype=np.int32)
            self.models[lang].baum_welch(observations, n_iterations=n_iterations)
    
    def predict(self, char_ids: list) -> Tuple[str, float]:
        """Predict language for sequence."""
        observations = np.array(char_ids, dtype=np.int32)
        scores = {}
        
        for lang, model in self.models.items():
            scores[lang] = model.score(observations)
        
        best_lang = max(scores, key=scores.get)
        confidence = self._scores_to_probs(scores)
        
        return best_lang, confidence[best_lang]
    
    def predict_batch(self, batch: list) -> Tuple[list, list]:
        """Predict languages for batch with minimal overhead."""
        predictions = []
        confidences = []
        
        for char_ids in batch:
            lang, conf = self.predict(char_ids)
            predictions.append(lang)
            confidences.append(conf)
        
        return predictions, confidences
    
    @staticmethod
    def _scores_to_probs(scores: Dict) -> Dict:
        """Convert scores to probabilities via softmax."""
        max_score = max(scores.values())
        exp_scores = {lang: np.exp(score - max_score) for lang, score in scores.items()}
        total = sum(exp_scores.values())
        return {lang: exp / total for lang, exp in exp_scores.items()}
    
    def __repr__(self) -> str:
        model_sample = list(self.models.values())[0]
        return f"HiddenMarkovModelLangID(languages={len(self.languages)}, n_states={model_sample.n_states})"