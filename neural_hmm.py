"""Neural-HMM: Neural networks for emission probabilities in HMM framework."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


logger = logging.getLogger(__name__)


class NeuralHMM(nn.Module):
    """Neural-HMM: HMM with neural emission network."""
    
    def __init__(
        self,
        vocab_size: int,
        n_states: int = 8,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        use_bilstm: bool = True,
        bilstm_hidden: int = 128
    ):
        """Initialize Neural-HMM.
        
        Args:
            vocab_size: Vocabulary size
            n_states: Number of hidden states
            embedding_dim: Character embedding dimension
            hidden_dim: MLP hidden dimension
            n_layers: Number of MLP layers
            dropout: Dropout rate
            use_bilstm: Use BiLSTM for context modeling
            bilstm_hidden: BiLSTM hidden dimension
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.n_states = n_states
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # BiLSTM for context encoding (optional)
        self.use_bilstm = use_bilstm
        if use_bilstm:
            self.bilstm = nn.LSTM(
                embedding_dim,
                bilstm_hidden // 2,
                n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0,
                bidirectional=True
            )
            emission_input_dim = bilstm_hidden
        else:
            emission_input_dim = embedding_dim
        
        # Emission network: neural network to compute p(o_t | s_t)
        self.emission_layers = nn.ModuleList()
        prev_dim = emission_input_dim
        for _ in range(n_layers - 1):
            self.emission_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.emission_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.emission_output = nn.Linear(prev_dim, vocab_size)
        
        # Transition probabilities (trainable log-space)
        self.log_A = nn.Parameter(
            torch.zeros(n_states, n_states),
            requires_grad=True
        )
        nn.init.uniform_(self.log_A, -np.log(n_states), np.log(n_states))
        
        # Initial state distribution
        self.log_pi = nn.Parameter(
            torch.ones(n_states) * (-np.log(n_states)),
            requires_grad=True
        )
    
    def forward(
        self,
        observations: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: compute log-likelihood via forward algorithm.
        
        Args:
            observations: [batch_size, seq_len] tensor of char IDs
            seq_lengths: [batch_size] tensor of sequence lengths
            
        Returns:
            Tuple of (log_likelihood, emission_logits)
        """
        batch_size, seq_len = observations.shape
        
        # Get embeddings
        embeds = self.embedding(observations)  # [B, T, D_emb]
        
        # Apply BiLSTM if enabled
        if self.use_bilstm:
            if seq_lengths is not None:
                packed = pack_padded_sequence(
                    embeds, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                lstm_out, _ = self.bilstm(packed)
                context, _ = pad_packed_sequence(lstm_out, batch_first=True)
            else:
                context, _ = self.bilstm(embeds)
        else:
            context = embeds
        
        # Emission network: compute log p(o_t | s_t) for all states
        for layer in self.emission_layers:
            context = layer(context)
            if isinstance(layer, nn.Linear):
                context = F.relu(context)
        
        emission_logits = self.emission_output(context)  # [B, T, V]
        
        # Emission probabilities: [B, T, n_states, V]
        # We need p(o_t | s_t) for each state, so we expand
        log_B = F.log_softmax(emission_logits, dim=-1)  # [B, T, V]
        
        # Compute log-likelihood via forward algorithm
        log_likelihood = self._forward_algorithm(observations, log_B, seq_lengths)
        
        return log_likelihood, emission_logits
    
    def _forward_algorithm(
        self,
        observations: torch.Tensor,
        log_B: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward algorithm in log-space.
        
        Args:
            observations: [B, T] observation indices
            log_B: [B, T, V] log emission probabilities
            seq_lengths: [B] sequence lengths
            
        Returns:
            [B] log-likelihoods
        """
        batch_size, seq_len = observations.shape
        device = observations.device
        
        # Initialize alpha (log forward probabilities)
        alpha = torch.zeros(batch_size, seq_len, self.n_states, device=device)
        
        # alpha_1(i) = log(pi_i) + log(B_i(o_1))
        o_0 = observations[:, 0]  # [B]
        log_emit_0 = log_B[:, 0, o_0]  # [B, V] -> [B] (take o_0-th element)
        # Actually we need to properly index into log_B
        log_emit_0 = torch.gather(log_B[:, 0], 1, o_0.unsqueeze(1)).squeeze(1)  # [B]
        
        # Replicate emission for each state (simplified: same for all states here)
        # In reality, we'd parameterize per-state emissions
        alpha[:, 0] = self.log_pi + log_emit_0.unsqueeze(1)
        
        # Forward recurrence
        for t in range(1, seq_len):
            o_t = observations[:, t]  # [B]
            log_emit_t = torch.gather(log_B[:, t], 1, o_t.unsqueeze(1)).squeeze(1)  # [B]
            
            # alpha_t(j) = log(sum_i alpha_{t-1}(i) * A_ij) + log(B_j(o_t))
            # Use log-sum-exp for stability
            for j in range(self.n_states):
                # Values: [B, n_states]
                vals = alpha[:, t-1] + self.log_A[:, j].unsqueeze(0)
                # Log-sum-exp
                max_vals = torch.max(vals, dim=1, keepdim=True)[0]
                log_sum = max_vals.squeeze(1) + torch.logsumexp(vals - max_vals, dim=1)
                alpha[:, t, j] = log_sum + log_emit_t
        
        # Compute final log-likelihood
        if seq_lengths is not None:
            # Use actual sequence lengths
            log_likelihood = torch.zeros(batch_size, device=device)
            for b in range(batch_size):
                t = seq_lengths[b] - 1
                max_val = torch.max(alpha[b, t])
                log_likelihood[b] = max_val + torch.logsumexp(alpha[b, t] - max_val, dim=0)
        else:
            # Use full length
            max_val = torch.max(alpha[:, -1], dim=1)[0]
            log_likelihood = max_val + torch.logsumexp(alpha[:, -1] - max_val.unsqueeze(1), dim=1)
        
        return log_likelihood
    
    def decode(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Viterbi decoding: find most likely state sequence.
        
        Args:
            observations: [batch_size, seq_len] observation indices
            
        Returns:
            Tuple of (best_paths, log_likelihood) where best_paths is [batch_size, seq_len]
        """
        batch_size, seq_len = observations.shape
        device = observations.device
        
        # Get embeddings and emissions
        embeds = self.embedding(observations)
        if self.use_bilstm:
            context, _ = self.bilstm(embeds)
        else:
            context = embeds
        
        for layer in self.emission_layers:
            if isinstance(layer, nn.Dropout):
                continue
            context = layer(context)
            if isinstance(layer, nn.Linear):
                context = F.relu(context)
        
        emission_logits = self.emission_output(context)  # [B, T, V]
        log_B = F.log_softmax(emission_logits, dim=-1)
        
        # Viterbi algorithm
        delta = torch.zeros(batch_size, seq_len, self.n_states, device=device)
        psi = torch.zeros(batch_size, seq_len, self.n_states, dtype=torch.long, device=device)
        
        # Initialization
        o_0 = observations[:, 0]
        log_emit_0 = torch.gather(log_B[:, 0], 1, o_0.unsqueeze(1)).squeeze(1)
        delta[:, 0] = self.log_pi + log_emit_0.unsqueeze(1)
        
        # Recursion
        for t in range(1, seq_len):
            o_t = observations[:, t]
            log_emit_t = torch.gather(log_B[:, t], 1, o_t.unsqueeze(1)).squeeze(1)
            
            for j in range(self.n_states):
                # Compute max over previous states
                vals = delta[:, t-1] + self.log_A[:, j].unsqueeze(0)
                delta[:, t, j], psi[:, t, j] = torch.max(vals, dim=1)
                delta[:, t, j] += log_emit_t
        
        # Backtrack
        log_likelihood, best_last = torch.max(delta[:, -1], dim=1)
        best_paths = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        best_paths[:, -1] = best_last
        
        for t in range(seq_len - 2, -1, -1):
            best_paths[:, t] = torch.gather(
                psi[:, t+1], 1,
                best_paths[:, t+1].unsqueeze(1)
            ).squeeze(1)
        
        return best_paths, log_likelihood


class NeuralHMMLangID:
    """Multilingual language identifier using Neural-HMMs."""
    
    def __init__(
        self,
        languages: list[str],
        vocab_size: int,
        n_states: int = 8,
        embedding_dim: int = 64,
        device: str = "cpu"
    ):
        """Initialize multilingual Neural-HMM.
        
        Args:
            languages: Language codes
            vocab_size: Vocabulary size
            n_states: Number of hidden states
            embedding_dim: Character embedding dimension
            device: Device to use ("cpu" or "cuda")
        """
        self.languages = languages
        self.vocab_size = vocab_size
        self.device = torch.device(device)
        
        # One Neural-HMM per language
        self.models = {
            lang: NeuralHMM(vocab_size, n_states, embedding_dim).to(self.device)
            for lang in languages
        }
        
    def train(
        self,
        examples: list[dict],
        language_key: str = "language",
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 1e-3
    ) -> None:
        """Train per-language Neural-HMM models.
        
        Args:
            examples: List of training examples
            language_key: Key containing language label
            epochs: Number of training epochs
            batch_size: Training batch size
            lr: Learning rate
        """
        from torch.utils.data import DataLoader
        
        def collate_fn(batch):
            seqs = [torch.tensor(ex["char_ids"], dtype=torch.long) for ex in batch]
            lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
            padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
            return padded, lengths
            
        # Group examples by language
        by_lang = {}
        for example in examples:
            lang = example[language_key]
            if lang not in by_lang:
                by_lang[lang] = []
            by_lang[lang].append(example)
            
        for lang, lang_examples in by_lang.items():
            if lang not in self.models:
                continue
                
            logger.info(f"Training Neural-HMM for {lang} ({len(lang_examples)} examples)")
            model = self.models[lang]
            model.train()
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            dataloader = DataLoader(
                lang_examples,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )
            
            for epoch in range(epochs):
                for padded, lengths in dataloader:
                    padded = padded.to(self.device)
                    lengths = lengths.to(self.device)
                    
                    optimizer.zero_grad()
                    log_likelihood, _ = model(padded, lengths)
                    loss = -log_likelihood.mean()
                    
                    loss.backward()
                    optimizer.step()
    
    def predict(self, char_ids: list[int]) -> Tuple[str, float]:
        """Predict language for sequence.
        
        Args:
            char_ids: Character ID sequence
            
        Returns:
            Tuple of (language, confidence)
        """
        with torch.no_grad():
            observations = torch.tensor([char_ids], dtype=torch.long, device=self.device)
            scores = {}
            
            for lang, model in self.models.items():
                model.eval()
                log_likelihood, _ = model(observations)
                # Normalize by sequence length
                scores[lang] = (log_likelihood[0] / max(len(char_ids), 1)).item()
            
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
        max_score = max(scores.values())
        exp_scores = {lang: np.exp(score - max_score) for lang, score in scores.items()}
        total = sum(exp_scores.values())
        return {lang: exp / total for lang, exp in exp_scores.items()}
