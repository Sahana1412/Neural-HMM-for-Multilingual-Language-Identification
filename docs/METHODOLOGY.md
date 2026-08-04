# Methodology: Character-Level Language Identification Models

## Overview

This document provides comprehensive mathematical and technical details for the three language identification approaches implemented in this project.

## 1. Character Markov Chain (MC)

### Mathematical Formulation

An n-gram Markov Chain language model estimates the probability of a text sequence as:

$$P(\mathbf{c}) = P(c_1) \prod_{t=2}^{T} P(c_t | c_{t-n+1}, \ldots, c_{t-1})$$

where:
- $\mathbf{c} = (c_1, \ldots, c_T)$ is a sequence of characters
- $n$ is the n-gram order (1 ≤ n ≤ 4)
- Each character probability is conditioned on the preceding n-1 characters

### Maximum Likelihood Estimation

The MLE of n-gram probabilities is:

$$P(c_t | \text{context}) = \frac{\text{count}(c_t, \text{context})}{\text{count}(\text{context})}$$

This is implemented through counting observed n-grams in training data.

### Smoothing Techniques

#### Laplace Smoothing (Add-k)

Addresses zero-probability problem by adding pseudo-counts:

$$P(c_t | \text{context}) = \frac{\text{count}(c_t, \text{context}) + \alpha}{|\text{context}| + \alpha \cdot |V|}$$

where:
- $\alpha$ is the smoothing parameter (typically 1.0)
- $|V|$ is the vocabulary size

#### Kneser-Ney Smoothing

Higher-order smoothing method using continuation probabilities:

$$P_{\text{KN}}(c | \text{context}) = \frac{\max(\text{count}(c, \text{context}) - D, 0)}{|\text{context}|} + \lambda(\text{context}) \cdot P_{\text{KN}}(c | \text{shorter context})$$

Advantages:
- Better handling of unseen n-grams
- Especially effective for rare contexts
- Improved generalization

### Language Identification

For multilingual identification, maintain separate models $\lambda_1, \ldots, \lambda_k$ for each language.

Predict language $\hat{l}$ as:

$$\hat{l} = \arg\max_j \log P(\mathbf{c} | \lambda_j)$$

Scoring uses normalized log-likelihood:

$$\text{score}(\mathbf{c}, \lambda_j) = \frac{1}{T} \sum_{t=2}^{T} \log P(c_t | c_{t-n+1}, \ldots, c_{t-1})$$

### Computational Complexity

- Training: $O(\sum_j |D_j|)$ where $|D_j|$ is language $j$'s training size
- Prediction: $O(T \cdot k)$ where $T$ is sequence length, $k$ is number of languages
- Memory: $O(k \cdot V^n)$ for storing n-gram counts

## 2. Hidden Markov Model (HMM)

### Mathematical Formulation

A Hidden Markov Model consists of:

**Hidden States**: Latent states $Q = \{q_1, \ldots, q_N\}$ modeling sequential structure

**Observation Model**: 
- Transition probabilities: $A = [a_{ij}]$ where $a_{ij} = P(q_t = j | q_{t-1} = i)$
- Emission probabilities: $B = [b_j(o)]$ where $b_j(o) = P(o_t = o | q_t = j)$
- Initial state distribution: $\pi = [\pi_i]$ where $\pi_i = P(q_1 = i)$

**Joint Likelihood**:

$$P(\mathbf{o}, \mathbf{q} | \lambda) = \pi_{q_1} \prod_{t=1}^{T} b_{q_t}(o_t) \prod_{t=2}^{T} a_{q_{t-1}, q_t}$$

### Forward Algorithm (Inference)

Computes likelihood efficiently using dynamic programming:

$$\alpha_t(i) = P(o_1, \ldots, o_t, q_t = i | \lambda)$$

**Recurrence**:
- Base: $\alpha_1(i) = \pi_i \cdot b_i(o_1)$
- Recurrence: $\alpha_t(j) = b_j(o_t) \sum_i \alpha_{t-1}(i) \cdot a_{ij}$

**Likelihood**: $P(\mathbf{o} | \lambda) = \sum_i \alpha_T(i)$

Time complexity: $O(T \cdot N^2)$

### Backward Algorithm (Training)

Computes backward probabilities for EM algorithm:

$$\beta_t(i) = P(o_{t+1}, \ldots, o_T | q_t = i, \lambda)$$

**Recurrence** (backward in time):
- Base: $\beta_T(i) = 1$
- Recurrence: $\beta_t(i) = \sum_j a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j)$

### Viterbi Algorithm (Decoding)

Finds most likely state sequence:

$$\hat{\mathbf{q}} = \arg\max_\mathbf{q} P(\mathbf{o}, \mathbf{q} | \lambda)$$

**Recurrence**:
$$\delta_t(j) = \max_i \delta_{t-1}(i) \cdot a_{ij} \cdot b_j(o_t)$$

**Backtracking**: Store argmax indices for reconstruction

Time complexity: $O(T \cdot N^2)$

### Baum-Welch Training (EM Algorithm)

Unsupervised parameter estimation maximizing $P(\mathbf{o} | \lambda)$.

**Posterior State Probabilities** (E-step):

$$\gamma_t(i) = P(q_t = i | \mathbf{o}, \lambda) = \frac{\alpha_t(i) \cdot \beta_t(i)}{P(\mathbf{o}|\lambda)}$$

**Pairwise Posterior** (for transitions):

$$\xi_t(i,j) = P(q_t = i, q_{t+1} = j | \mathbf{o}, \lambda) = \frac{\alpha_t(i) \cdot a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j)}{P(\mathbf{o}|\lambda)}$$

**Parameter Re-estimation** (M-step):

$$\hat{\pi}_i = \gamma_1(i)$$

$$\hat{a}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$$

$$\hat{b}_j(o) = \frac{\sum_{t: o_t = o} \gamma_t(j)}{\sum_t \gamma_t(j)}$$

**Convergence**: Iterate until log-likelihood stabilizes

### Numerical Stability

Use **log-space computations** to avoid underflow:

- Store log-probabilities: $\log A, \log B, \log \pi$
- Log-sum-exp trick: $\log(\sum_i e^{x_i}) = x_{\max} + \log(\sum_i e^{x_i - x_{\max}})$

## 3. Neural Hidden Markov Model (Neural-HMM)

### Architecture

Combines HMM structure with neural networks for emission probabilities.

**Components**:

1. **Character Embeddings**: 
   - Embedding matrix: $\mathbf{E} \in \mathbb{R}^{V \times d_e}$
   - Embedding: $\mathbf{e}_t = \mathbf{E}[c_t]$

2. **Context Encoder** (optional BiLSTM):
   - Bidirectional LSTM processes context
   - Output: contextual representations $\mathbf{h}_t \in \mathbb{R}^{d_h}$

3. **Emission Network**:
   - MLP mapping state and observation to probabilities
   - Input: concatenation of embedding and state encoding
   - Output: logits for vocabulary

4. **Trainable Transition Matrix**:
   - Transition parameters: $A \in \mathbb{R}^{N \times N}$ (trainable)
   - Initialized uniformly
   - Constrained to valid probabilities via softmax during inference

### Emission Probability Parameterization

Emission probabilities computed via neural network:

$$p(o_t | s_t, \theta) = \text{softmax}(f_\theta(\mathbf{e}_t, s_t))$$

where $f_\theta$ is the emission network parameterized by $\theta$.

**Advantages**:
- Captures complex character distribution patterns
- Shares information across states
- Scales better to large vocabularies

### Forward Algorithm with Neural Emissions

Modified forward algorithm using learned emissions:

$$\alpha_t(j) = \left(\sum_i \alpha_{t-1}(i) \cdot a_{ij}\right) \cdot p_\theta(o_t | s_t = j)$$

Computed in log-space:

$$\log \alpha_t(j) = \log\left(\sum_i e^{\log \alpha_{t-1}(i) + \log a_{ij}}\right) + \log p_\theta(o_t | s_t = j)$$

### Training

**Objective**: Negative Log-Likelihood via forward algorithm

$$\mathcal{L}(\theta) = -\log P(\mathbf{o} | \lambda)$$

**Optimization**:
- Algorithm: Adam with weight decay
- Learning rate schedule: Cosine annealing with warmup
- Gradient clipping: Norm-based clipping for stability
- Early stopping: Monitor validation loss

**Regularization**:
- Dropout in LSTM and MLP layers
- $L_2$ weight decay
- Gradient clipping

### Implementation Details

**Batch Processing**:
- Pack padded sequences for efficient BiLSTM computation
- Variable-length sequence handling

**Numerical Stability**:
- Log-space forward algorithm
- Log-softmax for emission logits
- Careful numerical handling of transitions

**GPU Acceleration**:
- PyTorch GPU support
- Mixed precision training (FP16/FP32)
- Batched matrix operations

## 4. Experimental Setup

### Dataset: WiLI-2018

**Characteristics**:
- 14 European languages: Catalan, Danish, German, English, Spanish, Finnish, French, Icelandic, Italian, Dutch, Norwegian, Portuguese, Romanian, Swedish
- 1000 examples per language
- Total: 14,000 sequences
- Average length: ~180-200 characters

**Splits**:
- Training: 70% (9,800 examples)
- Validation: 15% (2,100 examples)
- Test (CleanEval): 15% (2,100 examples)
- Test (NoisyEval): Generated from test set with noise

### Vocabulary

**84-character vocabulary** including:
- Lowercase English letters (26)
- Digits (10)
- Punctuation and whitespace (15)
- Accented characters (21)
- Unknown token

### Noise Injection (NoisyEval)

Applied independently with configurable probabilities:

1. **Accent Stripping**: Remove diacritics from accented characters
2. **Random Deletion**: Remove random characters at rate $r_d$
3. **Random Insertion**: Insert random characters at rate $r_i$
4. **Character Substitution**: Replace with keyboard neighbor at rate $r_s$
5. **Truncation**: Keep fraction $f$ of sequence at rate $r_t$
6. **Lowercasing**: Convert to lowercase at rate $r_l$

### Evaluation Metrics

**Classification Accuracy**:
$$\text{Accuracy} = \frac{\text{# correct predictions}}{\text{total predictions}}$$

**Per-Language Metrics**:
- Precision: $P_i = \frac{TP_i}{TP_i + FP_i}$
- Recall: $R_i = \frac{TP_i}{TP_i + FN_i}$
- F1-Score: $F1_i = 2 \cdot \frac{P_i \cdot R_i}{P_i + R_i}$

**Macro-Averaged** (unweighted average across languages)
**Weighted-Averaged** (weighted by class frequency)

**Calibration**:
- Expected Calibration Error (ECE): Measures agreement between predicted confidence and actual accuracy
- Brier Score: Mean squared error of predicted probabilities

### Ablation Studies

1. **Markov Order**: Test $n \in \{1, 2, 3, 4\}$
2. **HMM States**: Test $N \in \{4, 8, 12, 16\}$
3. **Neural-HMM Embedding Dimension**: Test $d_e \in \{16, 32, 64, 128\}$
4. **Noise Severity**: Progressive noise injection levels

### Reproducibility

- Fixed random seeds across all experiments
- Deterministic CUDA operations
- Saved configs and checkpoints
- Multiple runs (≥3) with different seeds
- Results reported as mean ± standard deviation

## References

- Rabiner, L. R. (1989). "A tutorial on hidden Markov models and selected applications in speech recognition." Proceedings of the IEEE, 77(2), 257-286.

- Kneser, R., & Ney, H. (1995). "Improved backing-off for m-gram language modeling." ICASSP-95, 1, 181-184.

- Wiseman, S., Stratos, K., & Rush, A. M. (2017). "Sequence to sequence learning as beam-search optimization." EMNLP 2017.

- Thawani, A., Upadhyay, B., Srivastava, A., & Singh, A. (2018). "WiLI: An open collection for language identification." arXiv preprint arXiv:1801.07779.
