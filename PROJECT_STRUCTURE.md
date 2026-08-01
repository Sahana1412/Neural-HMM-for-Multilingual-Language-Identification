# Project Structure Documentation

## Directory Hierarchy

```
neural-hmm-lang-id/
├── README.md                          # Main project documentation
├── QUICKSTART.md                      # 5-minute quick start guide
├── CONTRIBUTING.md                    # Contribution guidelines
├── PROJECT_STRUCTURE.md               # This file
├── LICENSE                            # MIT License
├── Makefile                           # Automation commands
├── pyproject.toml                     # Modern Python project config
├── setup.py                           # Package installation
├── requirements.txt                   # Dependencies (pip format)
├── .gitignore                         # Git ignore rules
│
├── config/                            # Configuration files
│   ├── default.yaml                  # Default hyperparameters
│   ├── markov_chain.yaml             # Markov Chain config
│   ├── hmm.yaml                      # HMM config
│   └── neural_hmm.yaml               # Neural-HMM config
│
├── data/                              # Dataset storage
│   └── .gitkeep                      # Placeholder for data
│
├── src/                               # Source code
│   ├── __init__.py                   # Package marker
│   │
│   ├── models/                       # Model implementations
│   │   ├── __init__.py
│   │   ├── markov_chain.py          # Markov Chain (n-gram LM)
│   │   ├── hmm.py                   # HMM (Baum-Welch, Viterbi)
│   │   └── neural_hmm.py            # Neural-HMM (PyTorch)
│   │
│   ├── preprocessing/                # Data processing pipeline
│   │   ├── __init__.py
│   │   ├── dataset.py               # WiLI-2018 loader
│   │   ├── vocab.py                 # Character vocabulary
│   │   ├── normalizer.py            # Unicode normalization
│   │   └── noise_generator.py       # Noise injection for robustness
│   │
│   ├── evaluation/                   # Evaluation metrics
│   │   ├── __init__.py
│   │   └── metrics.py               # Classification & calibration metrics
│   │
│   └── utils/                        # Utility modules
│       ├── __init__.py
│       ├── config.py                # YAML config loader
│       ├── logger.py                # Logging setup
│       ├── reproducibility.py       # Seed management
│       └── plotting.py              # Visualization functions
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_preprocessing.ipynb       # Preprocessing walkthrough
│   ├── 03_model_comparison.ipynb    # Results visualization
│   └── 04_embedding_analysis.ipynb  # Neural-HMM embeddings
│
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py        # Tests for preprocessing
│   ├── test_models.py               # Tests for models
│   └── test_metrics.py              # Tests for metrics
│
├── outputs/                          # Generated outputs
│   ├── checkpoints/                 # Saved model checkpoints
│   ├── metrics/                     # Computed metrics (JSON/CSV)
│   ├── plots/                       # Generated visualizations
│   └── runs/                        # TensorBoard logs
│
├── docs/                             # Additional documentation
│   ├── METHODOLOGY.md               # Mathematical details
│   ├── IMPLEMENTATION_NOTES.md      # Technical implementation notes
│   └── EXAMPLES.md                  # Usage examples
│
└── main.py                           # CLI entry point
```

## File Descriptions

### Core Project Files

| File | Purpose |
|------|---------|
| `main.py` | Command-line interface for all operations |
| `setup.py` | Package installation configuration |
| `pyproject.toml` | Modern Python project metadata |
| `requirements.txt` | Pip-format dependencies |
| `Makefile` | Build automation commands |

### Configuration Files (`config/`)

| File | Purpose |
|------|---------|
| `default.yaml` | Global settings (seed, device, dataset, paths) |
| `markov_chain.yaml` | Markov Chain hyperparameters |
| `hmm.yaml` | HMM hyperparameters |
| `neural_hmm.yaml` | Neural-HMM hyperparameters |

### Source Code (`src/`)

#### Models (`src/models/`)
- **markov_chain.py**: Character n-gram language models with Laplace/Kneser-Ney smoothing
- **hmm.py**: Hidden Markov Models with Baum-Welch, Forward, Backward, Viterbi
- **neural_hmm.py**: Neural-HMM with PyTorch, learned embeddings, neural emissions

#### Preprocessing (`src/preprocessing/`)
- **dataset.py**: WiLI-2018 dataset loader and preprocessor
- **vocab.py**: 84-character vocabulary with encoding/decoding
- **normalizer.py**: Unicode normalization with accent handling
- **noise_generator.py**: Noise injection (accent stripping, deletion, insertion, etc.)

#### Evaluation (`src/evaluation/`)
- **metrics.py**: Accuracy, precision, recall, F1, confusion matrices, calibration (ECE, Brier)

#### Utils (`src/utils/`)
- **config.py**: YAML config loading with inheritance
- **logger.py**: Logging setup with file/console handlers
- **reproducibility.py**: Seed management for NumPy/PyTorch/Python
- **plotting.py**: Publication-quality visualization functions

### Tests (`tests/`)
- **test_preprocessing.py**: Vocabulary, normalization, noise generator tests
- **test_models.py**: Markov Chain, HMM implementation tests
- **test_metrics.py**: Metrics computation verification

### Documentation

| File | Content |
|------|---------|
| `README.md` | Comprehensive project overview, setup, usage |
| `QUICKSTART.md` | 5-minute setup and quick examples |
| `CONTRIBUTING.md` | Guidelines for contributing to project |
| `docs/METHODOLOGY.md` | Mathematical foundations of all three models |
| `PROJECT_STRUCTURE.md` | This file |

## Key Features

### 1. Modular Architecture
- Separate concerns: models, preprocessing, evaluation, utils
- Easy to extend with new models or preprocessing steps
- Clear interfaces between components

### 2. Configuration-Driven
- YAML configs for all experiments
- Override defaults in subconfigs
- Reproducible experiments with fixed seeds

### 3. Comprehensive Testing
- Unit tests for preprocessing and models
- Fixtures for common test data
- Coverage reporting with pytest-cov

### 4. Production Quality
- Type hints throughout
- Docstrings (Google style)
- Black/Ruff formatting
- Logging at all levels

### 5. Reproducibility
- Fixed seeds for all randomness
- Saved configs and checkpoints
- Multi-seed experiments (mean ± std)
- Deterministic CUDA operations

### 6. Complete Documentation
- README with setup and usage
- QUICKSTART for 5-minute intro
- METHODOLOGY for mathematical details
- CONTRIBUTING for development
- Comprehensive docstrings

## Data Flow

```
Raw WiLI-2018
    ↓
[dataset.py] → Download & Extract
    ↓
[normalizer.py] → Unicode Normalization
    ↓
[vocab.py] → Character Encoding
    ↓
[noise_generator.py] → Create NoisyEval
    ↓
JSONL Splits (train, validation, clean_eval, noisy_eval)
    ↓
[markov_chain.py] → Train Model 1
[hmm.py] → Train Model 2
[neural_hmm.py] → Train Model 3
    ↓
[metrics.py] → Evaluate & Compute Metrics
    ↓
[plotting.py] → Generate Visualizations
    ↓
outputs/ (checkpoints, metrics, plots)
```

## Usage Patterns

### Command Line
```bash
# Preprocess
python main.py preprocess --config config/default.yaml

# Train
python main.py train markov --config config/markov_chain.yaml

# Evaluate
python main.py evaluate-all --config config/default.yaml

# Full pipeline
make full-pipeline
```

### Python API
```python
from src.models.markov_chain import MarkovChainLangID
from src.preprocessing.vocab import Vocabulary

vocab = Vocabulary.load("vocab.json")
model = MarkovChainLangID(languages=["en", "es"], n_gram_order=2)
model.train(training_data)
pred_lang, confidence = model.predict(char_ids)
```

### Configuration
```python
from src.utils.config import load_config

config = load_config("config/default.yaml")
print(config.dataset.languages)  # Attribute access
print(config["dataset"]["languages"])  # Dict access
```

## Dependencies

### Core
- numpy: Numerical computing
- torch: Deep learning framework
- scipy: Scientific computing
- scikit-learn: Machine learning metrics

### Utilities
- pyyaml: Config file parsing
- matplotlib/seaborn: Visualization
- tqdm: Progress bars

### Development (Optional)
- pytest: Unit testing
- black/ruff: Code formatting/linting
- mypy: Type checking
- jupyter: Interactive notebooks

## Extension Points

### Add a New Model
1. Create `src/models/new_model.py`
2. Implement `train()` and `predict_batch()` methods
3. Add config section in `config/new_model.yaml`
4. Update `main.py` with training logic

### Add New Metrics
1. Add computation to `src/evaluation/metrics.py`
2. Update `LanguageIDMetrics.compute_all()`
3. Add visualization in `src/utils/plotting.py`

### Add New Noise Types
1. Add method to `NoiseGenerator` class
2. Add configuration option to `noise_config`
3. Update documentation with example

## Best Practices

### Running Experiments
1. **Always use seeds**: `set_seed()` in reproducibility module
2. **Save configs**: Save used config with results
3. **Multiple runs**: Run with 3+ seeds, report mean ± std
4. **Document changes**: Update config file comments

### Code Quality
1. **Type hints**: Use for all function signatures
2. **Docstrings**: Google-style for public functions
3. **Tests**: Write tests for new code
4. **Linting**: Run `make lint` before commits

### Performance
1. **Batch processing**: Use batch_predict() for efficiency
2. **GPU**: Use CUDA for neural-hmm training
3. **Caching**: Vocabularies and datasets loaded once
4. **Profiling**: Use cProfile for bottleneck identification

## Troubleshooting

### Import Errors
```bash
# Ensure package installed in editable mode
pip install -e .
```

### Missing Data
```bash
# Manually download and extract
wget https://zenodo.org/record/841984/files/wili-2018.zip
unzip -d data/
python main.py preprocess --config config/default.yaml
```

### GPU Issues
```bash
# Check PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU if GPU unavailable
python main.py train neural-hmm --device cpu
```

---

For detailed information, see README.md and docs/METHODOLOGY.md
