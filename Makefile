.PHONY: help install install-dev test lint format clean preprocess train-all train-markov train-hmm train-neural-hmm evaluate-all plots full-pipeline

PYTHON := python3.11
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest

help:
	@echo "Neural-HMM Language Identification - Build Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install              - Install package in production mode"
	@echo "  make install-dev          - Install package with dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test                 - Run all tests"
	@echo "  make test-fast            - Run fast tests only"
	@echo "  make lint                 - Run linting (ruff, black, mypy)"
	@echo "  make format               - Format code with black and isort"
	@echo ""
	@echo "Data & Training:"
	@echo "  make preprocess           - Download and preprocess WiLI-2018"
	@echo "  make train-all            - Train all three models"
	@echo "  make train-markov         - Train Markov Chain model"
	@echo "  make train-hmm            - Train HMM model"
	@echo "  make train-neural-hmm     - Train Neural-HMM model"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate-all         - Evaluate all models"
	@echo "  make plots                - Generate all visualization plots"
	@echo ""
	@echo "Full Pipeline:"
	@echo "  make full-pipeline        - Run complete workflow (preprocess → train → evaluate → plot)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                - Remove generated files and caches"

# Setup targets
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,notebooks,tracking]"

# Testing targets
test:
	$(PYTEST) tests/ -v --cov=src

test-fast:
	$(PYTEST) tests/ -v --cov=src -m "not slow"

# Linting targets
lint:
	$(PYTHON) -m black --check src/ tests/
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m mypy src/ --ignore-missing-imports || true

format:
	$(PYTHON) -m isort src/ tests/
	$(PYTHON) -m black src/ tests/

# Data pipeline
preprocess:
	$(PYTHON) main.py preprocess --config config/default.yaml

# Training targets
train-all: train-markov train-hmm train-neural-hmm

train-markov:
	$(PYTHON) main.py train markov --config config/markov_chain.yaml

train-hmm:
	$(PYTHON) main.py train hmm --config config/hmm.yaml

train-neural-hmm:
	$(PYTHON) main.py train neural-hmm --config config/neural_hmm.yaml

# Evaluation targets
evaluate-all:
	$(PYTHON) main.py evaluate-all --config config/default.yaml

# Visualization targets
plots:
	$(PYTHON) src/utils/plotting.py --config config/default.yaml

# Full pipeline
full-pipeline: preprocess train-all evaluate-all plots
	@echo "✅ Full pipeline completed!"

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	rm -rf build/ dist/
	rm -rf outputs/checkpoints/* outputs/plots/* outputs/metrics/*

.DEFAULT_GOAL := help
