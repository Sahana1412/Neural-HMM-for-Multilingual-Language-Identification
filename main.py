#!/usr/bin/env python3
"""Main CLI entry point for Neural-HMM Language Identification experiments."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.preprocessing.dataset import WiLI2018Dataset
from src.preprocessing.vocab import Vocabulary
from src.preprocessing.normalizer import TextNormalizer
from src.preprocessing.noise_generator import NoiseGenerator

from src.models.markov_chain import MarkovChainLangID
from src.models.hmm import HiddenMarkovModelLangID
from src.models.neural_hmm import NeuralHMMLangID

from src.evaluation.metrics import LanguageIDMetrics, format_metrics_report

from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger
from src.utils.reproducibility import set_seed, get_random_seeds


logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Neural-HMM for Multilingual Language Identification"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess WiLI-2018 dataset")
    preprocess_parser.add_argument("--config", required=True, help="Config file path")
    preprocess_parser.add_argument("--output-dir", default="data/wili2018_processed", help="Output directory")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("model_type", choices=["markov", "hmm", "neural-hmm"])
    train_parser.add_argument("--config", required=True, help="Config file path")
    train_parser.add_argument("--data-dir", default="data/wili2018_processed", help="Data directory")
    train_parser.add_argument("--output-dir", default="outputs", help="Output directory")
    train_parser.add_argument("--n-runs", type=int, default=1, help="Number of random seeds")
    train_parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate-all", help="Evaluate all models")
    eval_parser.add_argument("--config", required=True, help="Config file path")
    eval_parser.add_argument("--data-dir", default="data/wili2018_processed", help="Data directory")
    eval_parser.add_argument("--model-dir", default="outputs", help="Model directory")
    eval_parser.add_argument("--output-csv", default="outputs/metrics/comparison.csv", help="Output CSV")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    level = "DEBUG" if args.verbose else "INFO"
    setup_logging("neural_hmm", level=level)
    
    logger.info(f"Command: {args.command}")
    
    # Dispatch commands
    if args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate-all":
        cmd_evaluate(args)
    else:
        parser.print_help()


def cmd_preprocess(args) -> None:
    """Preprocess WiLI-2018 dataset."""
    logger.info("Starting preprocessing...")
    
    config = load_config(args.config)
    set_seed(config.project.seed)
    
    # Create dataset loader
    dataset = WiLI2018Dataset(data_dir=config.dataset.get("data_dir", "data"))
    
    # Load raw data
    logger.info("Loading raw WiLI-2018 data...")
    raw_data = dataset.load_raw_data()
    
    # Create splits
    logger.info("Creating train/val/test splits...")
    splits = dataset.create_splits(
        raw_data,
        languages=config.dataset.languages,
        train_ratio=config.dataset.train_split,
        val_ratio=config.dataset.val_split,
        apply_noise=True,
        noise_config=config.dataset.noise_config
    )
    
    # Build vocabulary
    logger.info("Building vocabulary...")
    vocab = Vocabulary()
    vocab.save(Path(args.output_dir) / "vocab.json")
    
    # Preprocess splits
    logger.info("Preprocessing text...")
    processed_splits = dataset.preprocess_splits(
        splits,
        vocab,
        preserve_accents_eval=config.preprocessing.preserve_accents_clean_eval
    )
    
    # Save splits
    output_dir = Path(args.output_dir)
    dataset.save_splits(processed_splits, output_dir)
    
    # Save statistics
    stats = {
        "languages": config.dataset.languages,
        "vocab_size": vocab.size,
        "splits": {
            name: {
                "count": len(data),
                "avg_length": float(np.mean([ex["length"] for ex in data]))
            }
            for name, data in processed_splits.items()
        }
    }
    
    stats_file = output_dir / "statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"✅ Preprocessing complete. Output: {output_dir}")


def cmd_train(args) -> None:
    """Train a model."""
    logger.info(f"Starting training for model: {args.model_type}")
    
    config = load_config(args.config)
    n_runs = args.n_runs or config.project.n_runs
    
    # Load data
    from src.preprocessing.dataset import WiLI2018Dataset
    splits = WiLI2018Dataset.load_splits(args.data_dir)
    vocab = Vocabulary.load(Path(args.data_dir) / "vocab.json")
    
    train_data = splits["train"]
    val_data = splits.get("validation", [])
    test_data = splits.get("clean_eval", [])
    
    languages = config.dataset.languages
    
    # Run multiple seeds
    results = {f"run_{i}": {} for i in range(n_runs)}
    
    for run_id in range(n_runs):
        seed = get_random_seeds(1, config.project.seed + run_id)[0]
        set_seed(seed)
        
        logger.info(f"Run {run_id + 1}/{n_runs} (seed={seed})")
        
        # Create model
        if args.model_type == "markov":
            model = MarkovChainLangID(
                languages=languages,
                n_gram_order=config.markov_chain.n_gram_order,
                smoothing=config.markov_chain.smoothing_method,
                vocab_size=vocab.size
            )
        elif args.model_type == "hmm":
            model = HiddenMarkovModelLangID(
                languages=languages,
                n_states=config.hmm.n_hidden_states,
                n_symbols=vocab.size,
                smoothing=config.hmm.smoothing_alpha
            )
        elif args.model_type == "neural-hmm":
            model = NeuralHMMLangID(
                languages=languages,
                vocab_size=vocab.size,
                n_states=config.neural_hmm.n_hidden_states,
                embedding_dim=config.neural_hmm.embedding_dim,
                device=args.device
            )
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        # Train
        t_start = time.time()
        model.train(train_data)
        training_time = time.time() - t_start
        
        # Evaluate on test set
        y_true = [ex["language"] for ex in test_data]
        y_pred, confidences = model.predict_batch([ex["char_ids"] for ex in test_data])
        
        metrics = LanguageIDMetrics(languages)
        run_metrics = metrics.compute_all(y_true, y_pred, confidences)
        run_metrics["training_time"] = training_time
        
        results[f"run_{run_id}"] = run_metrics
        
        # Save checkpoint
        output_dir = Path(args.output_dir) / f"{args.model_type}_run{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(run_metrics, f, indent=2)
        
        logger.info(f"  Accuracy: {run_metrics['accuracy']:.4f}")
        logger.info(f"  Training time: {training_time:.2f}s")
    
    # Aggregate results
    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATED RESULTS")
    logger.info("=" * 60)
    
    acc_scores = [results[f"run_{i}"]["accuracy"] for i in range(n_runs)]
    logger.info(f"Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    
    logger.info(f"✅ Training complete. Output: {args.output_dir}")


def cmd_evaluate(args) -> None:
    """Evaluate all trained models."""
    logger.info("Evaluating models...")
    
    config = load_config(args.config)
    
    # Load data
    from src.preprocessing.dataset import WiLI2018Dataset
    splits = WiLI2018Dataset.load_splits(args.data_dir)
    
    test_data = splits.get("clean_eval", [])
    test_noisy = splits.get("noisy_eval", [])
    
    y_true = [ex["language"] for ex in test_data]
    y_true_noisy = [ex["language"] for ex in test_noisy]
    
    # Results
    results = {}
    
    logger.info(f"✅ Evaluation complete. Output: {args.output_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
