"""WiLI-2018 dataset loader and preprocessor."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .normalizer import TextNormalizer
from .noise_generator import NoiseGenerator
from .vocab import Vocabulary


logger = logging.getLogger(__name__)


class WiLI2018Dataset:
    """WiLI-2018 multilingual language identification dataset."""
    
    DOWNLOAD_URL = "https://zenodo.org/record/841984/files/wili-2018.zip"
    
    def __init__(self, data_dir: str | Path = "data", seed: int = 42):
        """Initialize dataset loader.
        
        Args:
            data_dir: Directory to store/load dataset
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        self.normalizer = TextNormalizer()
        self.noise_generator = NoiseGenerator(seed=seed)
    
    def download_and_extract(self) -> Path:
        """Download and extract WiLI-2018 dataset.
        
        Returns:
            Path to extracted dataset
        """
        import urllib.request
        import zipfile
        
        zip_path = self.data_dir / "wili-2018.zip"
        extract_dir = self.data_dir / "wili-2018"
        
        if extract_dir.exists():
            logger.info(f"Dataset already extracted at {extract_dir}")
            return extract_dir
        
        # Download
        if not zip_path.exists():
            logger.info(f"Downloading WiLI-2018 from {self.DOWNLOAD_URL}...")
            try:
                urllib.request.urlretrieve(self.DOWNLOAD_URL, zip_path)
                logger.info(f"Downloaded to {zip_path}")
            except Exception as e:
                logger.error(f"Failed to download: {e}")
                raise
        
        # Extract
        logger.info(f"Extracting to {extract_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        return extract_dir
    
    def load_raw_data(self, dataset_dir: Optional[Path] = None) -> Dict[str, List[Tuple[str, str]]]:
        """Load raw WiLI-2018 data from extracted directory.
        
        Args:
            dataset_dir: Path to extracted wili-2018 directory
            
        Returns:
            Dict mapping language codes to list of (text, label) tuples
        """
        if dataset_dir is None:
            dataset_dir = self.data_dir / "wili-2018"
        
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            logger.info("Attempting to download...")
            dataset_dir = self.download_and_extract()
        
        data = {}
        
        # WiLI-2018 structure: x_test.txt (texts) and y_test.txt (labels)
        # Similar for x_train.txt and y_train.txt
        files = list(dataset_dir.glob("*.txt"))
        
        if not files:
            # Try subdirectories
            for subdir in dataset_dir.iterdir():
                if subdir.is_dir():
                    files.extend(subdir.glob("*.txt"))
        
        logger.info(f"Found {len(files)} data files")
        x_files = list(dataset_dir.glob("x_train*")) + list(dataset_dir.glob("x_test*"))
        # Simple loader: assume x_*.txt and y_*.txt format
        for x_file in x_files:
            y_file = x_file.parent / x_file.name.replace("x_", "y_")
            if not y_file.exists():
                # Try alternate extension
                if y_file.suffix == ".txt":
                    y_file = x_file.parent / y_file.name[:-4]
                else:
                    y_file = x_file.parent / (y_file.name + ".txt")
            if y_file.exists():
                with open(x_file, encoding="utf-8") as fx, open(y_file, encoding="utf-8") as fy:
                    texts = fx.readlines()
                    labels = fy.readlines()
                    for text, label in zip(texts, labels):
                        text = text.strip()
                        label = label.strip()
                        if text and label:
                            if label not in data:
                                data[label] = []
                            data[label].append((text, label))
        
        logger.info(f"Loaded data for languages: {list(data.keys())}")
        return data
    
    def create_splits(
        self,
        data: Dict[str, List[Tuple[str, str]]],
        languages: List[str],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        apply_noise: bool = True,
        noise_config: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Create train/validation/clean_eval/noisy_eval splits.
        
        Args:
            data: Raw data dict from load_raw_data
            languages: Languages to include
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            apply_noise: Whether to generate noisy eval split
            noise_config: Noise configuration
            
        Returns:
            Dict with splits: train, validation, clean_eval, noisy_eval
        """
        splits = {"train": [], "validation": [], "clean_eval": [], "noisy_eval": []}
        
        if noise_config is None:
            noise_config = {}
        
        for lang in languages:
            if lang not in data:
                logger.warning(f"Language {lang} not found in dataset")
                continue
            
            examples = data[lang]
            logger.info(f"Language {lang}: {len(examples)} examples")
            
            # Shuffle
            self.rng.shuffle(examples)
            
            # Split
            n = len(examples)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train_data = examples[:n_train]
            val_data = examples[n_train:n_train + n_val]
            eval_data = examples[n_train + n_val:]
            
            splits["train"].extend(train_data)
            splits["validation"].extend(val_data)
            splits["clean_eval"].extend(eval_data)
            
            # Create noisy eval by applying noise
            if apply_noise:
                for text, label in eval_data:
                    noisy_text = self.noise_generator.add_noise(text, noise_config)
                    splits["noisy_eval"].append((noisy_text, label))
        
        # Log split sizes
        for split_name, split_data in splits.items():
            logger.info(f"Split '{split_name}': {len(split_data)} examples")
        
        return splits
    
    def preprocess_splits(
        self,
        splits: Dict[str, List[Tuple[str, str]]],
        vocab: Vocabulary,
        preserve_accents_eval: bool = True
    ) -> Dict[str, List[Dict]]:
        """Preprocess splits: normalize, tokenize, encode.
        
        Args:
            splits: Data splits from create_splits
            vocab: Vocabulary for encoding
            preserve_accents_eval: Keep accents in CleanEval
            
        Returns:
            Dict mapping split names to list of preprocessed examples
        """
        processed = {}
        
        for split_name, split_data in splits.items():
            processed_examples = []
            
            for text, label in split_data:
                # Normalize (preserve accents for clean_eval)
                if split_name == "clean_eval" and preserve_accents_eval:
                    normalizer = TextNormalizer(preserve_accents=True)
                    normalized = normalizer.normalize(text)
                else:
                    normalized = self.normalizer.normalize(text)
                
                # Encode to character IDs
                char_ids = vocab.encode(normalized)
                
                processed_examples.append({
                    "text": text,
                    "normalized": normalized,
                    "char_ids": char_ids,
                    "language": label,
                    "length": len(char_ids)
                })
            
            processed[split_name] = processed_examples
            logger.info(f"Processed split '{split_name}': {len(processed_examples)} examples")
        
        return processed
    
    def save_splits(self, splits: Dict[str, List[Dict]], output_dir: str | Path) -> None:
        """Save preprocessed splits to JSONL files.
        
        Args:
            splits: Processed splits dict
            output_dir: Directory to save to
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in splits.items():
            output_file = output_dir / f"{split_name}.jsonl"
            
            with open(output_file, "w", encoding="utf-8") as f:
                for example in split_data:
                    f.write(json.dumps(example) + "\n")
            
            logger.info(f"Saved {split_name} to {output_file}")
    
    @staticmethod
    def load_splits(data_dir: str | Path) -> Dict[str, List[Dict]]:
        """Load preprocessed splits from JSONL files.
        
        Args:
            data_dir: Directory containing split files
            
        Returns:
            Dict mapping split names to data
        """
        data_dir = Path(data_dir)
        splits = {}
        
        for split_file in data_dir.glob("*.jsonl"):
            split_name = split_file.stem
            examples = []
            
            with open(split_file, encoding="utf-8") as f:
                for line in f:
                    examples.append(json.loads(line))
            
            splits[split_name] = examples
            logger.info(f"Loaded {split_name}: {len(examples)} examples")
        
        return splits
