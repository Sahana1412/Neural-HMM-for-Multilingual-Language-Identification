"""Neural-HMM Language Identification package."""

__version__ = "1.0.0"
__author__ = "Neural-HMM Team"

# Import main classes for easy access
try:
    from src.models import MarkovChainLangID, HiddenMarkovModelLangID, NeuralHMMLangID
    from src.preprocessing import Vocabulary, WiLI2018Dataset, TextNormalizer, NoiseGenerator
    from src.evaluation import LanguageIDMetrics
except ImportError:
    pass

__all__ = [
    "MarkovChainLangID",
    "HiddenMarkovModelLangID",
    "NeuralHMMLangID",
    "Vocabulary",
    "WiLI2018Dataset",
    "TextNormalizer",
    "NoiseGenerator",
    "LanguageIDMetrics",
]
