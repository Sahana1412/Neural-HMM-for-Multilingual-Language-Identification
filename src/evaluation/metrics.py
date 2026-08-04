"""Comprehensive evaluation metrics for language identification."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)


logger = logging.getLogger(__name__)


class LanguageIDMetrics:
    """Comprehensive metrics for multilingual language identification."""
    
    def __init__(self, languages: List[str]):
        """Initialize metrics calculator.
        
        Args:
            languages: List of language codes
        """
        self.languages = sorted(languages)
        self.lang2id = {lang: idx for idx, lang in enumerate(self.languages)}
        self.id2lang = {idx: lang for lang, idx in self.lang2id.items()}
    
    def compute_all(
        self,
        y_true: List[str],
        y_pred: List[str],
        confidences: Optional[List[float]] = None
    ) -> Dict:
        """Compute all metrics.
        
        Args:
            y_true: True language labels
            y_pred: Predicted language labels
            confidences: Prediction confidences (optional)
            
        Returns:
            Dict containing all computed metrics
        """
        metrics = {}
        
        # Basic accuracy
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        
        # Per-language metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.languages, zero_division=0
        )
        
        metrics["precision_macro"] = np.mean(precision)
        metrics["precision_weighted"] = np.average(precision, weights=support)
        metrics["recall_macro"] = np.mean(recall)
        metrics["recall_weighted"] = np.average(recall, weights=support)
        metrics["f1_macro"] = np.mean(f1)
        metrics["f1_weighted"] = np.average(f1, weights=support)
        
        # Per-language breakdown
        metrics["per_language"] = {}
        for lang, prec, rec, f1_score, sup in zip(
            self.languages, precision, recall, f1, support
        ):
            metrics["per_language"][lang] = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1_score),
                "support": int(sup)
            }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.languages)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Calibration metrics
        if confidences is not None:
            metrics["calibration"] = self._compute_calibration(y_true, y_pred, confidences)
        
        return metrics
    
    def _compute_calibration(
        self,
        y_true: List[str],
        y_pred: List[str],
        confidences: List[float],
        n_bins: int = 10
    ) -> Dict:
        """Compute calibration metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            confidences: Prediction confidences
            n_bins: Number of calibration bins
            
        Returns:
            Dict with calibration metrics
        """
        confidences = np.array(confidences)
        is_correct = np.array([p == t for p, t in zip(y_pred, y_true)])
        
        # Expected Calibration Error (ECE)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for bin_start, bin_end in zip(bin_edges[:-1], bin_edges[1:]):
            in_bin = (confidences >= bin_start) & (confidences < bin_end)
            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(is_correct[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
                ece += np.abs(bin_accuracy - bin_confidence) * np.mean(in_bin)
        
        # Brier score (mean squared error)
        probs_correct = np.where(is_correct, confidences, 1 - confidences)
        brier = np.mean((confidences - is_correct) ** 2)
        
        return {
            "ece": float(ece),
            "brier_score": float(brier),
            "max_confidence": float(np.max(confidences)),
            "min_confidence": float(np.min(confidences)),
            "mean_confidence": float(np.mean(confidences))
        }
    
    def per_language_accuracy(self, y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """Compute accuracy for each language.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dict mapping language to accuracy
        """
        accuracy_by_lang = {}
        
        for lang in self.languages:
            lang_mask = np.array([t == lang for t in y_true])
            if np.sum(lang_mask) > 0:
                lang_correct = np.sum([p == t for p, t in zip(y_pred, y_true) if t == lang])
                lang_total = np.sum(lang_mask)
                accuracy_by_lang[lang] = lang_correct / lang_total
            else:
                accuracy_by_lang[lang] = 0.0
        
        return accuracy_by_lang
    
    def to_dict(self) -> Dict:
        """Convert to dict representation."""
        return {
            "languages": self.languages,
            "lang2id": self.lang2id,
            "id2lang": self.id2lang
        }


def compute_robustness_metrics(
    y_true: List[str],
    y_pred_clean: List[str],
    y_pred_noisy: List[str],
    confidences_clean: Optional[List[float]] = None,
    confidences_noisy: Optional[List[float]] = None
) -> Dict:
    """Compute robustness metrics comparing clean and noisy performance.
    
    Args:
        y_true: True labels
        y_pred_clean: Predictions on clean data
        y_pred_noisy: Predictions on noisy data
        confidences_clean: Clean confidences
        confidences_noisy: Noisy confidences
        
    Returns:
        Dict with robustness metrics
    """
    acc_clean = accuracy_score(y_true, y_pred_clean)
    acc_noisy = accuracy_score(y_true, y_pred_noisy)
    
    robustness = {
        "accuracy_clean": float(acc_clean),
        "accuracy_noisy": float(acc_noisy),
        "accuracy_drop": float(acc_clean - acc_noisy),
        "robustness_ratio": float(acc_noisy / max(acc_clean, 1e-10))
    }
    
    if confidences_clean and confidences_noisy:
        conf_clean_mean = np.mean(confidences_clean)
        conf_noisy_mean = np.mean(confidences_noisy)
        robustness["confidence_clean_mean"] = float(conf_clean_mean)
        robustness["confidence_noisy_mean"] = float(conf_noisy_mean)
        robustness["confidence_drop"] = float(conf_clean_mean - conf_noisy_mean)
    
    return robustness


def format_metrics_report(metrics: Dict, languages: List[str]) -> str:
    """Format metrics as readable report.
    
    Args:
        metrics: Metrics dict from compute_all
        languages: List of languages
        
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("LANGUAGE IDENTIFICATION METRICS")
    lines.append("=" * 60)
    
    # Overall metrics
    lines.append(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    lines.append(f"Macro-averaged F1: {metrics['f1_macro']:.4f}")
    lines.append(f"Weighted F1: {metrics['f1_weighted']:.4f}")
    
    # Per-language metrics
    lines.append("\nPer-Language Performance:")
    lines.append("-" * 60)
    lines.append(f"{'Language':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<8}")
    lines.append("-" * 60)
    
    for lang in languages:
        if lang in metrics["per_language"]:
            data = metrics["per_language"][lang]
            lines.append(
                f"{lang:<15} {data['precision']:<12.4f} {data['recall']:<12.4f} "
                f"{data['f1']:<12.4f} {data['support']:<8d}"
            )
    
    lines.append("-" * 60)
    
    # Calibration metrics
    if "calibration" in metrics:
        lines.append("\nCalibration Metrics:")
        cal = metrics["calibration"]
        lines.append(f"  Expected Calibration Error (ECE): {cal['ece']:.4f}")
        lines.append(f"  Brier Score: {cal['brier_score']:.4f}")
        lines.append(f"  Mean Confidence: {cal['mean_confidence']:.4f}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)
