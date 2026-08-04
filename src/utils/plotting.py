"""Visualization utilities for experimental results."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """Generate publication-quality plots for experimental results."""
    
    def __init__(self, output_dir: str | Path = "outputs/plots"):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 11
    
    def plot_confusion_matrix(
        self,
        y_true: List[str],
        y_pred: List[str],
        languages: List[str],
        title: str = "Confusion Matrix",
        normalize: bool = True
    ) -> Path:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            languages: Language codes
            title: Plot title
            normalize: Whether to normalize by true class
            
        Returns:
            Path to saved plot
        """
        cm = confusion_matrix(y_true, y_pred, labels=languages)
        
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=languages,
            yticklabels=languages,
            cbar_kws={"label": "Count" if not normalize else "Proportion"},
            ax=ax
        )
        
        ax.set_xlabel("Predicted Language")
        ax.set_ylabel("True Language")
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = self.output_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved plot to {output_path}")
        return output_path
    
    def plot_model_comparison(
        self,
        models: Dict[str, Dict],
        metric: str = "accuracy",
        title: Optional[str] = None
    ) -> Path:
        """Plot comparison of model performances.
        
        Args:
            models: Dict mapping model names to metrics dicts
            metric: Metric to compare
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        if title is None:
            title = f"Model Comparison ({metric})"
        
        names = list(models.keys())
        values = [models[name].get(metric, 0) for name in names]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, values, color=sns.color_palette("husl", len(names)))
        
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f"{height:.3f}",
                ha="center", va="bottom"
            )
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        output_path = self.output_dir / f"comparison_{metric}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved plot to {output_path}")
        return output_path
    
    def plot_noise_robustness(
        self,
        noise_levels: List[float],
        accuracies: Dict[str, List[float]],
        title: str = "Robustness to Noise"
    ) -> Path:
        """Plot model robustness to noise.
        
        Args:
            noise_levels: Noise severity levels
            accuracies: Dict mapping model names to accuracy lists
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name, acc_values in accuracies.items():
            ax.plot(noise_levels, acc_values, marker="o", label=model_name, linewidth=2)
        
        ax.set_xlabel("Noise Severity")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "noise_robustness.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved plot to {output_path}")
        return output_path
    
    def plot_per_language_performance(
        self,
        per_language: Dict[str, Dict[str, float]],
        metric: str = "f1",
        title: Optional[str] = None
    ) -> Path:
        """Plot per-language performance.
        
        Args:
            per_language: Dict mapping languages to metric dicts
            metric: Metric to plot
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        if title is None:
            title = f"Per-Language {metric.upper()}"
        
        languages = sorted(per_language.keys())
        values = [per_language[lang].get(metric, 0) for lang in languages]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(languages, values, color=sns.color_palette("husl", len(languages)))
        
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.set_ylim([0, 1])
        
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f"{height:.3f}",
                ha="center", va="bottom", fontsize=9
            )
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        output_path = self.output_dir / f"per_language_{metric}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved plot to {output_path}")
        return output_path
    
    def plot_ablation_study(
        self,
        parameter_values: List | np.ndarray,
        metric_values: List[float],
        parameter_name: str = "Parameter",
        metric_name: str = "Accuracy",
        title: Optional[str] = None
    ) -> Path:
        """Plot ablation study results.
        
        Args:
            parameter_values: Values of ablated parameter
            metric_values: Corresponding metric values
            parameter_name: Name of parameter
            metric_name: Name of metric
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        if title is None:
            title = f"Ablation: {parameter_name} vs {metric_name}"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(parameter_values, metric_values, marker="o", linewidth=2, markersize=8)
        
        ax.set_xlabel(parameter_name)
        ax.set_ylabel(metric_name)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Highlight best value
        best_idx = np.argmax(metric_values)
        ax.scatter(
            [parameter_values[best_idx]], [metric_values[best_idx]],
            color="red", s=200, marker="*", zorder=5, label="Best"
        )
        ax.legend()
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"ablation_{parameter_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved plot to {output_path}")
        return output_path
    
    def plot_training_curves(
        self,
        epochs: List[int],
        train_loss: List[float],
        val_loss: List[float],
        title: str = "Training Curves"
    ) -> Path:
        """Plot training and validation loss curves.
        
        Args:
            epochs: Epoch numbers
            train_loss: Training loss values
            val_loss: Validation loss values
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, train_loss, label="Training Loss", marker="o", linewidth=2)
        ax.plot(epochs, val_loss, label="Validation Loss", marker="s", linewidth=2)
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "training_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved plot to {output_path}")
        return output_path


# Convenience functions
def plot_confusion_matrices(results_dir: Path, languages: List[str]) -> None:
    """Plot confusion matrices for all models in results directory.
    
    Args:
        results_dir: Directory containing results
        languages: Language codes
    """
    visualizer = ResultsVisualizer()
    
    for model_dir in results_dir.glob("*"):
        if model_dir.is_dir():
            model_name = model_dir.name
            logger.info(f"Processing {model_name}...")
            # Implementation would load predictions and plot


def generate_report(output_dir: Path, template: Optional[str] = None) -> None:
    """Generate HTML report of experiments.
    
    Args:
        output_dir: Output directory
        template: Optional HTML template
    """
    report_path = output_dir / "report.html"
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Neural-HMM Language Identification - Results Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            img { max-width: 100%; margin: 20px 0; border: 1px solid #ddd; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
        </style>
    </head>
    <body>
        <h1>Neural-HMM Language Identification Results</h1>
        <p>Comparative study of Markov Chain, HMM, and Neural-HMM approaches.</p>
        
        <h2>Model Comparison</h2>
        <img src="comparison_accuracy.png" alt="Model Comparison">
        
        <h2>Confusion Matrices</h2>
        <img src="confusion_matrix.png" alt="Confusion Matrix">
        
        <h2>Robustness Analysis</h2>
        <img src="noise_robustness.png" alt="Noise Robustness">
        
        <footer>
            <hr>
            <p>Report generated by Neural-HMM Language ID project</p>
        </footer>
    </body>
    </html>
    """
    
    with open(report_path, "w") as f:
        f.write(html_content)
    
    logger.info(f"Generated report: {report_path}")
