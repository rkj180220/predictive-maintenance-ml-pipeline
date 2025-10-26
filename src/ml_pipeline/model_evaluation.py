"""
Model Evaluation Module
Author: ramkumarjayakumar
Date: 2025-10-26

Activity 2.3: Evaluate models using multiple metrics
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score
)
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation for predictive maintenance
    Implements multiple metrics and visualization
    """

    def __init__(self):
        self.evaluation_results = {}

    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str) -> Dict:
        """
        Evaluate a single model with comprehensive metrics

        Args:
            model: Trained model
            X_test: Test features
            y_test: True labels
            model_name: Name of the model

        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("="*80)
        logger.info(f"EVALUATING {model_name.upper()}")
        logger.info("="*80)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Calculate all metrics
        metrics = self._calculate_all_metrics(y_test, y_pred, y_pred_proba, model_name)

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Log results
        self._log_evaluation_results(metrics, model_name)

        # Store results
        self.evaluation_results[model_name] = metrics

        return metrics

    def _calculate_all_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                               y_pred_proba: np.ndarray, model_name: str) -> Dict:
        """
        Calculate comprehensive evaluation metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            model_name: Name of the model

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'model_name': model_name,

            # Primary Metrics (Required)
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),

            # Additional Classification Metrics
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'matthews_corrcoef': float(matthews_corrcoef(y_true, y_pred)),
            'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),

            # Confusion Matrix Components
            'true_positives': int(((y_pred == 1) & (y_true == 1)).sum()),
            'true_negatives': int(((y_pred == 0) & (y_true == 0)).sum()),
            'false_positives': int(((y_pred == 1) & (y_true == 0)).sum()),
            'false_negatives': int(((y_pred == 0) & (y_true == 1)).sum()),

            # Business Metrics for Predictive Maintenance
            'false_positive_rate': float(((y_pred == 1) & (y_true == 0)).sum() / (y_true == 0).sum()),
            'false_negative_rate': float(((y_pred == 0) & (y_true == 1)).sum() / (y_true == 1).sum()),
            'true_positive_rate': float(recall_score(y_true, y_pred, zero_division=0)),  # Same as recall
            'specificity': float(((y_pred == 0) & (y_true == 0)).sum() / (y_true == 0).sum()),
        }

        # Probability-based metrics (if available)
        if y_pred_proba is not None:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
            metrics['average_precision'] = float(average_precision_score(y_true, y_pred_proba))

        # Calculate cost-based metrics for predictive maintenance
        # Assumptions: False Negative cost >> False Positive cost
        fn_cost = 10000  # Cost of missed failure (equipment damage, downtime)
        fp_cost = 1000   # Cost of unnecessary maintenance

        total_cost = (metrics['false_negatives'] * fn_cost +
                     metrics['false_positives'] * fp_cost)

        metrics['business_metrics'] = {
            'false_negative_cost': metrics['false_negatives'] * fn_cost,
            'false_positive_cost': metrics['false_positives'] * fp_cost,
            'total_cost': total_cost,
            'cost_per_prediction': total_cost / len(y_true)
        }

        return metrics

    def _log_evaluation_results(self, metrics: Dict, model_name: str):
        """Log evaluation results in a formatted manner"""

        logger.info(f"\n📊 EVALUATION RESULTS FOR {model_name.upper()}")
        logger.info("-"*80)

        logger.info("\n🎯 PRIMARY METRICS (Required):")
        logger.info(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"   Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        logger.info(f"   Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        logger.info(f"   F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")

        logger.info("\n📈 ADDITIONAL METRICS:")
        logger.info(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        logger.info(f"   Matthews Corr:     {metrics['matthews_corrcoef']:.4f}")
        logger.info(f"   Cohen's Kappa:     {metrics['cohen_kappa']:.4f}")

        if 'roc_auc' in metrics:
            logger.info(f"   ROC-AUC Score:     {metrics['roc_auc']:.4f}")
            logger.info(f"   Average Precision: {metrics['average_precision']:.4f}")

        logger.info("\n🔢 CONFUSION MATRIX:")
        logger.info(f"   True Positives:  {metrics['true_positives']:,}")
        logger.info(f"   True Negatives:  {metrics['true_negatives']:,}")
        logger.info(f"   False Positives: {metrics['false_positives']:,}")
        logger.info(f"   False Negatives: {metrics['false_negatives']:,}")

        logger.info("\n⚠️ ERROR RATES:")
        logger.info(f"   False Positive Rate: {metrics['false_positive_rate']:.4f} ({metrics['false_positive_rate']*100:.2f}%)")
        logger.info(f"   False Negative Rate: {metrics['false_negative_rate']:.4f} ({metrics['false_negative_rate']*100:.2f}%)")
        logger.info(f"   Specificity:         {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")

        logger.info("\n💰 BUSINESS IMPACT:")
        logger.info(f"   False Negative Cost: ${metrics['business_metrics']['false_negative_cost']:,.2f}")
        logger.info(f"   False Positive Cost: ${metrics['business_metrics']['false_positive_cost']:,.2f}")
        logger.info(f"   Total Cost:          ${metrics['business_metrics']['total_cost']:,.2f}")
        logger.info(f"   Cost per Prediction: ${metrics['business_metrics']['cost_per_prediction']:.2f}")

        logger.info("="*80 + "\n")

    def compare_models(self, results: Dict) -> pd.DataFrame:
        """
        Compare multiple models side by side

        Args:
            results: Dictionary with evaluation results for all models

        Returns:
            DataFrame with comparison metrics
        """
        logger.info("="*80)
        logger.info("MODEL COMPARISON")
        logger.info("="*80)

        comparison_data = []

        for model_name, metrics in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics.get('roc_auc', 'N/A'),
                'False Positive Rate': metrics['false_positive_rate'],
                'False Negative Rate': metrics['false_negative_rate'],
                'Total Cost': metrics['business_metrics']['total_cost']
            })

        comparison_df = pd.DataFrame(comparison_data)

        logger.info("\n📊 MODEL COMPARISON TABLE:")
        logger.info("\n" + comparison_df.to_string(index=False))

        # Identify best model for each metric
        logger.info("\n🏆 BEST MODEL PER METRIC:")
        for col in comparison_df.columns[1:]:
            if col == 'Total Cost' or 'Rate' in col:
                best_idx = comparison_df[col].astype(float).idxmin()
                best_model = comparison_df.loc[best_idx, 'Model']
                best_value = comparison_df.loc[best_idx, col]
                logger.info(f"   {col}: {best_model} ({best_value})")
            else:
                best_idx = comparison_df[col].astype(float).idxmax()
                best_model = comparison_df.loc[best_idx, 'Model']
                best_value = comparison_df.loc[best_idx, col]
                logger.info(f"   {col}: {best_model} ({best_value})")

        logger.info("="*80 + "\n")

        return comparison_df

    def plot_confusion_matrices(self, output_dir: Path):
        """
        Plot confusion matrices for all models

        Args:
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_models = len(self.evaluation_results)
        fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, metrics) in enumerate(self.evaluation_results.items()):
            cm = np.array(metrics['confusion_matrix'])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['No Failure', 'Failure'],
                       yticklabels=['No Failure', 'Failure'])

            axes[idx].set_title(f'{model_name}\nAccuracy: {metrics["accuracy"]:.3f}',
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)

        plt.tight_layout()
        plot_path = output_dir / 'confusion_matrices.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 Confusion matrices saved to: {plot_path}")

        return plot_path

    def plot_metrics_comparison(self, comparison_df: pd.DataFrame, output_dir: Path):
        """
        Plot metrics comparison bar chart

        Args:
            comparison_df: DataFrame with comparison metrics
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Select key metrics for visualization
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(comparison_df))
        width = 0.2

        for idx, metric in enumerate(metrics_to_plot):
            offset = width * (idx - len(metrics_to_plot)/2 + 0.5)
            ax.bar(x + offset, comparison_df[metric], width, label=metric)

        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plot_path = output_dir / 'metrics_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 Metrics comparison saved to: {plot_path}")

        return plot_path

    def save_evaluation_results(self, output_dir: Path):
        """
        Save evaluation results to JSON

        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / 'evaluation_results.json'

        with open(results_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)

        logger.info(f"💾 Evaluation results saved to: {results_path}")

        return results_path

    def generate_evaluation_report(self, output_dir: Path):
        """
        Generate comprehensive evaluation report

        Args:
            output_dir: Directory to save report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / 'evaluation_report.txt'

        with open(report_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("MODEL EVALUATION REPORT - PREDICTIVE MAINTENANCE\n")
            f.write("NASA C-MAPSS Turbofan Engine Dataset\n")
            f.write("="*100 + "\n\n")

            for model_name, metrics in self.evaluation_results.items():
                f.write(f"\n{'='*100}\n")
                f.write(f"{model_name.upper()}\n")
                f.write(f"{'='*100}\n\n")

                f.write("PRIMARY METRICS:\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
                f.write(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
                f.write(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
                f.write(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)\n\n")

                f.write("CONFUSION MATRIX:\n")
                f.write(f"  True Positives:  {metrics['true_positives']:,}\n")
                f.write(f"  True Negatives:  {metrics['true_negatives']:,}\n")
                f.write(f"  False Positives: {metrics['false_positives']:,}\n")
                f.write(f"  False Negatives: {metrics['false_negatives']:,}\n\n")

                f.write("BUSINESS IMPACT:\n")
                f.write(f"  Total Cost: ${metrics['business_metrics']['total_cost']:,.2f}\n\n")

            f.write("="*100 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*100 + "\n")

        logger.info(f"📄 Evaluation report saved to: {report_path}")

        return report_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Model Evaluation Module Initialized")

