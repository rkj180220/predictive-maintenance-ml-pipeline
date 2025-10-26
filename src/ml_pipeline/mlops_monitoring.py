"""
MLOps Monitoring Module
Author: ramkumarjayakumar
Date: 2025-10-26

Activity 2.4: Monitor models and log metrics
Integrates with AWS CloudWatch and provides comprehensive ML metrics tracking
"""

import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MLOpsMonitor:
    """
    MLOps monitoring and logging for predictive maintenance models
    Tracks model performance, drift, and operational metrics
    """

    def __init__(self, log_dir: Path, enable_cloudwatch: bool = False):
        """
        Initialize MLOps Monitor

        Args:
            log_dir: Directory for logging
            enable_cloudwatch: Enable AWS CloudWatch integration
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enable_cloudwatch = enable_cloudwatch
        self.metrics_history = []

        if enable_cloudwatch:
            self._setup_cloudwatch()

    def _setup_cloudwatch(self):
        """Setup AWS CloudWatch integration"""
        try:
            import boto3
            import watchtower

            self.cloudwatch_client = boto3.client('logs')

            # Create CloudWatch handler
            cw_handler = watchtower.CloudWatchLogHandler(
                log_group='/ml-pipeline/predictive-maintenance',
                stream_name=f'model-metrics-{datetime.now().strftime("%Y%m%d")}',
                send_interval=10
            )

            logger.addHandler(cw_handler)
            logger.info("✅ AWS CloudWatch integration enabled")

        except Exception as e:
            logger.warning(f"CloudWatch setup failed: {e}. Continuing with local logging only.")
            self.enable_cloudwatch = False

    def log_training_metrics(self, model_name: str, training_metadata: Dict,
                            evaluation_metrics: Dict):
        """
        Log comprehensive training and evaluation metrics

        Args:
            model_name: Name of the model
            training_metadata: Training metadata from ModelTrainer
            evaluation_metrics: Evaluation metrics from ModelEvaluator
        """
        logger.info("="*80)
        logger.info(f"LOGGING MLOPS METRICS FOR {model_name.upper()}")
        logger.info("="*80)

        # Combine all metrics
        mlops_metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'pipeline_stage': 'training_and_evaluation',

            # Training Metrics
            'training': {
                'training_samples': training_metadata.get('training_samples'),
                'n_features': training_metadata.get('n_features'),
                'training_time_seconds': training_metadata.get('training_time_seconds'),
                'training_time_minutes': training_metadata.get('training_time_minutes'),
            },

            # Core Performance Metrics (4 Required)
            'performance_metrics': {
                'accuracy': evaluation_metrics['accuracy'],
                'precision': evaluation_metrics['precision'],
                'recall': evaluation_metrics['recall'],
                'f1_score': evaluation_metrics['f1_score']
            },

            # Extended Performance Metrics
            'extended_metrics': {
                'balanced_accuracy': evaluation_metrics['balanced_accuracy'],
                'matthews_corrcoef': evaluation_metrics['matthews_corrcoef'],
                'cohen_kappa': evaluation_metrics['cohen_kappa'],
                'roc_auc': evaluation_metrics.get('roc_auc', None),
                'average_precision': evaluation_metrics.get('average_precision', None)
            },

            # Confusion Matrix
            'confusion_matrix': {
                'true_positives': evaluation_metrics['true_positives'],
                'true_negatives': evaluation_metrics['true_negatives'],
                'false_positives': evaluation_metrics['false_positives'],
                'false_negatives': evaluation_metrics['false_negatives']
            },

            # Error Rates (Critical for Predictive Maintenance)
            'error_rates': {
                'false_positive_rate': evaluation_metrics['false_positive_rate'],
                'false_negative_rate': evaluation_metrics['false_negative_rate'],
                'true_positive_rate': evaluation_metrics['true_positive_rate'],
                'specificity': evaluation_metrics['specificity']
            },

            # Business Impact Metrics
            'business_impact': evaluation_metrics['business_metrics'],

            # Model Health Status
            'model_health': self._assess_model_health(evaluation_metrics)
        }

        # Store in history
        self.metrics_history.append(mlops_metrics)

        # Log to local file
        self._save_metrics_to_file(mlops_metrics, model_name)

        # Log structured metrics to logger (will go to CloudWatch if enabled)
        self._log_structured_metrics(mlops_metrics)

        # Display summary
        self._display_mlops_summary(mlops_metrics)

        logger.info("="*80 + "\n")

        return mlops_metrics

    def _assess_model_health(self, metrics: Dict) -> Dict:
        """
        Assess overall model health based on metrics

        Args:
            metrics: Evaluation metrics

        Returns:
            Dictionary with health assessment
        """
        health_score = 0
        issues = []
        warnings = []

        # Check accuracy
        if metrics['accuracy'] >= 0.90:
            health_score += 25
        elif metrics['accuracy'] >= 0.80:
            health_score += 15
            warnings.append("Accuracy below 90%")
        else:
            health_score += 5
            issues.append("Accuracy below 80%")

        # Check recall (critical for catching failures)
        if metrics['recall'] >= 0.85:
            health_score += 25
        elif metrics['recall'] >= 0.70:
            health_score += 15
            warnings.append("Recall below 85% - may miss failures")
        else:
            health_score += 5
            issues.append("Recall below 70% - HIGH RISK of missed failures")

        # Check precision (avoid unnecessary maintenance)
        if metrics['precision'] >= 0.80:
            health_score += 25
        elif metrics['precision'] >= 0.60:
            health_score += 15
            warnings.append("Precision below 80% - more false alarms")
        else:
            health_score += 5
            issues.append("Precision below 60% - many false alarms")

        # Check F1-score (balance)
        if metrics['f1_score'] >= 0.85:
            health_score += 25
        elif metrics['f1_score'] >= 0.70:
            health_score += 15
            warnings.append("F1-score below 85%")
        else:
            health_score += 5
            issues.append("F1-score below 70%")

        # Determine health status
        if health_score >= 90:
            status = "EXCELLENT"
        elif health_score >= 70:
            status = "GOOD"
        elif health_score >= 50:
            status = "FAIR"
        else:
            status = "POOR"

        return {
            'status': status,
            'health_score': health_score,
            'issues': issues,
            'warnings': warnings,
            'recommendation': self._get_health_recommendation(status, issues, warnings)
        }

    def _get_health_recommendation(self, status: str, issues: list, warnings: list) -> str:
        """Generate recommendation based on health status"""
        if status == "EXCELLENT":
            return "Model performing exceptionally well. Deploy to production."
        elif status == "GOOD":
            return "Model performing well. Minor improvements possible through hyperparameter tuning."
        elif status == "FAIR":
            return "Model needs improvement. Consider feature engineering or algorithm change."
        else:
            return "Model performance inadequate. Recommend retraining with different approach."

    def _save_metrics_to_file(self, metrics: Dict, model_name: str):
        """Save metrics to JSON file"""
        filename = f"mlops_metrics_{model_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.log_dir / filename

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info(f"💾 MLOps metrics saved to: {filepath}")

    def _log_structured_metrics(self, metrics: Dict):
        """Log metrics in structured format (for CloudWatch)"""
        logger.info(f"MLOps Structured Metrics: {json.dumps(metrics, default=str)}")

    def _display_mlops_summary(self, metrics: Dict):
        """Display MLOps summary"""
        logger.info("\n📊 MLOPS MONITORING SUMMARY")
        logger.info("-"*80)

        logger.info(f"\n🎯 Core Performance Metrics (4 Required):")
        for metric, value in metrics['performance_metrics'].items():
            logger.info(f"   {metric.capitalize():20s}: {value:.4f} ({value*100:.2f}%)")

        logger.info(f"\n⚠️ Error Rates (Predictive Maintenance Critical):")
        logger.info(f"   False Positive Rate: {metrics['error_rates']['false_positive_rate']:.4f} ({metrics['error_rates']['false_positive_rate']*100:.2f}%)")
        logger.info(f"   False Negative Rate: {metrics['error_rates']['false_negative_rate']:.4f} ({metrics['error_rates']['false_negative_rate']*100:.2f}%)")

        logger.info(f"\n💰 Business Impact:")
        logger.info(f"   Total Cost: ${metrics['business_impact']['total_cost']:,.2f}")
        logger.info(f"   Cost per Prediction: ${metrics['business_impact']['cost_per_prediction']:.2f}")

        logger.info(f"\n🏥 Model Health Status: {metrics['model_health']['status']}")
        logger.info(f"   Health Score: {metrics['model_health']['health_score']}/100")

        if metrics['model_health']['issues']:
            logger.warning(f"   ⚠️ Issues: {', '.join(metrics['model_health']['issues'])}")

        if metrics['model_health']['warnings']:
            logger.info(f"   ⚡ Warnings: {', '.join(metrics['model_health']['warnings'])}")

        logger.info(f"\n💡 Recommendation: {metrics['model_health']['recommendation']}")

    def monitor_data_drift(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict:
        """
        Monitor data drift between training and test sets

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Dictionary with drift metrics
        """
        logger.info("\n" + "="*80)
        logger.info("DATA DRIFT MONITORING")
        logger.info("="*80)

        drift_metrics = {
            'timestamp': datetime.now().isoformat(),
            'n_features': X_train.shape[1],
            'feature_drift': {}
        }

        # Calculate drift for each feature
        drifted_features = []

        for col in X_train.columns[:20]:  # Sample first 20 features for efficiency
            train_mean = X_train[col].mean()
            test_mean = X_test[col].mean()
            train_std = X_train[col].std()

            # Calculate normalized drift
            if train_std > 0:
                drift_score = abs(test_mean - train_mean) / train_std
            else:
                drift_score = 0

            drift_metrics['feature_drift'][col] = {
                'train_mean': float(train_mean),
                'test_mean': float(test_mean),
                'drift_score': float(drift_score),
                'drifted': drift_score > 2.0  # More than 2 std deviations
            }

            if drift_score > 2.0:
                drifted_features.append(col)

        drift_metrics['summary'] = {
            'total_features_checked': len(drift_metrics['feature_drift']),
            'drifted_features_count': len(drifted_features),
            'drifted_features': drifted_features,
            'drift_detected': len(drifted_features) > 0
        }

        logger.info(f"\n📊 Data Drift Summary:")
        logger.info(f"   Features Checked: {drift_metrics['summary']['total_features_checked']}")
        logger.info(f"   Drifted Features: {drift_metrics['summary']['drifted_features_count']}")

        if drifted_features:
            logger.warning(f"   ⚠️ Drift Detected in: {', '.join(drifted_features[:5])}")
            logger.warning("   Consider retraining the model with recent data")
        else:
            logger.info("   ✅ No significant drift detected")

        # Save drift metrics
        drift_path = self.log_dir / f"data_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(drift_path, 'w') as f:
            json.dump(drift_metrics, f, indent=2, default=str)

        logger.info(f"\n💾 Drift metrics saved to: {drift_path}")
        logger.info("="*80 + "\n")

        return drift_metrics

    def generate_mlops_dashboard_data(self, output_path: Path) -> Dict:
        """
        Generate data for MLOps dashboard

        Args:
            output_path: Path to save dashboard data

        Returns:
            Dictionary with dashboard data
        """
        if not self.metrics_history:
            logger.warning("No metrics history available")
            return {}

        dashboard_data = {
            'last_updated': datetime.now().isoformat(),
            'total_models_monitored': len(self.metrics_history),
            'models': []
        }

        for metrics in self.metrics_history:
            model_summary = {
                'model_name': metrics['model_name'],
                'timestamp': metrics['timestamp'],
                'accuracy': metrics['performance_metrics']['accuracy'],
                'precision': metrics['performance_metrics']['precision'],
                'recall': metrics['performance_metrics']['recall'],
                'f1_score': metrics['performance_metrics']['f1_score'],
                'health_status': metrics['model_health']['status'],
                'health_score': metrics['model_health']['health_score'],
                'total_cost': metrics['business_impact']['total_cost']
            }
            dashboard_data['models'].append(model_summary)

        # Save dashboard data
        with open(output_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2)

        logger.info(f"📊 Dashboard data saved to: {output_path}")

        return dashboard_data

    def save_all_metrics(self):
        """Save all collected metrics"""
        all_metrics_path = self.log_dir / f"mlops_all_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(all_metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)

        logger.info(f"💾 All MLOps metrics saved to: {all_metrics_path}")

        return all_metrics_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("MLOps Monitoring Module Initialized")

