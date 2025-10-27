"""
ML Pipeline Service - Retrieve ML pipeline information and metrics from logs
Author: ramkumarjayakumar
Date: 2025-10-27
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class MLPipelineService:
    """
    Service to retrieve ML pipeline information and monitoring data from logs
    """

    def __init__(self, project_root: Path):
        """
        Initialize ML pipeline service

        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.logs_dir = self.project_root / 'logs'
        self.models_dir = self.project_root / 'models'
        logger.info(f"Initializing MLPipelineService with project_root: {self.project_root}")

    def get_evaluation_results(self) -> Dict[str, Any]:
        """
        Retrieve model evaluation results from evaluation_results.json

        Returns:
            Model evaluation metrics for all models
        """
        logger.info("Retrieving evaluation results...")

        try:
            eval_path = self.logs_dir / 'evaluation_results.json'

            if eval_path.exists():
                with open(eval_path, 'r') as f:
                    data = json.load(f)
                return {
                    'status': 'success',
                    'data': data,
                    'source': str(eval_path)
                }
            else:
                logger.warning("Evaluation results file not found")
                return {
                    'status': 'error',
                    'message': 'Evaluation results file not found',
                    'source': str(eval_path)
                }
        except Exception as e:
            logger.error(f"Error retrieving evaluation results: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Retrieve feature importance data from feature_importance.json

        Returns:
            Feature importance scores for all models
        """
        logger.info("Retrieving feature importance...")

        try:
            importance_path = self.logs_dir / 'feature_importance.json'

            if importance_path.exists():
                with open(importance_path, 'r') as f:
                    data = json.load(f)
                return {
                    'status': 'success',
                    'data': data,
                    'source': str(importance_path)
                }
            else:
                logger.warning("Feature importance file not found")
                return {
                    'status': 'error',
                    'message': 'Feature importance file not found',
                    'source': str(importance_path)
                }
        except Exception as e:
            logger.error(f"Error retrieving feature importance: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_mlops_metrics(self, model_name: str = None) -> Dict[str, Any]:
        """
        Retrieve MLOps metrics from mlops_metrics_*.json files

        Args:
            model_name: Optional model name to filter (e.g., 'xgboost', 'random_forest')

        Returns:
            MLOps metrics for specified model or all models
        """
        logger.info(f"Retrieving MLOps metrics for model: {model_name or 'all'}...")

        try:
            metric_files = sorted(self.logs_dir.glob('mlops_metrics_*.json'), reverse=True)

            if not metric_files:
                logger.warning("No MLOps metric files found")
                return {
                    'status': 'error',
                    'message': 'No MLOps metric files found'
                }

            # Group by model and get latest
            models_data = {}

            for metric_file in metric_files:
                # Extract model name from filename (e.g., mlops_metrics_xgboost_20251027_122310.json)
                parts = metric_file.stem.split('_')
                if len(parts) >= 3:
                    model = parts[2]  # Get the model name
                    if model not in models_data:
                        with open(metric_file, 'r') as f:
                            models_data[model] = {
                                'timestamp': '_'.join(parts[3:]),
                                'file': metric_file.name,
                                'data': json.load(f)
                            }

            if model_name:
                # Filter by specific model
                filtered = {k: v for k, v in models_data.items() if model_name.lower() in k.lower()}
                if not filtered:
                    return {
                        'status': 'error',
                        'message': f'No metrics found for model: {model_name}',
                        'available_models': list(models_data.keys())
                    }
                return {
                    'status': 'success',
                    'data': filtered,
                    'source': str(self.logs_dir)
                }
            else:
                return {
                    'status': 'success',
                    'data': models_data,
                    'source': str(self.logs_dir)
                }
        except Exception as e:
            logger.error(f"Error retrieving MLOps metrics: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_mlops_all_metrics(self) -> Dict[str, Any]:
        """
        Retrieve aggregated MLOps metrics from mlops_all_metrics_*.json files

        Returns:
            Aggregated MLOps metrics for all models
        """
        logger.info("Retrieving all MLOps metrics...")

        try:
            all_metrics_files = sorted(self.logs_dir.glob('mlops_all_metrics_*.json'), reverse=True)

            if not all_metrics_files:
                logger.warning("No aggregated MLOps metrics files found")
                return {
                    'status': 'error',
                    'message': 'No aggregated MLOps metrics files found'
                }

            latest_metrics_file = all_metrics_files[0]

            with open(latest_metrics_file, 'r') as f:
                data = json.load(f)

            return {
                'status': 'success',
                'latest_file': latest_metrics_file.name,
                'timestamp': latest_metrics_file.stem.split('_')[-1] if '_' in latest_metrics_file.stem else 'unknown',
                'data': data,
                'source': str(latest_metrics_file)
            }
        except Exception as e:
            logger.error(f"Error retrieving all MLOps metrics: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_mlops_dashboard_data(self) -> Dict[str, Any]:
        """
        Retrieve MLOps dashboard data from mlops_dashboard_data.json

        Returns:
            Dashboard data including model health and metrics
        """
        logger.info("Retrieving MLOps dashboard data...")

        try:
            dashboard_path = self.logs_dir / 'mlops_dashboard_data.json'

            if dashboard_path.exists():
                with open(dashboard_path, 'r') as f:
                    data = json.load(f)
                return {
                    'status': 'success',
                    'data': data,
                    'source': str(dashboard_path)
                }
            else:
                logger.warning("MLOps dashboard data file not found")
                return {
                    'status': 'error',
                    'message': 'MLOps dashboard data file not found',
                    'source': str(dashboard_path)
                }
        except Exception as e:
            logger.error(f"Error retrieving MLOps dashboard data: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_model_comparison(self) -> Dict[str, Any]:
        """
        Get comparison between all trained models

        Returns:
            Model comparison metrics
        """
        logger.info("Retrieving model comparison...")

        try:
            eval_data = self.get_evaluation_results()

            if eval_data['status'] != 'success':
                return eval_data

            models = eval_data['data']

            # Extract key metrics for comparison
            comparison = {}
            for model_name, metrics in models.items():
                comparison[model_name] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'roc_auc': metrics.get('roc_auc', 0),
                    'false_positive_rate': metrics.get('false_positive_rate', 0),
                    'false_negative_rate': metrics.get('false_negative_rate', 0),
                    'business_cost': metrics.get('business_metrics', {}).get('total_cost', 0)
                }

            # Determine best model
            best_model = max(comparison.items(), key=lambda x: x[1]['accuracy'])[0] if comparison else None

            return {
                'status': 'success',
                'best_model': best_model,
                'comparison': comparison,
                'timestamp': datetime.now().isoformat(),
                'source': str(self.logs_dir / 'evaluation_results.json')
            }
        except Exception as e:
            logger.error(f"Error retrieving model comparison: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_pipeline_logs(self, limit: int = 10) -> Dict[str, Any]:
        """
        Retrieve recent pipeline logs

        Args:
            limit: Number of recent logs to retrieve

        Returns:
            Recent pipeline execution logs
        """
        logger.info(f"Retrieving pipeline logs (limit: {limit})...")

        try:
            log_files = sorted(self.logs_dir.glob('ml_pipeline_*.log'), reverse=True)

            if not log_files:
                logger.warning("No pipeline log files found")
                return {
                    'status': 'error',
                    'message': 'No pipeline log files found'
                }

            recent_logs = []
            for log_file in log_files[:limit]:
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                    recent_logs.append({
                        'timestamp': log_file.stem.replace('ml_pipeline_', ''),
                        'file': log_file.name,
                        'size_kb': log_file.stat().st_size / 1024,
                        'preview': content[:500] if content else 'Empty log',
                        'full_content': content
                    })
                except Exception as e:
                    logger.warning(f"Error reading log file {log_file}: {str(e)}")

            return {
                'status': 'success',
                'count': len(recent_logs),
                'data': recent_logs,
                'source': str(self.logs_dir)
            }
        except Exception as e:
            logger.error(f"Error retrieving pipeline logs: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_model_health_status(self) -> Dict[str, Any]:
        """
        Get current model health status from dashboard data

        Returns:
            Model health information
        """
        logger.info("Retrieving model health status...")

        try:
            dashboard_data = self.get_mlops_dashboard_data()

            if dashboard_data['status'] != 'success':
                return dashboard_data

            models = dashboard_data['data'].get('models', [])

            health_status = {
                'status': 'success',
                'last_updated': dashboard_data['data'].get('last_updated'),
                'total_models': len(models),
                'models': []
            }

            for model in models:
                health_status['models'].append({
                    'name': model.get('model_name'),
                    'health_status': model.get('health_status'),
                    'health_score': model.get('health_score'),
                    'accuracy': model.get('accuracy'),
                    'f1_score': model.get('f1_score'),
                    'timestamp': model.get('timestamp')
                })

            return health_status
        except Exception as e:
            logger.error(f"Error retrieving model health status: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_all_ml_pipeline_details(self) -> Dict[str, Any]:
        """
        Get all ML pipeline information in one response

        Returns:
            Complete ML pipeline details
        """
        logger.info("Retrieving all ML pipeline details...")

        return {
            'timestamp': datetime.now().isoformat(),
            'evaluation_results': self.get_evaluation_results(),
            'feature_importance': self.get_feature_importance(),
            'model_comparison': self.get_model_comparison(),
            'mlops_metrics': self.get_mlops_metrics(),
            'mlops_all_metrics': self.get_mlops_all_metrics(),
            'mlops_dashboard': self.get_mlops_dashboard_data(),
            'model_health': self.get_model_health_status(),
            'pipeline_logs': self.get_pipeline_logs(limit=5)
        }

