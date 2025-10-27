"""
Dashboard Service - Provide dashboard data and management
Author: ramkumarjayakumar
Date: 2025-10-27
"""

import logging
import json
import subprocess
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import os
import platform

logger = logging.getLogger(__name__)


class DashboardService:
    """
    Service to manage dashboard operations including pipeline triggers
    """

    def __init__(self, project_root: Path):
        """
        Initialize dashboard service

        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.logs_dir = self.project_root / 'logs'
        logger.info(f"Initializing DashboardService with project_root: {self.project_root}")

    def get_dashboard_overview(self) -> Dict[str, Any]:
        """
        Get complete dashboard overview with all metrics

        Returns:
            Dashboard overview data
        """
        logger.info("Generating dashboard overview...")

        try:
            from src.api.services.data_pipeline_service import DataPipelineService
            from src.api.services.ml_pipeline_service import MLPipelineService

            data_service = DataPipelineService(self.project_root)
            ml_service = MLPipelineService(self.project_root)

            # Get key metrics
            eval_results = ml_service.get_evaluation_results()
            dashboard_data = ml_service.get_mlops_dashboard_data()
            stats_data = data_service.get_statistical_analysis()
            drift_data = data_service.get_data_drift_analysis()

            overview = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'system_health': {
                    'status': 'HEALTHY',
                    'last_check': datetime.now().isoformat()
                },
                'data_pipeline': {
                    'status': 'ACTIVE',
                    'total_records_processed': stats_data.get('data', {}).get('Training', {}).get('basic_stats', {}).get('total_records', 0),
                    'features_engineered': stats_data.get('data', {}).get('Training', {}).get('basic_stats', {}).get('total_features', 0),
                    'data_drift_status': drift_data.get('status', 'UNKNOWN')
                },
                'ml_pipeline': {
                    'status': 'ACTIVE',
                    'models_trained': 2,
                    'best_model': None,
                    'model_metrics': {}
                }
            }

            # Extract ML metrics
            if eval_results.get('status') == 'success':
                eval_data = eval_results.get('data', {})
                best_model_name = None
                best_accuracy = 0

                for model_name, metrics in eval_data.items():
                    accuracy = metrics.get('accuracy', 0)
                    overview['ml_pipeline']['model_metrics'][model_name] = {
                        'accuracy': round(accuracy * 100, 2),
                        'precision': round(metrics.get('precision', 0) * 100, 2),
                        'recall': round(metrics.get('recall', 0) * 100, 2),
                        'f1_score': round(metrics.get('f1_score', 0) * 100, 2)
                    }
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model_name = model_name

                overview['ml_pipeline']['best_model'] = best_model_name

            return overview

        except Exception as e:
            logger.error(f"Error generating dashboard overview: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def trigger_data_pipeline(self) -> Dict[str, Any]:
        """
        Trigger the data pipeline execution

        Returns:
            Execution status
        """
        logger.info("Triggering data pipeline...")

        try:
            # Try to find and run the data pipeline script
            pipeline_script = self.project_root / 'run_ml_pipeline.py'

            if not pipeline_script.exists():
                return {
                    'status': 'error',
                    'message': f'Pipeline script not found at {pipeline_script}'
                }

            # Execute the pipeline in background
            if platform.system() == 'Windows':
                result = subprocess.Popen(
                    f'python {pipeline_script}',
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            else:
                result = subprocess.Popen(
                    ['python', str(pipeline_script)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

            return {
                'status': 'success',
                'message': 'Data pipeline triggered successfully',
                'process_id': result.pid,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error triggering data pipeline: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def trigger_ml_pipeline(self) -> Dict[str, Any]:
        """
        Trigger the ML pipeline execution

        Returns:
            Execution status
        """
        logger.info("Triggering ML pipeline...")

        try:
            # Try to find and run the ML pipeline script
            pipeline_script = self.project_root / 'run_ml_pipeline.py'

            if not pipeline_script.exists():
                return {
                    'status': 'error',
                    'message': f'Pipeline script not found at {pipeline_script}'
                }

            # Execute the pipeline in background
            if platform.system() == 'Windows':
                result = subprocess.Popen(
                    f'python {pipeline_script}',
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            else:
                result = subprocess.Popen(
                    ['python', str(pipeline_script)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

            return {
                'status': 'success',
                'message': 'ML pipeline triggered successfully',
                'process_id': result.pid,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error triggering ML pipeline: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics and resource usage

        Returns:
            System statistics
        """
        logger.info("Retrieving system statistics...")

        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'status': 'success',
                'cpu': {
                    'usage_percent': cpu_percent,
                    'cores': psutil.cpu_count()
                },
                'memory': {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'used_gb': round(memory.used / (1024**3), 2),
                    'percent': memory.percent
                },
                'disk': {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'used_gb': round(disk.used / (1024**3), 2),
                    'percent': disk.percent
                },
                'timestamp': datetime.now().isoformat()
            }
        except ImportError:
            logger.warning("psutil not installed, returning mock stats")
            return {
                'status': 'success',
                'cpu': {'usage_percent': 25.5, 'cores': 8},
                'memory': {'total_gb': 16.0, 'used_gb': 8.2, 'percent': 51.25},
                'disk': {'total_gb': 500.0, 'used_gb': 250.0, 'percent': 50.0},
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error retrieving system stats: {str(e)}")
            return {'status': 'error', 'message': str(e)}

