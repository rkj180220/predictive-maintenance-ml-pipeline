"""
Data Pipeline Service - Retrieve data pipeline information and metrics from logs
Author: ramkumarjayakumar
Date: 2025-10-27
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class DataPipelineService:
    """
    Service to retrieve data pipeline information and analytics from logs
    """

    def __init__(self, project_root: Path):
        """
        Initialize data pipeline service

        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.logs_dir = self.project_root / 'logs'
        self.data_dir = self.project_root / 'data'
        logger.info(f"Initializing DataPipelineService with project_root: {self.project_root}")

    def get_correlation_analysis(self) -> Dict[str, Any]:
        """
        Retrieve correlation analysis data from correlation_analysis.json

        Returns:
            Correlation analysis including feature correlations
        """
        logger.info("Retrieving correlation analysis...")

        try:
            correlation_path = self.logs_dir / 'correlation_analysis.json'

            if correlation_path.exists():
                with open(correlation_path, 'r') as f:
                    data = json.load(f)
                return {
                    'status': 'success',
                    'data': data,
                    'source': str(correlation_path)
                }
            else:
                logger.warning("Correlation analysis file not found")
                return {
                    'status': 'error',
                    'message': 'Correlation analysis file not found',
                    'source': str(correlation_path)
                }
        except Exception as e:
            logger.error(f"Error retrieving correlation analysis: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_statistical_analysis(self) -> Dict[str, Any]:
        """
        Retrieve statistical analysis data from statistical_analysis.json

        Returns:
            Statistical analysis with distribution stats
        """
        logger.info("Retrieving statistical analysis...")

        try:
            stats_path = self.logs_dir / 'statistical_analysis.json'

            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    data = json.load(f)
                return {
                    'status': 'success',
                    'data': data,
                    'source': str(stats_path)
                }
            else:
                logger.warning("Statistical analysis file not found")
                return {
                    'status': 'error',
                    'message': 'Statistical analysis file not found',
                    'source': str(stats_path)
                }
        except Exception as e:
            logger.error(f"Error retrieving statistical analysis: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_data_validation_report(self) -> Dict[str, Any]:
        """
        Retrieve data validation report from data_validation_report.txt

        Returns:
            Data validation details
        """
        logger.info("Retrieving data validation report...")

        try:
            report_path = self.logs_dir / 'data_validation_report.txt'

            if report_path.exists():
                with open(report_path, 'r') as f:
                    content = f.read()
                return {
                    'status': 'success',
                    'report': content,
                    'source': str(report_path)
                }
            else:
                logger.warning("Data validation report file not found")
                return {
                    'status': 'error',
                    'message': 'Data validation report file not found',
                    'source': str(report_path)
                }
        except Exception as e:
            logger.error(f"Error retrieving data validation report: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_data_drift_analysis(self) -> Dict[str, Any]:
        """
        Retrieve latest data drift analysis from data_drift_*.json files

        Returns:
            Data drift information
        """
        logger.info("Retrieving data drift analysis...")

        try:
            drift_files = sorted(self.logs_dir.glob('data_drift_*.json'), reverse=True)

            if not drift_files:
                logger.warning("No data drift files found")
                return {
                    'status': 'error',
                    'message': 'No data drift files found'
                }

            latest_drift_file = drift_files[0]

            with open(latest_drift_file, 'r') as f:
                data = json.load(f)

            return {
                'status': 'success',
                'data': data,
                'latest_file': latest_drift_file.name,
                'source': str(latest_drift_file)
            }
        except Exception as e:
            logger.error(f"Error retrieving data drift analysis: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_all_data_drift_history(self) -> Dict[str, Any]:
        """
        Retrieve all data drift analysis history

        Returns:
            List of all data drift analyses
        """
        logger.info("Retrieving data drift history...")

        try:
            drift_files = sorted(self.logs_dir.glob('data_drift_*.json'), reverse=True)

            if not drift_files:
                logger.warning("No data drift files found")
                return {
                    'status': 'error',
                    'message': 'No data drift files found',
                    'count': 0
                }

            drift_history = []
            for drift_file in drift_files:
                try:
                    with open(drift_file, 'r') as f:
                        data = json.load(f)
                    drift_history.append({
                        'timestamp': drift_file.stem.replace('data_drift_', ''),
                        'file': drift_file.name,
                        'data': data
                    })
                except Exception as e:
                    logger.warning(f"Error reading drift file {drift_file}: {str(e)}")

            return {
                'status': 'success',
                'count': len(drift_history),
                'data': drift_history,
                'source': str(self.logs_dir)
            }
        except Exception as e:
            logger.error(f"Error retrieving data drift history: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_pipeline_execution_metrics(self) -> Dict[str, Any]:
        """
        Retrieve pipeline execution metrics from pipeline_metrics.json

        Returns:
            Pipeline execution and performance metrics
        """
        logger.info("Retrieving pipeline execution metrics...")

        try:
            metrics_path = self.logs_dir / 'pipeline_metrics.json'

            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                return {
                    'status': 'success',
                    'data': data,
                    'source': str(metrics_path)
                }
            else:
                logger.warning("Pipeline metrics file not found")
                return {
                    'status': 'error',
                    'message': 'Pipeline metrics file not found',
                    'source': str(metrics_path)
                }
        except Exception as e:
            logger.error(f"Error retrieving pipeline metrics: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_data_quality_summary(self) -> Dict[str, Any]:
        """
        Get a summary of data quality metrics

        Returns:
            Data quality summary
        """
        logger.info("Retrieving data quality summary...")

        try:
            stats_data = self.get_statistical_analysis()
            validation_data = self.get_data_validation_report()
            drift_data = self.get_data_drift_analysis()

            summary = {
                'status': 'success',
                'statistical_analysis': 'Available' if stats_data['status'] == 'success' else 'Not Available',
                'validation_report': 'Available' if validation_data['status'] == 'success' else 'Not Available',
                'drift_analysis': 'Available' if drift_data['status'] == 'success' else 'Not Available',
                'timestamp': datetime.now().isoformat(),
                'details': {
                    'statistical_analysis': stats_data,
                    'validation_report': validation_data,
                    'drift_analysis': drift_data
                }
            }

            return summary
        except Exception as e:
            logger.error(f"Error retrieving data quality summary: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_all_data_pipeline_details(self) -> Dict[str, Any]:
        """
        Get all data pipeline information in one response

        Returns:
            Complete data pipeline details
        """
        logger.info("Retrieving all data pipeline details...")

        return {
            'timestamp': datetime.now().isoformat(),
            'correlation_analysis': self.get_correlation_analysis(),
            'statistical_analysis': self.get_statistical_analysis(),
            'data_validation': self.get_data_validation_report(),
            'data_drift_latest': self.get_data_drift_analysis(),
            'pipeline_metrics': self.get_pipeline_execution_metrics(),
            'data_quality_summary': self.get_data_quality_summary()
        }

