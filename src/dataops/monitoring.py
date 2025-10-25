"""
Monitoring Module for Predictive Maintenance Pipeline
Author: ramkumarjayakumar
Date: 2025-10-18
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path
import psutil
import time
import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types and booleans"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return int(obj) if isinstance(obj, np.integer) else float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        return super().default(obj)


class PipelineMonitor:
    """
    Monitor pipeline execution, performance, and data quality
    Track metrics and generate alerts
    """

    def __init__(self, config: Dict):
        """
        Initialize pipeline monitor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics_history = []
        self.alerts = []
        self.current_execution = None

    def start_execution(self, execution_id: str = None) -> str:
        """
        Start monitoring a pipeline execution

        Args:
            execution_id: Optional execution ID

        Returns:
            Execution ID
        """
        if execution_id is None:
            execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_execution = {
            'execution_id': execution_id,
            'start_time': datetime.now(),
            'end_time': None,
            'status': 'RUNNING',
            'stages': [],
            'metrics': {},
            'alerts': [],
            'resource_usage': {}
        }

        logger.info(f"Started monitoring execution: {execution_id}")

        return execution_id

    def end_execution(self, status: str = "SUCCESS"):
        """
        End monitoring of current execution

        Args:
            status: Execution status (SUCCESS, FAILED, etc.)
        """
        if self.current_execution is None:
            logger.warning("No execution to end")
            return

        self.current_execution['end_time'] = datetime.now()
        self.current_execution['status'] = status

        # Calculate duration
        duration = (self.current_execution['end_time'] -
                   self.current_execution['start_time']).total_seconds()
        self.current_execution['duration_seconds'] = duration

        # Check for performance alerts
        threshold = self.config.get('execution_time_threshold_seconds', 300)
        if duration > threshold:
            self.add_alert(
                'PERFORMANCE',
                f"Execution time ({duration:.2f}s) exceeded threshold ({threshold}s)"
            )

        # Save to history
        self.metrics_history.append(self.current_execution.copy())

        logger.info(f"Ended execution {self.current_execution['execution_id']}: {status} ({duration:.2f}s)")

        self.current_execution = None

    def start_stage(self, stage_name: str):
        """
        Start monitoring a pipeline stage

        Args:
            stage_name: Name of the stage
        """
        if self.current_execution is None:
            logger.warning("No active execution to add stage to")
            return

        stage = {
            'name': stage_name,
            'start_time': datetime.now(),
            'end_time': None,
            'status': 'RUNNING',
            'metrics': {}
        }

        self.current_execution['stages'].append(stage)

        logger.debug(f"Started monitoring stage: {stage_name}")

    def end_stage(self, stage_name: str, status: str = "SUCCESS", metrics: Dict = None):
        """
        End monitoring of a pipeline stage

        Args:
            stage_name: Name of the stage
            status: Stage status
            metrics: Optional metrics dictionary
        """
        if self.current_execution is None:
            logger.warning("No active execution")
            return

        # Find the stage
        stage = None
        for s in reversed(self.current_execution['stages']):
            if s['name'] == stage_name and s['end_time'] is None:
                stage = s
                break

        if stage is None:
            logger.warning(f"Stage {stage_name} not found")
            return

        stage['end_time'] = datetime.now()
        stage['status'] = status
        stage['duration_seconds'] = (stage['end_time'] - stage['start_time']).total_seconds()

        if metrics:
            stage['metrics'] = metrics

        logger.debug(f"Ended stage {stage_name}: {status} ({stage['duration_seconds']:.2f}s)")

    def record_metric(self, metric_name: str, value: float, stage: str = None):
        """
        Record a metric

        Args:
            metric_name: Name of the metric
            value: Metric value
            stage: Optional stage name
        """
        if self.current_execution is None:
            logger.warning("No active execution to record metric")
            return

        if stage:
            # Record to stage metrics
            for s in reversed(self.current_execution['stages']):
                if s['name'] == stage:
                    s['metrics'][metric_name] = value
                    break
        else:
            # Record to execution metrics
            self.current_execution['metrics'][metric_name] = value

        logger.debug(f"Recorded metric {metric_name}: {value}")

    def record_data_quality(self, quality_score: float, details: Dict = None):
        """
        Record data quality metrics

        Args:
            quality_score: Quality score (0-1)
            details: Optional quality details
        """
        self.record_metric('data_quality_score', quality_score)

        if details:
            # Use custom encoder to handle numpy types and booleans
            self.record_metric('data_quality_details', json.dumps(details, cls=NumpyEncoder))

        # Check quality threshold
        threshold = self.config.get('data_quality_threshold', 0.95)
        if quality_score < threshold:
            self.add_alert(
                'DATA_QUALITY',
                f"Data quality ({quality_score:.2%}) below threshold ({threshold:.2%})"
            )

    def record_resource_usage(self):
        """Record current resource usage"""
        if self.current_execution is None:
            return

        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            usage = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }

            if 'resource_snapshots' not in self.current_execution['resource_usage']:
                self.current_execution['resource_usage']['resource_snapshots'] = []

            self.current_execution['resource_usage']['resource_snapshots'].append(usage)

        except Exception as e:
            logger.warning(f"Failed to record resource usage: {e}")

    def add_alert(self, alert_type: str, message: str, severity: str = "WARNING"):
        """
        Add an alert

        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (INFO, WARNING, ERROR, CRITICAL)
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'severity': severity,
            'message': message
        }

        if self.current_execution:
            self.current_execution['alerts'].append(alert)

        self.alerts.append(alert)

        logger.log(
            getattr(logging, severity),
            f"ALERT [{alert_type}]: {message}"
        )

    def get_current_status(self) -> Dict:
        """
        Get current execution status

        Returns:
            Dictionary with current status
        """
        if self.current_execution is None:
            return {'status': 'IDLE', 'message': 'No active execution'}

        return {
            'execution_id': self.current_execution['execution_id'],
            'status': self.current_execution['status'],
            'start_time': self.current_execution['start_time'].isoformat(),
            'running_time_seconds': (datetime.now() -
                                    self.current_execution['start_time']).total_seconds(),
            'current_stage': self.current_execution['stages'][-1]['name']
                           if self.current_execution['stages'] else None,
            'completed_stages': len([s for s in self.current_execution['stages']
                                    if s['status'] != 'RUNNING']),
            'total_stages': len(self.current_execution['stages']),
            'alerts_count': len(self.current_execution['alerts'])
        }

    def get_execution_summary(self, execution_id: str = None) -> Optional[Dict]:
        """
        Get summary of a specific execution

        Args:
            execution_id: Execution ID (uses current if None)

        Returns:
            Execution summary dictionary
        """
        if execution_id is None:
            return self.current_execution

        # Search in history
        for execution in reversed(self.metrics_history):
            if execution['execution_id'] == execution_id:
                return execution

        return None

    def get_performance_summary(self) -> Dict:
        """
        Get performance summary across all executions

        Returns:
            Performance summary dictionary
        """
        if not self.metrics_history:
            return {'message': 'No execution history available'}

        durations = [e['duration_seconds'] for e in self.metrics_history
                    if 'duration_seconds' in e]

        successful = [e for e in self.metrics_history if e['status'] == 'SUCCESS']
        failed = [e for e in self.metrics_history if e['status'] != 'SUCCESS']

        return {
            'total_executions': len(self.metrics_history),
            'successful_executions': len(successful),
            'failed_executions': len(failed),
            'success_rate': len(successful) / len(self.metrics_history) if self.metrics_history else 0,
            'average_duration_seconds': sum(durations) / len(durations) if durations else 0,
            'min_duration_seconds': min(durations) if durations else 0,
            'max_duration_seconds': max(durations) if durations else 0,
            'total_alerts': len(self.alerts)
        }

    def export_metrics(self, filepath: Path):
        """
        Export metrics history to JSON file

        Args:
            filepath: Path to save metrics
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            'export_time': datetime.now().isoformat(),
            'executions': self.metrics_history,
            'alerts': self.alerts,
            'performance_summary': self.get_performance_summary()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f"Metrics exported to {filepath}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from src.config.settings import PIPELINE_METRICS

    monitor = PipelineMonitor(PIPELINE_METRICS)

    # Test monitoring
    execution_id = monitor.start_execution()
    monitor.start_stage("Test Stage")
    time.sleep(1)
    monitor.record_metric("test_metric", 42.0, stage="Test Stage")
    monitor.end_stage("Test Stage", "SUCCESS")
    monitor.end_execution("SUCCESS")

    print("\nPerformance Summary:")
    print(json.dumps(monitor.get_performance_summary(), indent=2))
