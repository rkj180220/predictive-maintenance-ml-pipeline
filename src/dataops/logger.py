"""
Advanced Logger for Predictive Maintenance Pipeline
Author: ramkumarjayakumar
Date: 2025-10-18
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
import json
from typing import Optional

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


class PipelineLogger:
    """
    Advanced logging system for predictive maintenance pipeline
    Supports console and file logging with rotation
    """

    def __init__(self, name: str, log_dir: Path, config: dict):
        """
        Initialize pipeline logger

        Args:
            name: Logger name
            log_dir: Directory for log files
            config: Configuration dictionary
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup and configure logger"""
        logger = logging.getLogger(self.name)

        # Set level
        log_level = self.config.get('log_level', 'INFO')
        logger.setLevel(getattr(logging, log_level))

        # Remove existing handlers
        logger.handlers = []

        # Create formatters
        log_format = self.config.get('log_format',
                                     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler with color
        console_handler = self._create_console_handler(log_format)
        logger.addHandler(console_handler)

        # File handler with rotation
        file_handler = self._create_file_handler(log_format)
        logger.addHandler(file_handler)

        # JSON file handler for structured logs
        json_handler = self._create_json_handler()
        logger.addHandler(json_handler)

        return logger

    def _create_console_handler(self, log_format: str) -> logging.Handler:
        """Create console handler with optional colors"""
        console_handler = logging.StreamHandler(sys.stdout)

        if HAS_COLORLOG:
            # Colored formatter
            color_formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(color_formatter)
        else:
            # Regular formatter
            formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
            console_handler.setFormatter(formatter)

        return console_handler

    def _create_file_handler(self, log_format: str) -> logging.Handler:
        """Create rotating file handler"""
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"

        max_bytes = self.config.get('log_rotation_mb', 10) * 1024 * 1024
        backup_count = self.config.get('max_log_files', 10)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )

        formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        return file_handler

    def _create_json_handler(self) -> logging.Handler:
        """Create JSON file handler for structured logging"""
        json_log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}_structured.json"

        json_handler = logging.FileHandler(json_log_file)
        json_handler.setFormatter(JSONFormatter())

        return json_handler

    def get_logger(self) -> logging.Logger:
        """Get the configured logger"""
        return self.logger

    def log_pipeline_start(self, pipeline_name: str, config: dict = None):
        """Log pipeline start"""
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING PIPELINE: {pipeline_name}")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        if config:
            self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        self.logger.info("=" * 80)

    def log_pipeline_end(self, pipeline_name: str, status: str, duration: float):
        """Log pipeline end"""
        self.logger.info("=" * 80)
        self.logger.info(f"PIPELINE COMPLETED: {pipeline_name}")
        self.logger.info(f"Status: {status}")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info("=" * 80)

    def log_stage_start(self, stage_name: str):
        """Log pipeline stage start"""
        self.logger.info("-" * 80)
        self.logger.info(f"Starting Stage: {stage_name}")
        self.logger.info("-" * 80)

    def log_stage_end(self, stage_name: str, status: str, duration: float):
        """Log pipeline stage end"""
        self.logger.info(f"Stage '{stage_name}' completed: {status} ({duration:.2f}s)")
        self.logger.info("-" * 80)

    def log_metrics(self, metrics: dict, stage: str = None):
        """Log metrics"""
        prefix = f"[{stage}] " if stage else ""
        self.logger.info(f"{prefix}Metrics: {json.dumps(metrics, indent=2)}")

    def log_data_quality(self, quality_score: float, details: dict = None):
        """Log data quality metrics"""
        self.logger.info(f"Data Quality Score: {quality_score:.2%}")
        if details:
            self.logger.info(f"Quality Details: {json.dumps(details, indent=2)}")


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_pipeline_logging(log_dir: Path, config: dict) -> PipelineLogger:
    """
    Setup pipeline logging

    Args:
        log_dir: Directory for log files
        config: Configuration dictionary

    Returns:
        PipelineLogger instance
    """
    return PipelineLogger('predictive_maintenance', log_dir, config)


if __name__ == "__main__":
    from src.config.settings import LOGS_DIR, DATAOPS_CONFIG

    # Setup logger
    pipeline_logger = setup_pipeline_logging(LOGS_DIR, DATAOPS_CONFIG)
    logger = pipeline_logger.get_logger()

    # Test logging
    pipeline_logger.log_pipeline_start("Test Pipeline", {"test": True})
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    pipeline_logger.log_pipeline_end("Test Pipeline", "SUCCESS", 10.5)

    print("Pipeline Logger initialized and tested successfully")

