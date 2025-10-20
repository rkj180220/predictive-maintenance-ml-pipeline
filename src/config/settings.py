"""
Configuration settings for Predictive Maintenance ML Pipeline
Author: ramkumarjayakumar
Date: 2025-10-18
"""

import os
from pathlib import Path
from typing import List, Dict

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

# Logs Directory
LOGS_DIR = BASE_DIR / "logs"

# Notebooks Directory
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, LOGS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# NASA C-MAPSS Dataset Configuration
DATASET_CONFIG = {
    "kaggle_dataset": "behrad3d/nasa-cmaps",
    "datasets": ["FD002", "FD004"],
    "file_patterns": {
        "train": "train_{dataset}.txt",
        "test": "test_{dataset}.txt",
        "rul": "RUL_{dataset}.txt"
    }
}

# Column Names for NASA C-MAPSS Dataset
COLUMN_NAMES = [
    'unit_id', 'time_cycles', 'setting_1', 'setting_2', 'setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

# Sensor columns only
SENSOR_COLUMNS = [f'sensor_{i}' for i in range(1, 22)]

# Operational settings columns
SETTING_COLUMNS = ['setting_1', 'setting_2', 'setting_3']

# Feature Engineering Configuration
FEATURE_CONFIG = {
    "rolling_windows": [5, 20],  # Reduced from [5, 10, 20] - only short and medium term
    "lag_features": [1, 10],  # Reduced from [1, 5, 10] - only immediate and delayed
    "statistical_features": ["mean", "std"],  # Reduced from ["mean", "std", "min", "max"]
    "trend_window": 10,
    "enable_interaction_features": True,
    "max_sensors_for_interaction": 4  # Reduced from 6
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    "missing_value_threshold": 0.5,  # Drop columns with >50% missing values
    "outlier_method": "iqr",  # 'iqr' or 'zscore'
    "outlier_threshold": 3,
    "normalization_method": "standard",  # 'standard', 'minmax', 'robust'
    "variance_threshold": 0.01  # Drop low variance features
}

# EDA Configuration
EDA_CONFIG = {
    "correlation_threshold": 0.7,  # High correlation threshold
    "n_bins": 10,  # Number of bins for continuous variables
    "figure_size": (12, 8),
    "plot_style": "seaborn-v0_8-darkgrid",
    "color_palette": "viridis"
}

# DataOps Configuration
DATAOPS_CONFIG = {
    "pipeline_schedule_minutes": 2,  # Run every 2 minutes
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "max_log_files": 10,
    "log_rotation_mb": 10
}

# AWS Configuration
AWS_CONFIG = {
    "region": "us-east-1",
    "s3_bucket": "predictive-maintenance-data",
    "cloudwatch_log_group": "/aws/predictive-maintenance/pipeline",
    "cloudwatch_log_stream": "pipeline-execution"
}

# Model Configuration (for future use)
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "failure_threshold": 30  # RUL threshold for failure classification
}

# Business KPIs
BUSINESS_KPIS = {
    "target_precision": 0.85,
    "target_recall": 0.90,
    "target_f1_score": 0.87,
    "max_false_negatives": 0.05,  # Maximum acceptable false negative rate
    "maintenance_cost_savings_target": 0.30,  # 30% cost reduction target
    "downtime_reduction_target": 0.40  # 40% downtime reduction target
}

# Pipeline Execution Tracking
PIPELINE_METRICS = {
    "execution_time_threshold_seconds": 300,  # Alert if pipeline takes >5 minutes
    "data_quality_threshold": 0.95,  # Minimum data quality score
    "feature_count_min": 20,  # Minimum number of features after engineering
    "record_count_min": 1000  # Minimum records for processing
}
