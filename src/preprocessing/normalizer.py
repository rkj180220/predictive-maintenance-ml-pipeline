"""
Data Normalizer for NASA C-MAPSS Dataset
Author: ramkumarjayakumar
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    Normalize features for machine learning models
    Supports StandardScaler, MinMaxScaler, and RobustScaler
    """

    def __init__(self, config: Dict):
        """
        Initialize data normalizer

        Args:
            config: Configuration dictionary with normalization parameters
        """
        self.config = config
        self.scalers = {}
        self.normalization_method = config.get('normalization_method', 'standard')

    def fit_transform(self, df: pd.DataFrame, feature_cols: list,
                     dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Fit scaler and transform data

        Args:
            df: DataFrame to normalize
            feature_cols: List of columns to normalize
            dataset_name: Name for logging

        Returns:
            DataFrame with normalized features
        """
        logger.info(f"Fitting and transforming {dataset_name} data")
        logger.info(f"Normalization method: {self.normalization_method}")

        df_normalized = df.copy()

        # Create scaler
        scaler = self._create_scaler()

        # Fit and transform
        df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])

        # Store scaler for later use
        self.scalers[dataset_name] = scaler

        logger.info(f"Normalized {len(feature_cols)} features")

        return df_normalized

    def transform(self, df: pd.DataFrame, feature_cols: list,
                 dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Transform data using pre-fitted scaler

        Args:
            df: DataFrame to normalize
            feature_cols: List of columns to normalize
            dataset_name: Name of scaler to use

        Returns:
            DataFrame with normalized features
        """
        if dataset_name not in self.scalers:
            raise ValueError(f"No scaler found for {dataset_name}. Call fit_transform first.")

        logger.info(f"Transforming {dataset_name} data using pre-fitted scaler")

        df_normalized = df.copy()
        scaler = self.scalers[dataset_name]

        df_normalized[feature_cols] = scaler.transform(df[feature_cols])

        logger.info(f"Normalized {len(feature_cols)} features")

        return df_normalized

    def _create_scaler(self):
        """Create scaler based on configuration"""
        if self.normalization_method == 'standard':
            return StandardScaler()
        elif self.normalization_method == 'minmax':
            return MinMaxScaler()
        elif self.normalization_method == 'robust':
            return RobustScaler()
        else:
            logger.warning(f"Unknown normalization method: {self.normalization_method}. Using StandardScaler.")
            return StandardScaler()

    def save_scalers(self, filepath: Path):
        """
        Save fitted scalers to disk

        Args:
            filepath: Path to save scalers
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.scalers, filepath)
        logger.info(f"Saved scalers to {filepath}")

    def load_scalers(self, filepath: Path):
        """
        Load fitted scalers from disk

        Args:
            filepath: Path to load scalers from
        """
        self.scalers = joblib.load(filepath)
        logger.info(f"Loaded scalers from {filepath}")

    def get_feature_statistics(self, dataset_name: str) -> Optional[Dict]:
        """
        Get statistics from fitted scaler

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with scaler statistics
        """
        if dataset_name not in self.scalers:
            logger.warning(f"No scaler found for {dataset_name}")
            return None

        scaler = self.scalers[dataset_name]

        stats = {}

        if isinstance(scaler, StandardScaler):
            stats = {
                'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                'std': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
                'var': scaler.var_.tolist() if hasattr(scaler, 'var_') else None
            }
        elif isinstance(scaler, MinMaxScaler):
            stats = {
                'min': scaler.min_.tolist() if hasattr(scaler, 'min_') else None,
                'max': scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else None,
                'range': scaler.data_range_.tolist() if hasattr(scaler, 'data_range_') else None
            }
        elif isinstance(scaler, RobustScaler):
            stats = {
                'center': scaler.center_.tolist() if hasattr(scaler, 'center_') else None,
                'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
            }

        return stats


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from src.config.settings import PREPROCESSING_CONFIG

    normalizer = DataNormalizer(PREPROCESSING_CONFIG)
    print("Data Normalizer initialized successfully")

