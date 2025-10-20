"""
Feature Engineer for NASA C-MAPSS Dataset
Author: ramkumarjayakumar
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for time-series turbofan engine data
    Creates rolling averages, lag features, statistical features, and trend indicators
    """

    def __init__(self, config: Dict):
        """
        Initialize feature engineer

        Args:
            config: Configuration dictionary with feature engineering parameters
        """
        self.config = config
        self.feature_stats = {}

    def engineer_features(self, df: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Perform comprehensive feature engineering

        Args:
            df: DataFrame with raw features
            dataset_name: Name for logging

        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Starting feature engineering for {dataset_name}")
        logger.info(f"Initial features: {len(df.columns)}")

        # Debug: Log available columns
        logger.info(f"Available columns: {df.columns.tolist()}")

        # Check if required columns exist
        required_cols = ['unit_id', 'time_cycles']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Required columns missing from dataframe: {missing_cols}")

        df_features = df.copy()

        # Sort by unit_id and time_cycles for proper time-series operations
        df_features = df_features.sort_values(['unit_id', 'time_cycles'])

        # 1. Rolling window features
        df_features = self._create_rolling_features(df_features)

        # 2. Lag features
        df_features = self._create_lag_features(df_features)

        # 3. Statistical features
        df_features = self._create_statistical_features(df_features)

        # 4. Trend features
        df_features = self._create_trend_features(df_features)

        # 5. Interaction features
        df_features = self._create_interaction_features(df_features)

        # 6. Time-based features
        df_features = self._create_time_features(df_features)

        logger.info(f"Feature engineering complete. Final features: {len(df_features.columns)}")
        logger.info(f"Added {len(df_features.columns) - len(df.columns)} new features")

        return df_features

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features with progress tracking"""
        logger.info("Creating rolling window features...")

        df_roll = df.copy()
        windows = self.config.get('rolling_windows', [5, 10, 20])
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]

        # Calculate total operations for progress bar
        total_ops = len(windows) * len(sensor_cols) * 2  # 2 stats per sensor per window

        # Collect all new features in a dictionary to avoid fragmentation
        new_features = {}

        logger.info(f"Creating {total_ops} rolling features across {len(windows)} window sizes...")

        with tqdm(total=total_ops, desc="Rolling Features", unit="feature") as pbar:
            for window in windows:
                for col in sensor_cols:
                    # Rolling mean
                    new_features[f'{col}_rolling_mean_{window}'] = (
                        df_roll.groupby('unit_id')[col]
                        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                    )
                    pbar.update(1)

                    # Rolling std
                    new_features[f'{col}_rolling_std_{window}'] = (
                        df_roll.groupby('unit_id')[col]
                        .transform(lambda x: x.rolling(window=window, min_periods=1).std())
                    )
                    pbar.update(1)

        # Concatenate all new features at once to avoid fragmentation
        df_roll = pd.concat([df_roll, pd.DataFrame(new_features, index=df_roll.index)], axis=1)

        logger.info(f"✓ Added {len(new_features)} rolling window features")
        return df_roll

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features with progress tracking"""
        logger.info("Creating lag features...")

        df_lag = df.copy()
        lags = self.config.get('lag_features', [1, 5, 10])
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]

        # Calculate total operations
        total_ops = len(lags) * len(sensor_cols)
        new_features = {}

        logger.info(f"Creating {total_ops} lag features across {len(lags)} lag periods...")

        with tqdm(total=total_ops, desc="Lag Features", unit="feature") as pbar:
            for lag in lags:
                for col in sensor_cols:
                    new_features[f'{col}_lag_{lag}'] = (
                        df_lag.groupby('unit_id')[col].shift(lag)
                    )
                    pbar.update(1)

        # Concatenate all new lag features at once
        df_lag = pd.concat([df_lag, pd.DataFrame(new_features, index=df_lag.index)], axis=1)

        # Fill initial NaN values from lag with backward fill (within each unit)
        logger.info("Filling NaN values in lag features...")
        for col in new_features.keys():
            df_lag[col] = df_lag.groupby('unit_id')[col].transform(lambda x: x.bfill())

        logger.info(f"✓ Added {len(new_features)} lag features")
        return df_lag

    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical aggregate features with progress tracking"""
        logger.info("Creating statistical features...")

        # Check if unit_id exists
        if 'unit_id' not in df.columns:
            logger.error("CRITICAL: 'unit_id' column is missing from dataframe!")
            raise ValueError("'unit_id' column not found in dataframe")

        df_stats = df.copy()
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]

        # Calculate total operations (4 stats per sensor)
        total_ops = len(sensor_cols) * 4
        new_features = {}

        logger.info(f"Creating {total_ops} statistical features for {len(sensor_cols)} sensors...")

        with tqdm(total=total_ops, desc="Statistical Features", unit="feature", disable=False) as pbar:
            for col in sensor_cols:
                # Vectorized cumulative operations - much faster
                grouped = df_stats.groupby('unit_id')[col]

                # Cumulative mean
                new_features[f'{col}_cumulative_mean'] = grouped.transform(lambda x: x.expanding().mean())
                pbar.update(1)

                # Cumulative std
                new_features[f'{col}_cumulative_std'] = grouped.transform(lambda x: x.expanding().std().fillna(0))
                pbar.update(1)

                # Min and max so far
                new_features[f'{col}_cumulative_min'] = grouped.transform(lambda x: x.expanding().min())
                pbar.update(1)

                new_features[f'{col}_cumulative_max'] = grouped.transform(lambda x: x.expanding().max())
                pbar.update(1)

        # Concatenate all new features at once
        df_stats = pd.concat([df_stats, pd.DataFrame(new_features, index=df_stats.index)], axis=1)

        logger.info(f"✓ Added {len(new_features)} statistical features")
        return df_stats

    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend indicators with progress tracking - OPTIMIZED VERSION"""
        logger.info("Creating trend features...")

        df_trend = df.copy()
        window = self.config.get('trend_window', 10)
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]

        # Only create 2 features per sensor instead of 3 (removed slow polyfit)
        total_ops = len(sensor_cols) * 2
        new_features = {}

        logger.info(f"Creating {total_ops} trend features for {len(sensor_cols)} sensors...")

        with tqdm(total=total_ops, desc="Trend Features", unit="feature") as pbar:
            for col in sensor_cols:
                # Rate of change (first derivative)
                rate_col = f'{col}_rate_of_change'
                grouped = df_trend.groupby('unit_id')[col]
                new_features[rate_col] = grouped.transform(lambda x: x.diff().fillna(0))
                pbar.update(1)

                # Acceleration (second derivative) - simplified
                new_features[f'{col}_acceleration'] = (
                    df_trend.groupby('unit_id')[col]
                    .transform(lambda x: x.diff().diff().fillna(0))
                )
                pbar.update(1)

        # Concatenate all new features at once
        df_trend = pd.concat([df_trend, pd.DataFrame(new_features, index=df_trend.index)], axis=1)

        logger.info(f"✓ Added {len(new_features)} trend features")
        return df_trend

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between sensors"""
        logger.info("Creating interaction features...")

        df_interact = df.copy()

        # Create ratios between key sensors
        # Example: sensor pairs that might be related
        sensor_pairs = [
            ('sensor_2', 'sensor_3'),
            ('sensor_4', 'sensor_11'),
            ('sensor_7', 'sensor_8'),
            ('sensor_12', 'sensor_13'),
        ]

        features_added = 0
        for sensor1, sensor2 in sensor_pairs:
            if sensor1 in df.columns and sensor2 in df.columns:
                # Ratio
                df_interact[f'{sensor1}_{sensor2}_ratio'] = (
                    df_interact[sensor1] / (df_interact[sensor2] + 1e-6)
                )

                # Difference
                df_interact[f'{sensor1}_{sensor2}_diff'] = (
                    df_interact[sensor1] - df_interact[sensor2]
                )

                features_added += 2

        logger.info(f"Added {features_added} interaction features")
        return df_interact

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        logger.info("Creating time-based features...")

        df_time = df.copy()

        features_added = 0

        # Normalized cycle (0 to 1 within each engine's life)
        max_cycles = df_time.groupby('unit_id')['time_cycles'].transform('max')
        df_time['cycle_normalized'] = df_time['time_cycles'] / max_cycles
        features_added += 1

        # Remaining cycles percentage (if RUL exists)
        if 'RUL' in df_time.columns:
            total_life = df_time['time_cycles'] + df_time['RUL']
            df_time['remaining_life_pct'] = df_time['RUL'] / total_life
            features_added += 1

        # Early, mid, late life indicator
        df_time['life_stage'] = pd.cut(
            df_time['cycle_normalized'],
            bins=[0, 0.33, 0.66, 1.0],
            labels=[0, 1, 2]
        ).astype(float)
        features_added += 1

        logger.info(f"Added {features_added} time-based features")
        return df_time

    def get_feature_importance_candidates(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of engineered features for importance analysis

        Args:
            df: DataFrame with features

        Returns:
            List of feature column names
        """
        # Exclude identifiers and target variables
        exclude_cols = ['unit_id', 'time_cycles', 'RUL', 'failure_label', 'dataset']

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        return feature_cols

    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary of engineered features

        Args:
            df: DataFrame with features

        Returns:
            Dictionary with feature summary
        """
        summary = {
            'total_features': len(df.columns),
            'sensor_features': len([col for col in df.columns if col.startswith('sensor_')]),
            'rolling_features': len([col for col in df.columns if 'rolling' in col]),
            'lag_features': len([col for col in df.columns if 'lag' in col]),
            'trend_features': len([col for col in df.columns if 'trend' in col or 'rate_of_change' in col]),
            'statistical_features': len([col for col in df.columns if 'cumulative' in col]),
            'interaction_features': len([col for col in df.columns if 'ratio' in col or 'diff' in col]),
            'time_features': len([col for col in df.columns if 'cycle' in col or 'life' in col])
        }

        return summary


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from src.config.settings import FEATURE_CONFIG

    engineer = FeatureEngineer(FEATURE_CONFIG)
    print("Feature Engineer initialized successfully")
