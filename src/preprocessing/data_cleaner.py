"""
Data Cleaner for NASA C-MAPSS Dataset
Author: ramkumarjayakumar
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from scipy import stats

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Comprehensive data cleaning for NASA C-MAPSS dataset
    Handles missing values, outliers, and data quality issues
    """

    def __init__(self, config: Dict):
        """
        Initialize data cleaner

        Args:
            config: Configuration dictionary with cleaning parameters
        """
        self.config = config
        self.cleaning_stats = {}

    def clean_dataset(self, df: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Perform comprehensive data cleaning

        Args:
            df: DataFrame to clean
            dataset_name: Name for logging purposes

        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting data cleaning for {dataset_name}")
        logger.info(f"Initial shape: {df.shape}")

        df_cleaned = df.copy()

        # 1. Handle missing values
        df_cleaned = self._handle_missing_values(df_cleaned, dataset_name)

        # 2. Remove duplicate rows
        df_cleaned = self._remove_duplicates(df_cleaned, dataset_name)

        # 3. Handle outliers
        df_cleaned = self._handle_outliers(df_cleaned, dataset_name)

        # 4. Remove low variance features
        df_cleaned = self._remove_low_variance_features(df_cleaned, dataset_name)

        # 5. Ensure data consistency
        df_cleaned = self._ensure_data_consistency(df_cleaned, dataset_name)

        logger.info(f"Cleaning complete. Final shape: {df_cleaned.shape}")

        return df_cleaned

    def _handle_missing_values(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Handle missing values in the dataset

        Args:
            df: DataFrame
            dataset_name: Name for logging

        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values...")

        missing_before = df.isnull().sum().sum()

        if missing_before == 0:
            logger.info("No missing values found")
            return df

        df_clean = df.copy()

        # Get missing value statistics
        missing_stats = {}
        for col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(df_clean) * 100
                missing_stats[col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2)
                }

                # Drop column if too many missing values
                threshold = self.config.get('missing_value_threshold', 0.5)
                if missing_pct > threshold * 100:
                    logger.warning(f"Dropping column {col} ({missing_pct:.2f}% missing)")
                    df_clean = df_clean.drop(col, axis=1)
                else:
                    # Impute numeric columns with median
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        median_value = df_clean[col].median()
                        df_clean[col].fillna(median_value, inplace=True)
                        logger.info(f"Imputed {col} with median: {median_value:.2f}")
                    else:
                        # Impute categorical with mode
                        mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                        df_clean[col].fillna(mode_value, inplace=True)
                        logger.info(f"Imputed {col} with mode: {mode_value}")

        missing_after = df_clean.isnull().sum().sum()

        self.cleaning_stats[f'{dataset_name}_missing_values'] = {
            'missing_before': int(missing_before),
            'missing_after': int(missing_after),
            'columns_affected': missing_stats
        }

        logger.info(f"Missing values: {missing_before} → {missing_after}")

        return df_clean

    def _remove_duplicates(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Remove duplicate rows

        Args:
            df: DataFrame
            dataset_name: Name for logging

        Returns:
            DataFrame with duplicates removed
        """
        logger.info("Checking for duplicates...")

        duplicates_before = df.duplicated().sum()

        if duplicates_before == 0:
            logger.info("No duplicates found")
            return df

        df_clean = df.drop_duplicates()
        duplicates_removed = duplicates_before - df_clean.duplicated().sum()

        self.cleaning_stats[f'{dataset_name}_duplicates'] = {
            'duplicates_found': int(duplicates_before),
            'duplicates_removed': int(duplicates_removed)
        }

        logger.info(f"Removed {duplicates_removed} duplicate rows")

        return df_clean

    def _handle_outliers(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Detect and handle outliers using IQR or Z-score method

        Args:
            df: DataFrame
            dataset_name: Name for logging

        Returns:
            DataFrame with outliers handled
        """
        logger.info("Handling outliers...")

        df_clean = df.copy()
        outlier_method = self.config.get('outlier_method', 'iqr')
        threshold = self.config.get('outlier_threshold', 3)

        # Get numeric columns (excluding identifiers and labels)
        exclude_cols = ['unit_id', 'time_cycles', 'RUL', 'failure_label', 'dataset']
        numeric_cols = [col for col in df_clean.select_dtypes(include=[np.number]).columns
                       if col not in exclude_cols]

        outlier_stats = {}

        for col in numeric_cols:
            if outlier_method == 'iqr':
                # IQR method
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            else:
                # Z-score method
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                outliers = pd.Series(False, index=df_clean.index)
                outliers.loc[df_clean[col].notna()] = z_scores > threshold

            outlier_count = outliers.sum()

            if outlier_count > 0:
                outlier_pct = outlier_count / len(df_clean) * 100
                outlier_stats[col] = {
                    'count': int(outlier_count),
                    'percentage': round(outlier_pct, 2)
                }

                # Cap outliers at bounds instead of removing (preserve data)
                if outlier_method == 'iqr':
                    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                    logger.debug(f"Capped {outlier_count} outliers in {col}")

        self.cleaning_stats[f'{dataset_name}_outliers'] = outlier_stats

        logger.info(f"Handled outliers in {len(outlier_stats)} columns")

        return df_clean

    def _remove_low_variance_features(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Remove features with very low variance

        Args:
            df: DataFrame
            dataset_name: Name for logging

        Returns:
            DataFrame with low variance features removed
        """
        logger.info("Checking for low variance features...")

        df_clean = df.copy()
        variance_threshold = self.config.get('variance_threshold', 0.01)

        # Get numeric columns (excluding identifiers and labels)
        exclude_cols = ['unit_id', 'time_cycles', 'RUL', 'failure_label', 'dataset']
        numeric_cols = [col for col in df_clean.select_dtypes(include=[np.number]).columns
                       if col not in exclude_cols]

        low_variance_cols = []

        for col in numeric_cols:
            variance = df_clean[col].var()

            # Normalize variance by mean to get coefficient of variation
            if df_clean[col].mean() != 0:
                cv = df_clean[col].std() / abs(df_clean[col].mean())
            else:
                cv = 0

            if cv < variance_threshold:
                low_variance_cols.append(col)
                logger.info(f"Removing low variance column: {col} (CV={cv:.6f})")

        if low_variance_cols:
            df_clean = df_clean.drop(low_variance_cols, axis=1)

        self.cleaning_stats[f'{dataset_name}_low_variance'] = {
            'columns_removed': low_variance_cols,
            'count': len(low_variance_cols)
        }

        logger.info(f"Removed {len(low_variance_cols)} low variance features")

        return df_clean

    def _ensure_data_consistency(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Ensure data consistency and fix any logical issues

        Args:
            df: DataFrame
            dataset_name: Name for logging

        Returns:
            DataFrame with consistency fixes applied
        """
        logger.info("Ensuring data consistency...")

        df_clean = df.copy()

        # Ensure time_cycles is positive
        if 'time_cycles' in df_clean.columns:
            negative_cycles = (df_clean['time_cycles'] < 0).sum()
            if negative_cycles > 0:
                logger.warning(f"Found {negative_cycles} negative time_cycles, setting to 0")
                df_clean.loc[df_clean['time_cycles'] < 0, 'time_cycles'] = 0

        # Ensure RUL is non-negative
        if 'RUL' in df_clean.columns:
            negative_rul = (df_clean['RUL'] < 0).sum()
            if negative_rul > 0:
                logger.warning(f"Found {negative_rul} negative RUL values, setting to 0")
                df_clean.loc[df_clean['RUL'] < 0, 'RUL'] = 0

        # Ensure unit_id is positive integer
        if 'unit_id' in df_clean.columns:
            df_clean['unit_id'] = df_clean['unit_id'].astype(int)

        # Replace infinite values with NaN, then handle
        inf_count = np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            logger.warning(f"Found {inf_count} infinite values, replacing with NaN")
            df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Re-handle missing values
            df_clean = self._handle_missing_values(df_clean, dataset_name)

        logger.info("Data consistency checks complete")

        return df_clean

    def get_cleaning_report(self) -> str:
        """
        Generate cleaning report

        Returns:
            String report of cleaning operations
        """
        report = f"""
{'='*80}
DATA CLEANING REPORT
{'='*80}

"""
        for key, stats in self.cleaning_stats.items():
            report += f"\n{key}:\n"
            for stat_key, stat_value in stats.items():
                report += f"  {stat_key}: {stat_value}\n"

        return report


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from src.config.settings import PREPROCESSING_CONFIG

    cleaner = DataCleaner(PREPROCESSING_CONFIG)
    print("Data Cleaner initialized successfully")

