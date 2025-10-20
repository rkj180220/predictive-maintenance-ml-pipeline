"""
Data Loader for NASA C-MAPSS Dataset
Author: ramkumarjayakumar
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import glob

logger = logging.getLogger(__name__)


class CMAPSSDataLoader:
    """
    Load and prepare NASA C-MAPSS turbofan engine dataset
    Handles FD002 and FD004 datasets
    """

    def __init__(self, data_dir: Path, column_names: list):
        """
        Initialize data loader

        Args:
            data_dir: Directory containing raw data files
            column_names: List of column names for the dataset
        """
        self.data_dir = Path(data_dir)
        self.column_names = column_names

    def load_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train, test, and RUL data for a specific dataset

        Args:
            dataset_name: Name of dataset (e.g., 'FD002', 'FD004')

        Returns:
            Tuple of (train_df, test_df, rul_df)
        """
        logger.info(f"Loading {dataset_name} dataset...")

        # Load training data
        train_file = self.data_dir / f"train_{dataset_name}.txt"
        train_df = self._load_txt_file(train_file)
        logger.info(f"Loaded {len(train_df)} training records from {dataset_name}")

        # Load test data
        test_file = self.data_dir / f"test_{dataset_name}.txt"
        test_df = self._load_txt_file(test_file)
        logger.info(f"Loaded {len(test_df)} test records from {dataset_name}")

        # Load RUL (Remaining Useful Life) data
        rul_file = self.data_dir / f"RUL_{dataset_name}.txt"
        rul_df = self._load_rul_file(rul_file)
        logger.info(f"Loaded {len(rul_df)} RUL values from {dataset_name}")

        return train_df, test_df, rul_df

    def _load_txt_file(self, filepath: Path) -> pd.DataFrame:
        """
        Load space-delimited txt file

        Args:
            filepath: Path to the file

        Returns:
            DataFrame with loaded data
        """
        try:
            # Read space-delimited file
            df = pd.read_csv(
                filepath,
                sep=r'\s+',  # One or more spaces
                header=None,
                names=self.column_names
            )

            logger.debug(f"Loaded {len(df)} rows from {filepath.name}")
            return df

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise

    def _load_rul_file(self, filepath: Path) -> pd.DataFrame:
        """
        Load RUL (Remaining Useful Life) file

        Args:
            filepath: Path to RUL file

        Returns:
            DataFrame with RUL values
        """
        try:
            # RUL file has one value per line (one per test engine)
            df = pd.read_csv(
                filepath,
                sep=r'\s+',
                header=None,
                names=['RUL']
            )

            # Add unit_id (1-based index)
            df['unit_id'] = range(1, len(df) + 1)

            logger.debug(f"Loaded {len(df)} RUL values from {filepath.name}")
            return df

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise

    def load_all_datasets(self) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Load all available datasets (FD002, FD004)

        Returns:
            Dictionary with dataset names as keys and (train, test, rul) tuples as values
        """
        datasets = {}
        dataset_names = ['FD002', 'FD004']

        for dataset_name in dataset_names:
            try:
                train_df, test_df, rul_df = self.load_dataset(dataset_name)
                datasets[dataset_name] = (train_df, test_df, rul_df)
                logger.info(f"Successfully loaded {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")

        logger.info(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
        return datasets

    def add_rul_to_train(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add RUL (Remaining Useful Life) column to training data

        Args:
            train_df: Training dataframe

        Returns:
            DataFrame with RUL column added
        """
        logger.info("Calculating RUL for training data...")

        # Calculate max cycle for each engine
        max_cycles = train_df.groupby('unit_id')['time_cycles'].max().reset_index()
        max_cycles.columns = ['unit_id', 'max_cycle']

        # Merge max cycles
        df = train_df.merge(max_cycles, on='unit_id', how='left')

        # Calculate RUL: max_cycle - current_cycle
        df['RUL'] = df['max_cycle'] - df['time_cycles']

        # Drop temporary column
        df = df.drop('max_cycle', axis=1)

        logger.info("RUL calculated for training data")
        return df

    def add_rul_to_test(self, test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add RUL (Remaining Useful Life) column to test data

        Args:
            test_df: Test dataframe
            rul_df: DataFrame with true RUL values

        Returns:
            DataFrame with RUL column added
        """
        logger.info("Adding RUL to test data...")

        # Get max cycle for each test engine
        max_cycles = test_df.groupby('unit_id')['time_cycles'].max().reset_index()
        max_cycles.columns = ['unit_id', 'max_cycle']

        # Merge with test data
        df = test_df.merge(max_cycles, on='unit_id', how='left')

        # Merge with true RUL values
        df = df.merge(rul_df[['unit_id', 'RUL']], on='unit_id', how='left', suffixes=('', '_true'))

        # Calculate RUL for each cycle: RUL_true + (max_cycle - current_cycle)
        df['RUL'] = df['RUL'] + (df['max_cycle'] - df['time_cycles'])

        # Drop temporary column
        df = df.drop('max_cycle', axis=1)

        logger.info("RUL added to test data")
        return df

    def create_failure_labels(self, df: pd.DataFrame, threshold: int = 30) -> pd.DataFrame:
        """
        Create binary failure labels based on RUL threshold

        Args:
            df: DataFrame with RUL column
            threshold: RUL threshold for failure classification

        Returns:
            DataFrame with failure label column
        """
        logger.info(f"Creating failure labels with threshold={threshold}")

        # Create binary label: 1 if RUL <= threshold, 0 otherwise
        df['failure_label'] = (df['RUL'] <= threshold).astype(int)

        failure_count = df['failure_label'].sum()
        failure_pct = failure_count / len(df) * 100

        logger.info(f"Failure cases: {failure_count:,} ({failure_pct:.2f}%)")

        return df

    def merge_datasets(self, datasets: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge multiple datasets into single train and test dataframes

        Args:
            datasets: Dictionary of datasets

        Returns:
            Tuple of (merged_train, merged_test)
        """
        logger.info("Merging datasets...")

        train_dfs = []
        test_dfs = []

        for dataset_name, (train_df, test_df, rul_df) in datasets.items():
            # Add RUL to training data
            train_with_rul = self.add_rul_to_train(train_df)

            # Add RUL to test data
            test_with_rul = self.add_rul_to_test(test_df, rul_df)

            # Add dataset identifier
            train_with_rul['dataset'] = dataset_name
            test_with_rul['dataset'] = dataset_name

            # Adjust unit_id to be unique across datasets
            if dataset_name == 'FD004':
                max_train_id = train_with_rul['unit_id'].max()
                max_test_id = test_with_rul['unit_id'].max()

                # Offset by 1000 for FD004
                train_with_rul['unit_id'] = train_with_rul['unit_id'] + 1000
                test_with_rul['unit_id'] = test_with_rul['unit_id'] + 1000

            train_dfs.append(train_with_rul)
            test_dfs.append(test_with_rul)

        # Concatenate all datasets
        merged_train = pd.concat(train_dfs, ignore_index=True)
        merged_test = pd.concat(test_dfs, ignore_index=True)

        logger.info(f"Merged training data: {len(merged_train):,} records")
        logger.info(f"Merged test data: {len(merged_test):,} records")

        return merged_train, merged_test

    def get_dataset_summary(self, df: pd.DataFrame, dataset_type: str = "Dataset") -> Dict:
        """
        Get summary statistics for a dataset

        Args:
            df: DataFrame to summarize
            dataset_type: Type of dataset (for reporting)

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "dataset_type": dataset_type,
            "total_records": len(df),
            "total_engines": df['unit_id'].nunique(),
            "total_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        }

        if 'RUL' in df.columns:
            summary['rul_statistics'] = {
                "min": float(df['RUL'].min()),
                "max": float(df['RUL'].max()),
                "mean": float(df['RUL'].mean()),
                "median": float(df['RUL'].median()),
                "std": float(df['RUL'].std())
            }

        if 'time_cycles' in df.columns:
            summary['cycles_per_engine'] = {
                "min": int(df.groupby('unit_id')['time_cycles'].max().min()),
                "max": int(df.groupby('unit_id')['time_cycles'].max().max()),
                "mean": float(df.groupby('unit_id')['time_cycles'].max().mean()),
                "median": float(df.groupby('unit_id')['time_cycles'].max().median())
            }

        return summary


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    from pathlib import Path
    from src.config.settings import COLUMN_NAMES, RAW_DATA_DIR

    loader = CMAPSSDataLoader(RAW_DATA_DIR, COLUMN_NAMES)

    # Load all datasets
    datasets = loader.load_all_datasets()

    # Merge datasets
    if datasets:
        train_df, test_df = loader.merge_datasets(datasets)

        # Get summaries
        train_summary = loader.get_dataset_summary(train_df, "Training")
        test_summary = loader.get_dataset_summary(test_df, "Test")

        print("\nTraining Data Summary:")
        print(f"  Total Records: {train_summary['total_records']:,}")
        print(f"  Total Engines: {train_summary['total_engines']}")
        print(f"  Memory Usage: {train_summary['memory_usage_mb']} MB")

