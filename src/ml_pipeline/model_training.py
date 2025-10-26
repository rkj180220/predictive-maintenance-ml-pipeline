"""
Model Training Module
Author: ramkumarjayakumar
Date: 2025-10-26

Activity 2.2: Split dataset (70/30) and train models
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Train machine learning models for predictive maintenance
    Handles data splitting, model training, and model persistence
    """

    def __init__(self, test_size: float = 0.3, random_state: int = 42):
        """
        Initialize Model Trainer

        Args:
            test_size: Proportion of dataset for testing (default: 0.3 for 70/30 split)
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.training_history = {}

    def load_processed_data(self, data_path: Path) -> pd.DataFrame:
        """
        Load processed data from parquet file

        Args:
            data_path: Path to processed data file

        Returns:
            Processed DataFrame
        """
        logger.info(f"Loading processed data from {data_path}")

        if str(data_path).endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif str(data_path).endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        logger.info(f"Loaded data shape: {df.shape}")
        logger.info(f"Columns: {len(df.columns)}")

        return df

    def prepare_features_target(self, df: pd.DataFrame, target_col: str = 'failure_label') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable

        Args:
            df: Input DataFrame
            target_col: Name of target column

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info(f"Preparing features and target (target column: {target_col})")

        # Identify columns to exclude from features
        exclude_cols = [target_col, 'unit_id', 'time_cycles', 'RUL']
        exclude_cols = [col for col in exclude_cols if col in df.columns]

        # Separate features and target
        X = df.drop(columns=exclude_cols)
        y = df[target_col]

        # Handle any remaining non-numeric columns
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            logger.warning(f"Dropping non-numeric columns: {non_numeric_cols}")
            X = X.select_dtypes(include=[np.number])

        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)

        # Fill any NaN values with column median
        if X.isnull().any().any():
            logger.warning("Found NaN values, filling with median")
            X = X.fillna(X.median())

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        logger.info(f"Class balance: {y.value_counts(normalize=True).to_dict()}")

        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training (70%) and testing (30%) sets

        Args:
            X: Features DataFrame
            y: Target Series

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("="*80)
        logger.info("SPLITTING DATA: 70% Training, 30% Testing")
        logger.info("="*80)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # Maintain class distribution in both sets
        )

        logger.info(f"\n📊 TRAIN SET:")
        logger.info(f"   Shape: {X_train.shape}")
        logger.info(f"   Samples: {len(X_train):,}")
        logger.info(f"   Features: {X_train.shape[1]}")
        logger.info(f"   Class 0 (No Failure): {(y_train == 0).sum():,} ({(y_train == 0).sum() / len(y_train) * 100:.2f}%)")
        logger.info(f"   Class 1 (Failure): {(y_train == 1).sum():,} ({(y_train == 1).sum() / len(y_train) * 100:.2f}%)")

        logger.info(f"\n📊 TEST SET:")
        logger.info(f"   Shape: {X_test.shape}")
        logger.info(f"   Samples: {len(X_test):,}")
        logger.info(f"   Features: {X_test.shape[1]}")
        logger.info(f"   Class 0 (No Failure): {(y_test == 0).sum():,} ({(y_test == 0).sum() / len(y_test) * 100:.2f}%)")
        logger.info(f"   Class 1 (Failure): {(y_test == 1).sum():,} ({(y_test == 1).sum() / len(y_test) * 100:.2f}%)")

        logger.info("="*80 + "\n")

        return X_train, X_test, y_train, y_test

    def train_model(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                   model_name: str) -> Tuple[Any, Dict]:
        """
        Train a single model and track metrics

        Args:
            model: Model instance to train
            X_train: Training features
            y_train: Training target
            model_name: Name of the model for logging

        Returns:
            Tuple of (trained model, training metadata)
        """
        logger.info("="*80)
        logger.info(f"TRAINING {model_name.upper()}")
        logger.info("="*80)

        start_time = time.time()

        logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Training samples: {len(X_train):,}")
        logger.info(f"Number of features: {X_train.shape[1]}")

        # Train the model
        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        logger.info(f"✅ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

        # Collect training metadata
        metadata = {
            'model_name': model_name,
            'training_samples': len(X_train),
            'n_features': X_train.shape[1],
            'training_time_seconds': round(training_time, 2),
            'training_time_minutes': round(training_time / 60, 2),
            'timestamp': datetime.now().isoformat(),
            'test_size': self.test_size,
            'random_state': self.random_state
        }

        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info(f"\n🔍 Top 10 Most Important Features for {model_name}:")
            for idx, row in feature_importance.head(10).iterrows():
                logger.info(f"   {row['feature']}: {row['importance']:.6f}")

            metadata['top_10_features'] = feature_importance.head(10).to_dict('records')

        logger.info("="*80 + "\n")

        self.training_history[model_name] = metadata

        return model, metadata

    def save_model(self, model: Any, model_name: str, output_dir: Path):
        """
        Save trained model to disk

        Args:
            model: Trained model
            model_name: Name of the model
            output_dir: Directory to save model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"{model_name.lower().replace(' ', '_')}.pkl"

        joblib.dump(model, model_path)
        logger.info(f"💾 Model saved to: {model_path}")

        return model_path

    def save_training_history(self, output_dir: Path):
        """
        Save training history to JSON

        Args:
            output_dir: Directory to save history
        """
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        history_path = output_dir / "training_history.json"

        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        logger.info(f"📊 Training history saved to: {history_path}")

        return history_path

    def full_training_pipeline(self, data_path: Path, rf_model, xgb_model,
                              output_dir: Path) -> Dict:
        """
        Execute complete training pipeline

        Args:
            data_path: Path to processed data
            rf_model: Random Forest model instance
            xgb_model: XGBoost model instance
            output_dir: Directory to save models

        Returns:
            Dictionary with all training results
        """
        logger.info("\n" + "🚀"*40)
        logger.info("STARTING FULL MODEL TRAINING PIPELINE")
        logger.info("🚀"*40 + "\n")

        # Load data
        df = self.load_processed_data(data_path)

        # Prepare features and target
        X, y = self.prepare_features_target(df)

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Train Random Forest
        rf_trained, rf_metadata = self.train_model(rf_model, X_train, y_train, "Random Forest")
        rf_model_path = self.save_model(rf_trained, "Random_Forest", output_dir)

        # Train XGBoost
        xgb_trained, xgb_metadata = self.train_model(xgb_model, X_train, y_train, "XGBoost")
        xgb_model_path = self.save_model(xgb_trained, "XGBoost", output_dir)

        # Save training history
        history_path = self.save_training_history(output_dir)

        results = {
            'random_forest': {
                'model': rf_trained,
                'metadata': rf_metadata,
                'model_path': str(rf_model_path)
            },
            'xgboost': {
                'model': xgb_trained,
                'metadata': xgb_metadata,
                'model_path': str(xgb_model_path)
            },
            'data_split': {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            },
            'history_path': str(history_path)
        }

        logger.info("\n" + "✅"*40)
        logger.info("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("✅"*40 + "\n")

        return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Model Training Module Initialized")

