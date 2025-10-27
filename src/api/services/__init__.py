"""
Services module initialization
"""


__all__ = ["ModelService", "PipelineService"]
"""
Model Service - Load and manage trained models
Author: ramkumarjayakumar
Date: 2025-10-27
"""

import logging
import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class ModelService:
    """
    Service to load, manage, and use trained ML models
    """

    def __init__(self, models_dir: Path):
        """
        Initialize model service

        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.best_model = None
        self.model_info = {}

        logger.info(f"Initializing ModelService with models_dir: {self.models_dir}")
        self._load_models()

    def _load_models(self):
        """Load all available models from disk"""
        logger.info("Loading trained models...")

        try:
            # Load Random Forest
            rf_path = self.models_dir / 'random_forest.pkl'
            if rf_path.exists():
                self.models['random_forest'] = joblib.load(rf_path)
                logger.info(f"✓ Loaded Random Forest from {rf_path}")

            # Load XGBoost
            xgb_path = self.models_dir / 'xgboost.pkl'
            if xgb_path.exists():
                self.models['xgboost'] = joblib.load(xgb_path)
                logger.info(f"✓ Loaded XGBoost from {xgb_path}")

            # Set best model (XGBoost based on evaluation)
            if 'xgboost' in self.models:
                self.best_model = 'xgboost'
                logger.info("✓ XGBoost set as best model (highest accuracy: 99.61%)")
            elif 'random_forest' in self.models:
                self.best_model = 'random_forest'
                logger.info("✓ Random Forest set as best model")

            # Store model info
            self.model_info = {
                'random_forest': {
                    'accuracy': 0.9824,
                    'precision': 0.9017,
                    'recall': 0.9787,
                    'f1_score': 0.9386,
                    'roc_auc': 0.9987
                },
                'xgboost': {
                    'accuracy': 0.9961,
                    'precision': 0.9782,
                    'recall': 0.9939,
                    'f1_score': 0.9860,
                    'roc_auc': 0.9999
                }
            }

            logger.info(f"✓ Loaded {len(self.models)} models successfully")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def get_available_models(self) -> list:
        """Get list of available models"""
        return list(self.models.keys())

    def get_best_model_name(self) -> str:
        """Get name of best performing model"""
        return self.best_model

    def get_model_metrics(self, model_name: str = None) -> Dict[str, float]:
        """
        Get metrics for a specific model

        Args:
            model_name: Model name (defaults to best model)

        Returns:
            Dictionary of metrics
        """
        if model_name is None:
            model_name = self.best_model

        return self.model_info.get(model_name, {})

    def get_all_model_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics for all models"""
        return self.model_info

    def predict(self, features: np.ndarray, model_name: str = None) -> Tuple[int, float]:
        """
        Make a prediction using specified model

        Args:
            features: Input features (numpy array)
            model_name: Model to use (defaults to best model)

        Returns:
            Tuple of (prediction, probability)
        """
        if model_name is None:
            model_name = self.best_model

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]

        # Make prediction
        prediction = model.predict(features.reshape(1, -1))[0]

        # Get probability
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features.reshape(1, -1))[0]
            probability = probabilities[int(prediction)]
        else:
            probability = 0.5  # Fallback for models without probability

        logger.info(f"Prediction from {model_name}: {prediction} (confidence: {probability:.4f})")

        return int(prediction), float(probability)

    def get_model_info_summary(self) -> Dict[str, Any]:
        """Get comprehensive model information summary"""
        return {
            'best_model': self.best_model,
            'models_available': self.get_available_models(),
            'model_metrics': self.get_all_model_metrics(),
            'total_models': len(self.models)
        }

