"""
Model Preparation Module
Author: ramkumarjayakumar
Date: 2025-10-26

Activity 2.1: Identify and prepare suitable ML algorithms for predictive maintenance
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class ModelPreparation:
    """
    Prepare and justify machine learning algorithms for predictive maintenance

    Business Context:
    - Binary Classification Task (Failure vs No Failure)
    - Imbalanced Dataset (~14% failure class)
    - High-dimensional feature space (100+ engineered features)
    - Time-series sensor data with temporal dependencies
    - Cost-sensitive: False Positives (unnecessary maintenance) vs False Negatives (missed failures)
    """

    def __init__(self):
        self.selected_algorithms = {}
        self.algorithm_justifications = {}

    def get_algorithm_selection(self) -> Dict:
        """
        Select and justify two algorithms for the predictive maintenance problem

        Selected Algorithms:
        1. Random Forest Classifier
        2. XGBoost Classifier

        Returns:
            Dictionary with algorithm details and justifications
        """

        selection = {
            "primary_algorithm": {
                "name": "Random Forest Classifier",
                "class": "RandomForestClassifier",
                "justification": {
                    "strengths": [
                        "Handles high-dimensional data (100+ features) effectively",
                        "Built-in feature importance for interpretability",
                        "Robust to outliers and noise in sensor data",
                        "No feature scaling required",
                        "Handles non-linear relationships well",
                        "Less prone to overfitting through ensemble averaging",
                        "Can handle class imbalance with class_weight parameter"
                    ],
                    "business_fit": [
                        "Provides feature importance rankings for maintenance teams",
                        "Interpretable decision paths for failure prediction",
                        "Proven track record in industrial predictive maintenance",
                        "Fast prediction time for real-time deployment"
                    ],
                    "technical_fit": [
                        "Effective with time-series engineered features (lags, rolling stats)",
                        "Handles missing values naturally",
                        "Parallelizable training for large datasets",
                        "Minimal hyperparameter tuning required for baseline"
                    ]
                },
                "hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 20,
                    "min_samples_split": 10,
                    "min_samples_leaf": 4,
                    "max_features": "sqrt",
                    "class_weight": "balanced",
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbose": 1
                }
            },

            "secondary_algorithm": {
                "name": "XGBoost Classifier",
                "class": "XGBClassifier",
                "justification": {
                    "strengths": [
                        "State-of-the-art gradient boosting performance",
                        "Handles imbalanced data with scale_pos_weight",
                        "Built-in regularization (L1/L2) prevents overfitting",
                        "Excellent with structured/tabular data",
                        "GPU acceleration available for faster training",
                        "Automatic handling of missing values",
                        "Feature importance via gain/cover/frequency"
                    ],
                    "business_fit": [
                        "Industry-leading accuracy for failure prediction",
                        "Lower false positive rate reduces unnecessary maintenance costs",
                        "Confidence scores help prioritize maintenance actions",
                        "Widely adopted in production environments"
                    ],
                    "technical_fit": [
                        "Superior with engineered time-series features",
                        "Handles complex feature interactions automatically",
                        "Early stopping prevents overfitting",
                        "Cross-validation support built-in",
                        "Efficient memory usage with sparse matrices"
                    ]
                },
                "hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 10,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 5,
                    "gamma": 0.1,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "scale_pos_weight": 6.3,  # Ratio of negative to positive class
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbosity": 1,
                    "eval_metric": "logloss"
                }
            }
        }

        logger.info("="*80)
        logger.info("MODEL SELECTION FOR PREDICTIVE MAINTENANCE")
        logger.info("="*80)
        logger.info(f"\nPrimary Algorithm: {selection['primary_algorithm']['name']}")
        logger.info(f"Secondary Algorithm: {selection['secondary_algorithm']['name']}")
        logger.info("="*80)

        return selection

    def initialize_models(self) -> Tuple[RandomForestClassifier, XGBClassifier]:
        """
        Initialize the two selected models with optimal hyperparameters

        Returns:
            Tuple of (RandomForest model, XGBoost model)
        """
        selection = self.get_algorithm_selection()

        # Initialize Random Forest
        rf_params = selection['primary_algorithm']['hyperparameters']
        rf_model = RandomForestClassifier(**rf_params)
        logger.info(f"Initialized Random Forest with parameters: {rf_params}")

        # Initialize XGBoost
        xgb_params = selection['secondary_algorithm']['hyperparameters']
        xgb_model = XGBClassifier(**xgb_params)
        logger.info(f"Initialized XGBoost with parameters: {xgb_params}")

        return rf_model, xgb_model

    def get_alternative_algorithms(self) -> List[Dict]:
        """
        Document alternative algorithms considered but not selected

        Returns:
            List of alternative algorithms with reasons for exclusion
        """
        alternatives = [
            {
                "name": "Logistic Regression",
                "reason_not_selected": "Linear model insufficient for complex non-linear sensor relationships"
            },
            {
                "name": "Support Vector Machine (SVM)",
                "reason_not_selected": "Poor scalability with 100K+ records, computationally expensive"
            },
            {
                "name": "LSTM/Neural Networks",
                "reason_not_selected": "Requires more data and longer training time, less interpretable for maintenance teams"
            },
            {
                "name": "Gradient Boosting (sklearn)",
                "reason_not_selected": "XGBoost offers superior performance and features"
            },
            {
                "name": "K-Nearest Neighbors",
                "reason_not_selected": "Poor performance with high-dimensional data, slow prediction time"
            }
        ]

        return alternatives

    def print_selection_report(self):
        """Generate comprehensive model selection report"""
        selection = self.get_algorithm_selection()
        alternatives = self.get_alternative_algorithms()

        report = []
        report.append("\n" + "="*100)
        report.append("MODEL PREPARATION REPORT - PREDICTIVE MAINTENANCE")
        report.append("NASA C-MAPSS Turbofan Engine Dataset")
        report.append("="*100)

        report.append("\n📊 PROBLEM CHARACTERISTICS:")
        report.append("  • Task Type: Binary Classification (Failure Prediction)")
        report.append("  • Dataset Size: 115,008 training records, 509 engines")
        report.append("  • Feature Space: 100+ engineered features from 21 sensors")
        report.append("  • Class Distribution: Imbalanced (~14% failure class)")
        report.append("  • Data Type: Time-series sensor readings with engineered features")

        report.append("\n🎯 SELECTED ALGORITHMS:")
        report.append(f"\n1️⃣  PRIMARY: {selection['primary_algorithm']['name']}")
        report.append("   Justification:")
        for strength in selection['primary_algorithm']['justification']['strengths'][:5]:
            report.append(f"     ✓ {strength}")
        report.append(f"   Hyperparameters: {selection['primary_algorithm']['hyperparameters']}")

        report.append(f"\n2️⃣  SECONDARY: {selection['secondary_algorithm']['name']}")
        report.append("   Justification:")
        for strength in selection['secondary_algorithm']['justification']['strengths'][:5]:
            report.append(f"     ✓ {strength}")
        report.append(f"   Hyperparameters: {selection['secondary_algorithm']['hyperparameters']}")

        report.append("\n❌ ALGORITHMS CONSIDERED BUT NOT SELECTED:")
        for alt in alternatives:
            report.append(f"   • {alt['name']}: {alt['reason_not_selected']}")

        report.append("\n💼 BUSINESS IMPACT:")
        report.append("   • Minimize False Negatives: Catch failures before they occur")
        report.append("   • Minimize False Positives: Reduce unnecessary maintenance costs")
        report.append("   • Interpretability: Provide actionable insights to maintenance teams")
        report.append("   • Real-time Prediction: Fast inference for operational deployment")

        report.append("\n" + "="*100)

        report_text = "\n".join(report)
        logger.info(report_text)

        return report_text


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    prep = ModelPreparation()
    prep.print_selection_report()
    rf_model, xgb_model = prep.initialize_models()
    print("\n✅ Models initialized successfully!")
"""
Machine Learning Pipeline for Predictive Maintenance
Author: ramkumarjayakumar
Date: 2025-10-26

Sub-Objective 2: ML Pipeline Implementation
"""

__version__ = "1.0.0"

