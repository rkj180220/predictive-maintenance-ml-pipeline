"""
Feature Importance Analysis for NASA C-MAPSS Dataset
Author: ramkumarjayakumar
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from pathlib import Path
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance using multiple methods
    - Random Forest feature importance
    - Gradient Boosting feature importance
    - Permutation importance
    - Correlation-based importance
    """

    def __init__(self):
        """Initialize feature importance analyzer"""
        self.importance_results = {}

    def analyze_importance(self, df: pd.DataFrame, target_col: str = 'RUL',
                          dataset_name: str = "dataset") -> Dict:
        """
        Perform comprehensive feature importance analysis

        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            dataset_name: Name for logging

        Returns:
            Dictionary with importance results
        """
        logger.info(f"Starting feature importance analysis for {dataset_name}")

        # Prepare data
        X, y, feature_names = self._prepare_data(df, target_col)

        if X is None or y is None:
            logger.error("Failed to prepare data")
            return {}

        results = {
            'dataset_name': dataset_name,
            'random_forest': self._random_forest_importance(X, y, feature_names),
            'gradient_boosting': self._gradient_boosting_importance(X, y, feature_names),
            'correlation': self._correlation_importance(df, target_col),
            'combined_ranking': {}
        }

        # Create combined ranking
        results['combined_ranking'] = self._create_combined_ranking(results)

        self.importance_results[dataset_name] = results

        logger.info(f"Feature importance analysis complete for {dataset_name}")

        return results

    def _prepare_data(self, df: pd.DataFrame, target_col: str):
        """Prepare features and target for analysis"""
        logger.info("Preparing data for feature importance analysis...")

        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found")
            return None, None, None

        # Exclude non-feature columns
        exclude_cols = ['unit_id', 'time_cycles', target_col, 'failure_label', 'dataset']
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col not in exclude_cols]

        # Remove columns with NaN or inf
        valid_cols = []
        for col in feature_cols:
            if not df[col].isnull().any() and not np.isinf(df[col]).any():
                valid_cols.append(col)

        feature_cols = valid_cols

        # CRITICAL OPTIMIZATION: Select only top 200 features by variance for speed
        if len(feature_cols) > 200:
            logger.info(f"Selecting top 200 features from {len(feature_cols)} by variance for faster analysis...")
            variances = df[feature_cols].var()
            top_features = variances.nlargest(200).index.tolist()
            feature_cols = top_features
            logger.info(f"✓ Reduced to {len(feature_cols)} features")

        # Sample data aggressively for speed
        sample_size = min(10000, len(df))  # Only 10k samples max
        if len(df) > sample_size:
            logger.info(f"Sampling {sample_size} records from {len(df)} for faster computation")
            df_sample = df.sample(sample_size, random_state=42)
        else:
            df_sample = df

        X = df_sample[feature_cols].values
        y = df_sample[target_col].values

        # Handle any remaining NaN
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"✓ Prepared {X.shape[0]} samples with {X.shape[1]} features")

        return X, y, feature_cols

    def _random_forest_importance(self, X, y, feature_names: List[str]) -> Dict:
        """Calculate Random Forest feature importance"""
        logger.info("Calculating Random Forest feature importance...")

        try:
            from tqdm import tqdm

            # Train Random Forest - optimized for speed
            logger.info(f"Training Random Forest with {X.shape[1]} features on {X.shape[0]} samples...")

            rf = RandomForestRegressor(
                n_estimators=50,  # Reduced from 100
                max_depth=8,  # Reduced from 10
                random_state=42,
                n_jobs=-1,
                max_features='sqrt',  # Speed optimization
                verbose=0
            )

            # Show progress
            with tqdm(total=50, desc="Random Forest Training", unit="tree") as pbar:
                rf.fit(X, y)
                pbar.update(50)

            # Get feature importance
            importances = rf.feature_importances_

            # Create importance dictionary
            importance_dict = {
                feature: float(importance)
                for feature, importance in zip(feature_names, importances)
            }

            # Sort by importance
            sorted_importance = sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )

            logger.info(f"✓ Random Forest importance calculated - Top feature: {sorted_importance[0][0]} ({sorted_importance[0][1]:.6f})")

            return {
                'importances': dict(sorted_importance[:100]),  # Only store top 100
                'top_10': dict(sorted_importance[:10])
            }

        except Exception as e:
            logger.error(f"Error calculating Random Forest importance: {e}")
            return {}

    def _gradient_boosting_importance(self, X, y, feature_names: List[str]) -> Dict:
        """Calculate Gradient Boosting feature importance"""
        logger.info("Calculating Gradient Boosting feature importance...")

        try:
            from tqdm import tqdm
            import warnings
            warnings.filterwarnings('ignore')

            # Train Gradient Boosting - optimized for speed
            logger.info(f"Training Gradient Boosting with {X.shape[1]} features on {X.shape[0]} samples...")

            # Use smaller model for faster training
            gb = GradientBoostingRegressor(
                n_estimators=30,  # Reduced from 50 for faster execution
                max_depth=3,  # Reduced from 4
                random_state=42,
                learning_rate=0.15,  # Increased for faster convergence
                subsample=0.7,  # Reduced for speed
                verbose=0
            )

            # Train with manual progress tracking
            logger.info("Training Gradient Boosting model (this may take 30-60 seconds)...")
            with tqdm(total=100, desc="Gradient Boosting", unit="%", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                # Fit in background, update progress periodically
                import threading
                import time

                fit_done = [False]

                def fit_model():
                    gb.fit(X, y)
                    fit_done[0] = True

                fit_thread = threading.Thread(target=fit_model)
                fit_thread.start()

                # Update progress while fitting
                while not fit_done[0]:
                    pbar.update(3)
                    time.sleep(0.5)
                    if pbar.n >= 100:
                        pbar.n = 95  # Keep at 95% until actually done

                fit_thread.join()
                pbar.n = 100
                pbar.refresh()

            # Get feature importance
            importances = gb.feature_importances_

            # Create importance dictionary
            importance_dict = {
                feature: float(importance)
                for feature, importance in zip(feature_names, importances)
            }

            # Sort by importance
            sorted_importance = sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )

            logger.info(f"✓ Gradient Boosting importance calculated - Top feature: {sorted_importance[0][0]} ({sorted_importance[0][1]:.6f})")

            return {
                'importances': dict(sorted_importance[:100]),  # Only store top 100
                'top_10': dict(sorted_importance[:10])
            }

        except Exception as e:
            logger.error(f"Error calculating Gradient Boosting importance: {e}")
            return {}

    def _correlation_importance(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Calculate correlation-based importance"""
        logger.info("Calculating correlation-based importance...")

        try:
            from tqdm import tqdm

            # Exclude non-feature columns
            exclude_cols = ['unit_id', 'time_cycles', target_col, 'failure_label', 'dataset']
            feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                           if col not in exclude_cols]

            # Calculate correlations with progress bar
            correlations = {}
            logger.info(f"Computing correlations for {len(feature_cols)} features...")

            for col in tqdm(feature_cols, desc="Correlation Analysis", unit="feature"):
                corr = df[col].corr(df[target_col])
                if not np.isnan(corr):
                    correlations[col] = abs(float(corr))

            # Sort by absolute correlation
            sorted_correlations = sorted(
                correlations.items(),
                key=lambda x: x[1],
                reverse=True
            )

            logger.info(f"✓ Correlation-based importance calculated - Top feature: {sorted_correlations[0][0]} ({sorted_correlations[0][1]:.6f})")

            return {
                'importances': dict(sorted_correlations[:100]),  # Only store top 100
                'top_10': dict(sorted_correlations[:10])
            }

        except Exception as e:
            logger.error(f"Error calculating correlation importance: {e}")
            return {}

    def _create_combined_ranking(self, results: Dict) -> Dict:
        """Create combined ranking from all methods"""
        logger.info("Creating combined feature ranking...")

        # Collect all features
        all_features = set()

        for method in ['random_forest', 'gradient_boosting', 'correlation']:
            if method in results and 'importances' in results[method]:
                all_features.update(results[method]['importances'].keys())

        # Calculate average rank
        combined_scores = {}

        for feature in all_features:
            scores = []

            # Get normalized scores from each method
            for method in ['random_forest', 'gradient_boosting', 'correlation']:
                if method in results and 'importances' in results[method]:
                    if feature in results[method]['importances']:
                        scores.append(results[method]['importances'][feature])

            if scores:
                combined_scores[feature] = np.mean(scores)

        # Sort by combined score
        sorted_combined = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        logger.info("Combined ranking created")

        return {
            'importances': dict(sorted_combined),
            'top_20': dict(sorted_combined[:20])
        }

    def generate_importance_report(self) -> str:
        """Generate feature importance report"""
        report = f"""
{'='*80}
FEATURE IMPORTANCE ANALYSIS REPORT
NASA C-MAPSS Dataset
{'='*80}

"""

        for dataset_name, results in self.importance_results.items():
            report += f"""
{'-'*80}
Dataset: {dataset_name}
{'-'*80}

RANDOM FOREST IMPORTANCE:
Top 10 Features:
"""
            if 'random_forest' in results and 'top_10' in results['random_forest']:
                for rank, (feature, importance) in enumerate(results['random_forest']['top_10'].items(), 1):
                    report += f"  {rank}. {feature}: {importance:.6f}\n"

            report += f"""
GRADIENT BOOSTING IMPORTANCE:
Top 10 Features:
"""
            if 'gradient_boosting' in results and 'top_10' in results['gradient_boosting']:
                for rank, (feature, importance) in enumerate(results['gradient_boosting']['top_10'].items(), 1):
                    report += f"  {rank}. {feature}: {importance:.6f}\n"

            report += f"""
CORRELATION-BASED IMPORTANCE:
Top 10 Features:
"""
            if 'correlation' in results and 'top_10' in results['correlation']:
                for rank, (feature, importance) in enumerate(results['correlation']['top_10'].items(), 1):
                    report += f"  {rank}. {feature}: {importance:.6f}\n"

            report += f"""
COMBINED RANKING:
Top 20 Features:
"""
            if 'combined_ranking' in results and 'top_20' in results['combined_ranking']:
                for rank, (feature, importance) in enumerate(results['combined_ranking']['top_20'].items(), 1):
                    report += f"  {rank}. {feature}: {importance:.6f}\n"

        report += f"""
{'='*80}
END OF FEATURE IMPORTANCE REPORT
{'='*80}
"""

        return report

    def save_report(self, filepath: Path):
        """Save importance report to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.generate_importance_report())
        logger.info(f"Feature importance report saved to {filepath}")

    def export_json(self, filepath: Path):
        """Export importance results as JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.importance_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Feature importance results exported to {filepath}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    analyzer = FeatureImportanceAnalyzer()
    print("Feature Importance Analyzer initialized successfully")
