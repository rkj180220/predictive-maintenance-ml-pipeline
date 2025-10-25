"""
Correlation Analysis for NASA C-MAPSS Dataset
Author: ramkumarjayakumar
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Comprehensive correlation analysis for sensor data
    Calculates correlations between sensors and with target variable (RUL)
    """

    def __init__(self, config: Dict):
        """
        Initialize correlation analyzer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        self.correlation_results = {}

    def analyze_correlations(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict:
        """
        Perform comprehensive correlation analysis

        Args:
            df: DataFrame to analyze
            dataset_name: Name for logging

        Returns:
            Dictionary with correlation results
        """
        logger.info(f"Starting correlation analysis for {dataset_name}")

        results = {
            'dataset_name': dataset_name,
            'feature_correlations': self._calculate_feature_correlations(df),
            'target_correlations': self._calculate_target_correlations(df),
            'high_correlations': self._find_high_correlations(df),
            'multicollinearity': self._detect_multicollinearity(df)
        }

        self.correlation_results[dataset_name] = results

        logger.info(f"Correlation analysis complete for {dataset_name}")

        return results

    def _calculate_feature_correlations(self, df: pd.DataFrame) -> Dict:
        """Calculate correlation matrix for all features"""
        logger.info("Calculating feature correlation matrix...")

        # Get numeric columns excluding identifiers
        exclude_cols = ['unit_id', 'time_cycles', 'dataset']
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col not in exclude_cols]

        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()

        # Convert to dictionary format
        corr_dict = {}
        for i, col1 in enumerate(numeric_cols):
            corr_dict[col1] = {}
            for j, col2 in enumerate(numeric_cols):
                if i <= j:  # Only store upper triangle to avoid duplication
                    corr_dict[col1][col2] = float(corr_matrix.loc[col1, col2])

        logger.info(f"Calculated correlations for {len(numeric_cols)} features")

        return {
            'matrix': corr_dict,
            'features': numeric_cols
        }

    def _calculate_target_correlations(self, df: pd.DataFrame) -> Dict:
        """Calculate correlations with target variable (RUL)"""
        logger.info("Calculating correlations with RUL...")

        if 'RUL' not in df.columns:
            logger.warning("RUL column not found, skipping target correlation")
            return {}

        # Get feature columns
        exclude_cols = ['unit_id', 'time_cycles', 'RUL', 'failure_label', 'dataset']
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col not in exclude_cols]

        # Calculate correlations with RUL
        rul_correlations = {}
        for col in feature_cols:
            corr = df[col].corr(df['RUL'])
            if not np.isnan(corr):
                rul_correlations[col] = float(corr)

        # Sort by absolute correlation
        sorted_correlations = sorted(
            rul_correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        logger.info(f"Calculated RUL correlations for {len(rul_correlations)} features")

        return {
            'correlations': dict(sorted_correlations),
            'top_positive': dict(sorted_correlations[:10]),
            'top_negative': dict(sorted(sorted_correlations, key=lambda x: x[1])[:10]),
            'top_absolute': dict(sorted_correlations[:10])
        }

    def _find_high_correlations(self, df: pd.DataFrame) -> Dict:
        """Find pairs of features with high correlation"""
        logger.info(f"Finding high correlations (threshold: {self.correlation_threshold})...")

        # Get sensor columns
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]

        if len(sensor_cols) < 2:
            return {}

        # Calculate correlation matrix
        corr_matrix = df[sensor_cols].corr()

        # Find high correlations
        high_corr_pairs = []
        for i, col1 in enumerate(sensor_cols):
            for j, col2 in enumerate(sensor_cols):
                if i < j:  # Only check upper triangle
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) >= self.correlation_threshold:
                        high_corr_pairs.append({
                            'feature1': col1,
                            'feature2': col2,
                            'correlation': float(corr_value),
                            'abs_correlation': float(abs(corr_value))
                        })

        # Sort by absolute correlation
        high_corr_pairs = sorted(high_corr_pairs, key=lambda x: x['abs_correlation'], reverse=True)

        logger.info(f"Found {len(high_corr_pairs)} high correlation pairs")

        return {
            'threshold': self.correlation_threshold,
            'total_pairs': len(high_corr_pairs),
            'pairs': high_corr_pairs
        }

    def _detect_multicollinearity(self, df: pd.DataFrame) -> Dict:
        """Detect multicollinearity among features"""
        logger.info("Detecting multicollinearity...")

        # Get sensor columns
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]

        if len(sensor_cols) < 2:
            return {}

        # Calculate correlation matrix
        corr_matrix = df[sensor_cols].corr()

        # Find features with many high correlations
        multicollinear_features = {}
        for col in sensor_cols:
            high_corr_count = (abs(corr_matrix[col]) >= self.correlation_threshold).sum() - 1  # Exclude self
            if high_corr_count > 0:
                multicollinear_features[col] = {
                    'high_correlation_count': int(high_corr_count),
                    'mean_abs_correlation': float(abs(corr_matrix[col]).mean()),
                    'max_correlation': float(abs(corr_matrix[col][corr_matrix.index != col]).max())
                }

        # Sort by number of high correlations
        multicollinear_features = dict(
            sorted(multicollinear_features.items(),
                  key=lambda x: x[1]['high_correlation_count'],
                  reverse=True)
        )

        logger.info(f"Found {len(multicollinear_features)} potentially multicollinear features")

        return {
            'threshold': self.correlation_threshold,
            'features': multicollinear_features,
            'total_features': len(multicollinear_features)
        }

    def get_sensor_correlation_summary(self, dataset_name: str = None) -> pd.DataFrame:
        """
        Get summary of sensor correlations

        Args:
            dataset_name: Name of dataset to summarize

        Returns:
            DataFrame with correlation summary
        """
        if dataset_name is None:
            dataset_name = list(self.correlation_results.keys())[0]

        if dataset_name not in self.correlation_results:
            logger.error(f"No results found for {dataset_name}")
            return pd.DataFrame()

        results = self.correlation_results[dataset_name]

        # Create summary DataFrame
        summary_data = []

        if 'high_correlations' in results and results['high_correlations']:
            for pair in results['high_correlations']['pairs'][:20]:  # Top 20
                summary_data.append({
                    'Feature 1': pair['feature1'],
                    'Feature 2': pair['feature2'],
                    'Correlation': round(pair['correlation'], 4)
                })

        return pd.DataFrame(summary_data)

    def generate_correlation_report(self) -> str:
        """Generate correlation analysis report"""
        report = f"""
{'='*80}
CORRELATION ANALYSIS REPORT
NASA C-MAPSS Dataset
{'='*80}

"""

        for dataset_name, results in self.correlation_results.items():
            report += f"""
{'-'*80}
Dataset: {dataset_name}
{'-'*80}

FEATURE CORRELATIONS:
  Total Features: {len(results['feature_correlations']['features'])}
"""

            if results['target_correlations']:
                report += f"""
TARGET (RUL) CORRELATIONS:
  Top 5 Positive Correlations:
"""
                for feature, corr in list(results['target_correlations']['top_positive'].items())[:5]:
                    report += f"    - {feature}: {corr:.4f}\n"

                report += f"""
  Top 5 Negative Correlations:
"""
                for feature, corr in list(results['target_correlations']['top_negative'].items())[:5]:
                    report += f"    - {feature}: {corr:.4f}\n"

            if results['high_correlations']:
                report += f"""
HIGH CORRELATIONS (threshold: {results['high_correlations']['threshold']}):
  Total High Correlation Pairs: {results['high_correlations']['total_pairs']}
  
  Top 10 Correlated Pairs:
"""
                for pair in results['high_correlations']['pairs'][:10]:
                    report += f"    - {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.4f}\n"

            if results['multicollinearity']:
                report += f"""
MULTICOLLINEARITY DETECTION:
  Features with Multiple High Correlations: {results['multicollinearity']['total_features']}
  
  Top 5 Multicollinear Features:
"""
                for feature, stats in list(results['multicollinearity']['features'].items())[:5]:
                    report += f"    - {feature}: {stats['high_correlation_count']} high correlations (max: {stats['max_correlation']:.4f})\n"

        report += f"""
{'='*80}
END OF CORRELATION ANALYSIS REPORT
{'='*80}
"""

        return report

    def save_report(self, filepath: Path):
        """Save correlation report to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.generate_correlation_report())
        logger.info(f"Correlation report saved to {filepath}")

    def export_json(self, filepath: Path):
        """Export correlation results as JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.correlation_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Correlation results exported to {filepath}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from src.config.settings import EDA_CONFIG

    analyzer = CorrelationAnalyzer(EDA_CONFIG)
    print("Correlation Analyzer initialized successfully")

