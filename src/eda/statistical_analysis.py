"""
Statistical Analysis for NASA C-MAPSS Dataset
Author: ramkumarjayakumar
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from scipy import stats
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for NASA C-MAPSS dataset
    Generates descriptive statistics, distributions, and statistical tests
    """

    def __init__(self):
        """Initialize statistical analyzer"""
        self.analysis_results = {}

    def analyze_dataset(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict:
        """
        Perform comprehensive statistical analysis

        Args:
            df: DataFrame to analyze
            dataset_name: Name for logging

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Starting statistical analysis for {dataset_name}")

        results = {
            'dataset_name': dataset_name,
            'basic_stats': self._get_basic_statistics(df),
            'distribution_stats': self._get_distribution_statistics(df),
            'sensor_stats': self._get_sensor_statistics(df),
            'engine_stats': self._get_engine_statistics(df),
            'rul_stats': self._get_rul_statistics(df)
        }

        self.analysis_results[dataset_name] = results

        logger.info(f"Statistical analysis complete for {dataset_name}")

        return results

    def _get_basic_statistics(self, df: pd.DataFrame) -> Dict:
        """Get basic dataset statistics"""
        logger.info("Calculating basic statistics...")

        stats_dict = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(exclude=[np.number]).columns)
        }

        if 'unit_id' in df.columns:
            stats_dict['total_engines'] = df['unit_id'].nunique()

        return stats_dict

    def _get_distribution_statistics(self, df: pd.DataFrame) -> Dict:
        """Get distribution statistics for numeric features"""
        logger.info("Calculating distribution statistics...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        distribution_stats = {}

        for col in numeric_cols:
            if col in ['unit_id']:
                continue

            col_data = df[col].dropna()

            if len(col_data) == 0:
                continue

            distribution_stats[col] = {
                'count': int(len(col_data)),
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'q25': float(col_data.quantile(0.25)),
                'median': float(col_data.median()),
                'q75': float(col_data.quantile(0.75)),
                'max': float(col_data.max()),
                'skewness': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis()),
                'variance': float(col_data.var()),
                'cv': float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else 0
            }

            # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for large)
            if len(col_data) < 5000:
                try:
                    statistic, p_value = stats.shapiro(col_data.sample(min(5000, len(col_data))))
                    distribution_stats[col]['normality_test'] = {
                        'test': 'Shapiro-Wilk',
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'is_normal': p_value > 0.05
                    }
                except:
                    pass

        return distribution_stats

    def _get_sensor_statistics(self, df: pd.DataFrame) -> Dict:
        """Get statistics specific to sensor data"""
        logger.info("Calculating sensor statistics...")

        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]

        if not sensor_cols:
            return {}

        sensor_stats = {
            'total_sensors': len(sensor_cols),
            'sensor_summary': {}
        }

        for sensor in sensor_cols:
            sensor_data = df[sensor].dropna()

            sensor_stats['sensor_summary'][sensor] = {
                'mean': float(sensor_data.mean()),
                'std': float(sensor_data.std()),
                'range': float(sensor_data.max() - sensor_data.min()),
                'coefficient_of_variation': float(sensor_data.std() / sensor_data.mean()) if sensor_data.mean() != 0 else 0,
                'zero_variance': bool(sensor_data.std() == 0)
            }

        return sensor_stats

    def _get_engine_statistics(self, df: pd.DataFrame) -> Dict:
        """Get statistics per engine"""
        logger.info("Calculating engine statistics...")

        if 'unit_id' not in df.columns or 'time_cycles' not in df.columns:
            return {}

        # Calculate cycles per engine
        cycles_per_engine = df.groupby('unit_id')['time_cycles'].max()

        engine_stats = {
            'total_engines': int(df['unit_id'].nunique()),
            'cycles_per_engine': {
                'mean': float(cycles_per_engine.mean()),
                'std': float(cycles_per_engine.std()),
                'min': int(cycles_per_engine.min()),
                'max': int(cycles_per_engine.max()),
                'median': float(cycles_per_engine.median())
            },
            'records_per_engine': {
                'mean': float(df.groupby('unit_id').size().mean()),
                'std': float(df.groupby('unit_id').size().std()),
                'min': int(df.groupby('unit_id').size().min()),
                'max': int(df.groupby('unit_id').size().max())
            }
        }

        return engine_stats

    def _get_rul_statistics(self, df: pd.DataFrame) -> Dict:
        """Get RUL-specific statistics"""
        logger.info("Calculating RUL statistics...")

        if 'RUL' not in df.columns:
            return {}

        rul_data = df['RUL'].dropna()

        rul_stats = {
            'mean_rul': float(rul_data.mean()),
            'std_rul': float(rul_data.std()),
            'min_rul': float(rul_data.min()),
            'max_rul': float(rul_data.max()),
            'median_rul': float(rul_data.median()),
            'distribution': {
                '0-30_cycles': int((rul_data <= 30).sum()),
                '31-60_cycles': int(((rul_data > 30) & (rul_data <= 60)).sum()),
                '61-100_cycles': int(((rul_data > 60) & (rul_data <= 100)).sum()),
                '100+_cycles': int((rul_data > 100).sum())
            }
        }

        if 'failure_label' in df.columns:
            rul_stats['failure_distribution'] = {
                'failure_cases': int(df['failure_label'].sum()),
                'healthy_cases': int((df['failure_label'] == 0).sum()),
                'failure_percentage': float(df['failure_label'].mean() * 100)
            }

        return rul_stats

    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report"""
        report = f"""
{'='*80}
STATISTICAL ANALYSIS REPORT
NASA C-MAPSS Dataset
{'='*80}

"""

        for dataset_name, results in self.analysis_results.items():
            report += f"""
{'-'*80}
Dataset: {dataset_name}
{'-'*80}

BASIC STATISTICS:
  Total Records: {results['basic_stats']['total_records']:,}
  Total Features: {results['basic_stats']['total_features']}
  Memory Usage: {results['basic_stats']['memory_usage_mb']} MB
"""

            if 'total_engines' in results['basic_stats']:
                report += f"  Total Engines: {results['basic_stats']['total_engines']}\n"

            if results['engine_stats']:
                report += f"""
ENGINE STATISTICS:
  Total Engines: {results['engine_stats']['total_engines']}
  Cycles per Engine:
    - Mean: {results['engine_stats']['cycles_per_engine']['mean']:.2f}
    - Std: {results['engine_stats']['cycles_per_engine']['std']:.2f}
    - Min: {results['engine_stats']['cycles_per_engine']['min']}
    - Max: {results['engine_stats']['cycles_per_engine']['max']}
"""

            if results['rul_stats']:
                report += f"""
RUL STATISTICS:
  Mean RUL: {results['rul_stats']['mean_rul']:.2f} cycles
  Std RUL: {results['rul_stats']['std_rul']:.2f} cycles
  Min RUL: {results['rul_stats']['min_rul']:.2f} cycles
  Max RUL: {results['rul_stats']['max_rul']:.2f} cycles
  
  RUL Distribution:
    - 0-30 cycles: {results['rul_stats']['distribution']['0-30_cycles']:,}
    - 31-60 cycles: {results['rul_stats']['distribution']['31-60_cycles']:,}
    - 61-100 cycles: {results['rul_stats']['distribution']['61-100_cycles']:,}
    - 100+ cycles: {results['rul_stats']['distribution']['100+_cycles']:,}
"""

                if 'failure_distribution' in results['rul_stats']:
                    report += f"""
  Failure Distribution:
    - Failure Cases: {results['rul_stats']['failure_distribution']['failure_cases']:,}
    - Healthy Cases: {results['rul_stats']['failure_distribution']['healthy_cases']:,}
    - Failure Rate: {results['rul_stats']['failure_distribution']['failure_percentage']:.2f}%
"""

            if results['sensor_stats'] and 'total_sensors' in results['sensor_stats']:
                report += f"""
SENSOR STATISTICS:
  Total Sensors: {results['sensor_stats']['total_sensors']}
"""

        report += f"""
{'='*80}
END OF STATISTICAL ANALYSIS REPORT
{'='*80}
"""

        return report

    def save_report(self, filepath: Path):
        """Save report to file"""
        with open(filepath, 'w') as f:
            f.write(self.generate_summary_report())
        logger.info(f"Statistical report saved to {filepath}")

    def export_json(self, filepath: Path):
        """Export results as JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        logger.info(f"Statistical results exported to {filepath}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    analyzer = StatisticalAnalyzer()
    print("Statistical Analyzer initialized successfully")

