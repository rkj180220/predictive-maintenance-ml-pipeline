"""
EDA Report Generator
Author: ramkumarjayakumar
Date: 2025-10-18

Generates comprehensive EDA summary reports
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EDAReportGenerator:
    """Generate comprehensive EDA reports"""

    def __init__(self):
        self.report_data = {}

    def generate_full_report(self, df: pd.DataFrame, output_dir: Path, dataset_name: str = "Training"):
        """
        Generate complete EDA report

        Args:
            df: DataFrame to analyze
            output_dir: Output directory for reports
            dataset_name: Name of dataset
        """
        logger.info(f"Generating EDA report for {dataset_name}...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate report sections
        report = {
            'metadata': self._generate_metadata(df, dataset_name),
            'overview': self._generate_overview(df),
            'sensors': self._analyze_sensors(df),
            'rul_analysis': self._analyze_rul(df),
            'correlations': self._analyze_correlations(df),
            'data_quality': self._analyze_data_quality(df)
        }

        # Save JSON report
        json_path = output_dir / f"eda_report_{dataset_name.lower()}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Save text report
        text_path = output_dir / f"eda_report_{dataset_name.lower()}.txt"
        self._save_text_report(report, text_path)

        logger.info(f"EDA report saved to {output_dir}")

        return report

    def _generate_metadata(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """Generate metadata section"""
        return {
            'report_date': datetime.now().isoformat(),
            'dataset_name': dataset_name,
            'total_records': len(df),
            'total_features': len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }

    def _generate_overview(self, df: pd.DataFrame) -> dict:
        """Generate overview section"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        return {
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'numeric_features': len(numeric_cols),
            'categorical_features': len(df.columns) - len(numeric_cols),
            'missing_values': int(df.isnull().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum()),
            'engines': int(df['unit_id'].nunique()) if 'unit_id' in df.columns else 0
        }

    def _analyze_sensors(self, df: pd.DataFrame) -> dict:
        """Analyze sensor data"""
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]

        sensor_stats = {}
        for sensor in sensor_cols:
            sensor_stats[sensor] = {
                'mean': float(df[sensor].mean()),
                'std': float(df[sensor].std()),
                'min': float(df[sensor].min()),
                'max': float(df[sensor].max()),
                'range': float(df[sensor].max() - df[sensor].min()),
                'cv': float(df[sensor].std() / df[sensor].mean()) if df[sensor].mean() != 0 else 0,
                'missing': int(df[sensor].isnull().sum())
            }

        return {
            'total_sensors': len(sensor_cols),
            'sensor_statistics': sensor_stats
        }

    def _analyze_rul(self, df: pd.DataFrame) -> dict:
        """Analyze RUL distribution"""
        if 'RUL' not in df.columns:
            return {}

        rul_data = df['RUL']

        return {
            'mean': float(rul_data.mean()),
            'median': float(rul_data.median()),
            'std': float(rul_data.std()),
            'min': float(rul_data.min()),
            'max': float(rul_data.max()),
            'q25': float(rul_data.quantile(0.25)),
            'q75': float(rul_data.quantile(0.75)),
            'distribution': {
                '0-30_cycles': int((rul_data <= 30).sum()),
                '31-60_cycles': int(((rul_data > 30) & (rul_data <= 60)).sum()),
                '61-100_cycles': int(((rul_data > 60) & (rul_data <= 100)).sum()),
                '100+_cycles': int((rul_data > 100).sum())
            }
        }

    def _analyze_correlations(self, df: pd.DataFrame) -> dict:
        """Analyze feature correlations"""
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')][:10]

        if len(sensor_cols) < 2:
            return {}

        corr_matrix = df[sensor_cols].corr()

        # Find high correlations
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })

        return {
            'high_correlations': high_corr,
            'total_high_correlations': len(high_corr)
        }

    def _analyze_data_quality(self, df: pd.DataFrame) -> dict:
        """Analyze data quality"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()

        return {
            'completeness': float(1 - (missing_cells / total_cells)),
            'missing_cells': int(missing_cells),
            'total_cells': int(total_cells),
            'columns_with_missing': int((df.isnull().sum() > 0).sum())
        }

    def _save_text_report(self, report: dict, output_path: Path):
        """Save report as formatted text"""
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
            f.write("NASA C-MAPSS Turbofan Engine Dataset\n")
            f.write("="*80 + "\n\n")

            # Metadata
            f.write("METADATA\n")
            f.write("-"*80 + "\n")
            for key, value in report['metadata'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

            # Overview
            f.write("DATASET OVERVIEW\n")
            f.write("-"*80 + "\n")
            for key, value in report['overview'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

            # Sensors
            f.write("SENSOR ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Sensors: {report['sensors']['total_sensors']}\n\n")

            # RUL Analysis
            if report.get('rul_analysis'):
                f.write("RUL ANALYSIS\n")
                f.write("-"*80 + "\n")
                for key, value in report['rul_analysis'].items():
                    if key != 'distribution':
                        f.write(f"{key}: {value}\n")
                f.write("\nRUL Distribution:\n")
                for key, value in report['rul_analysis']['distribution'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            # Data Quality
            f.write("DATA QUALITY\n")
            f.write("-"*80 + "\n")
            for key, value in report['data_quality'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("EDA Report Generator initialized")

