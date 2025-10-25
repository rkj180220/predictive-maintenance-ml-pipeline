"""
Data Validator for NASA C-MAPSS Dataset
Author: ramkumarjayakumar
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validation for NASA C-MAPSS dataset
    Performs quality checks, schema validation, and data integrity verification
    """

    def __init__(self, expected_columns: List[str]):
        """
        Initialize data validator

        Args:
            expected_columns: List of expected column names
        """
        self.expected_columns = expected_columns
        self.validation_results = {}

    def validate_dataframe(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Perform comprehensive validation on a dataframe

        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset for reporting

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating dataset: {dataset_name}")

        results = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "total_columns": len(df.columns),
            "validations": {}
        }

        # 1. Schema validation
        results["validations"]["schema"] = self._validate_schema(df)

        # 2. Missing values check
        results["validations"]["missing_values"] = self._check_missing_values(df)

        # 3. Data types validation
        results["validations"]["data_types"] = self._validate_data_types(df)

        # 4. Value ranges check
        results["validations"]["value_ranges"] = self._check_value_ranges(df)

        # 5. Duplicates check
        results["validations"]["duplicates"] = self._check_duplicates(df)

        # 6. Data quality score
        results["quality_score"] = self._calculate_quality_score(results["validations"])

        # 7. Overall status
        results["status"] = "PASS" if results["quality_score"] >= 0.95 else "FAIL"

        self.validation_results[dataset_name] = results

        logger.info(f"Validation complete. Quality Score: {results['quality_score']:.2%}")

        return results

    def _validate_schema(self, df: pd.DataFrame) -> Dict:
        """Validate dataframe schema"""
        actual_columns = df.columns.tolist()

        # Check if all expected columns are present
        missing_columns = [col for col in self.expected_columns if col not in actual_columns]
        extra_columns = [col for col in actual_columns if col not in self.expected_columns]

        validation = {
            "expected_columns": len(self.expected_columns),
            "actual_columns": len(actual_columns),
            "missing_columns": missing_columns,
            "extra_columns": extra_columns,
            "passed": len(missing_columns) == 0
        }

        if not validation["passed"]:
            logger.warning(f"Schema validation failed. Missing: {missing_columns}")
        else:
            logger.info("Schema validation passed")

        return validation

    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values"""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df) * 100).round(2)

        columns_with_missing = missing_counts[missing_counts > 0].to_dict()

        validation = {
            "total_missing": int(missing_counts.sum()),
            "columns_with_missing": len(columns_with_missing),
            "missing_by_column": {
                col: {
                    "count": int(count),
                    "percentage": float(missing_percentages[col])
                }
                for col, count in columns_with_missing.items()
            },
            "passed": missing_counts.sum() == 0
        }

        if validation["passed"]:
            logger.info("No missing values found")
        else:
            logger.warning(f"Found {validation['total_missing']} missing values")

        return validation

    def _validate_data_types(self, df: pd.DataFrame) -> Dict:
        """Validate data types"""
        data_types = df.dtypes.to_dict()

        # Expected: all numeric except identifiers
        numeric_columns = [col for col in df.columns if col not in ['unit_id']]

        type_issues = []
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                type_issues.append({
                    "column": col,
                    "expected": "numeric",
                    "actual": str(df[col].dtype)
                })

        validation = {
            "data_types": {col: str(dtype) for col, dtype in data_types.items()},
            "type_issues": type_issues,
            "passed": len(type_issues) == 0
        }

        if validation["passed"]:
            logger.info("Data type validation passed")
        else:
            logger.warning(f"Data type issues found: {type_issues}")

        return validation

    def _check_value_ranges(self, df: pd.DataFrame) -> Dict:
        """Check for suspicious value ranges"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        range_issues = []
        for col in numeric_cols:
            if col == 'unit_id':
                continue

            col_min = df[col].min()
            col_max = df[col].max()
            col_std = df[col].std()

            # Check for constant values (zero variance)
            if col_std == 0:
                range_issues.append({
                    "column": col,
                    "issue": "constant_value",
                    "value": col_min
                })

            # Check for infinite values
            if np.isinf(df[col]).any():
                range_issues.append({
                    "column": col,
                    "issue": "infinite_values",
                    "count": int(np.isinf(df[col]).sum())
                })

        validation = {
            "range_statistics": {
                col: {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std())
                }
                for col in numeric_cols if col != 'unit_id'
            },
            "range_issues": range_issues,
            "passed": len(range_issues) == 0
        }

        if validation["passed"]:
            logger.info("Value range validation passed")
        else:
            logger.warning(f"Value range issues found: {len(range_issues)}")

        return validation

    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate records"""
        duplicate_count = df.duplicated().sum()

        validation = {
            "duplicate_rows": int(duplicate_count),
            "duplicate_percentage": round(duplicate_count / len(df) * 100, 2),
            "passed": duplicate_count == 0
        }

        if validation["passed"]:
            logger.info("No duplicate rows found")
        else:
            logger.warning(f"Found {duplicate_count} duplicate rows")

        return validation

    def _calculate_quality_score(self, validations: Dict) -> float:
        """
        Calculate overall data quality score

        Args:
            validations: Dictionary of validation results

        Returns:
            Quality score between 0 and 1
        """
        scores = []

        # Schema validation (30% weight)
        if validations["schema"]["passed"]:
            scores.append(0.30)

        # Missing values (25% weight)
        if validations["missing_values"]["passed"]:
            scores.append(0.25)

        # Data types (20% weight)
        if validations["data_types"]["passed"]:
            scores.append(0.20)

        # Value ranges (15% weight)
        if validations["value_ranges"]["passed"]:
            scores.append(0.15)

        # Duplicates (10% weight)
        if validations["duplicates"]["passed"]:
            scores.append(0.10)

        return sum(scores)

    def generate_validation_report(self) -> str:
        """Generate human-readable validation report"""
        report = f"""
{'='*80}
DATA VALIDATION REPORT
NASA C-MAPSS Dataset
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""

        for dataset_name, results in self.validation_results.items():
            report += f"""
{'-'*80}
Dataset: {dataset_name}
{'-'*80}
Total Records: {results['total_records']:,}
Total Columns: {results['total_columns']}
Quality Score: {results['quality_score']:.2%}
Status: {results['status']}

Schema Validation: {'✓ PASSED' if results['validations']['schema']['passed'] else '✗ FAILED'}
  - Expected Columns: {results['validations']['schema']['expected_columns']}
  - Actual Columns: {results['validations']['schema']['actual_columns']}
  - Missing Columns: {results['validations']['schema']['missing_columns'] or 'None'}

Missing Values: {'✓ PASSED' if results['validations']['missing_values']['passed'] else '✗ FAILED'}
  - Total Missing: {results['validations']['missing_values']['total_missing']}
  - Columns Affected: {results['validations']['missing_values']['columns_with_missing']}

Data Types: {'✓ PASSED' if results['validations']['data_types']['passed'] else '✗ FAILED'}
  - Type Issues: {len(results['validations']['data_types']['type_issues'])}

Value Ranges: {'✓ PASSED' if results['validations']['value_ranges']['passed'] else '✗ FAILED'}
  - Range Issues: {len(results['validations']['value_ranges']['range_issues'])}

Duplicates: {'✓ PASSED' if results['validations']['duplicates']['passed'] else '✗ FAILED'}
  - Duplicate Rows: {results['validations']['duplicates']['duplicate_rows']}

"""

        report += f"""
{'='*80}
END OF VALIDATION REPORT
{'='*80}
"""
        return report

    def save_validation_report(self, filepath: Path):
        """Save validation report to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.generate_validation_report())
        logger.info(f"Validation report saved to {filepath}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    from src.config.settings import COLUMN_NAMES

    validator = DataValidator(COLUMN_NAMES)
    print("Data Validator initialized successfully")

