"""
Main Pipeline Orchestrator for Predictive Maintenance
Author: ramkumarjayakumar
Date: 2025-10-18

This is the main pipeline that orchestrates all components:
- Data Ingestion (Kaggle download)
- Data Validation
- Data Preprocessing
- Feature Engineering
- EDA (Exploratory Data Analysis)
- Monitoring and Logging
"""

import sys
import logging
from pathlib import Path
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration
from config.settings import *

# Import data ingestion components
from data_ingestion.kaggle_downloader import KaggleDownloader
from data_ingestion.data_validator import DataValidator
from data_ingestion.data_loader import CMAPSSDataLoader

# Import preprocessing components
from preprocessing.data_cleaner import DataCleaner
from preprocessing.feature_engineer import FeatureEngineer
from preprocessing.normalizer import DataNormalizer

# Import EDA components
from eda.statistical_analysis import StatisticalAnalyzer
from eda.correlation_analysis import CorrelationAnalyzer
from eda.visualization import DataVisualizer
from eda.feature_importance import FeatureImportanceAnalyzer

# Import DataOps components
from dataops.logger import setup_pipeline_logging
from dataops.monitoring import PipelineMonitor
from dataops.pipeline_scheduler import PipelineScheduler
from dataops.dashboard import PipelineDashboard

# Import business understanding
from business_understanding.problem_definition import BusinessProblemDefinition


class PredictiveMaintenancePipeline:
    """
    Main pipeline orchestrator for predictive maintenance system
    Integrates all components into a cohesive workflow
    """

    def __init__(self):
        """Initialize pipeline"""
        # Setup logging
        self.pipeline_logger = setup_pipeline_logging(LOGS_DIR, DATAOPS_CONFIG)
        self.logger = self.pipeline_logger.get_logger()

        # Initialize monitoring
        self.monitor = PipelineMonitor(PIPELINE_METRICS)

        # Initialize dashboard
        self.dashboard = PipelineDashboard(BASE_DIR / "dashboards")

        # Initialize components
        self.downloader = KaggleDownloader(RAW_DATA_DIR)
        self.validator = DataValidator(COLUMN_NAMES)
        self.loader = CMAPSSDataLoader(RAW_DATA_DIR, COLUMN_NAMES)
        self.cleaner = DataCleaner(PREPROCESSING_CONFIG)
        self.feature_engineer = FeatureEngineer(FEATURE_CONFIG)
        self.normalizer = DataNormalizer(PREPROCESSING_CONFIG)
        self.stat_analyzer = StatisticalAnalyzer()
        self.corr_analyzer = CorrelationAnalyzer(EDA_CONFIG)
        self.visualizer = DataVisualizer(EDA_CONFIG)
        self.importance_analyzer = FeatureImportanceAnalyzer()

        self.logger.info("Pipeline initialized successfully")

    def run_full_pipeline(self):
        """Execute complete pipeline"""
        self.pipeline_logger.log_pipeline_start("Predictive Maintenance Pipeline")
        execution_id = self.monitor.start_execution()

        start_time = time.time()

        try:
            # Stage 1: Business Understanding
            self._run_business_understanding()

            # Stage 2: Data Ingestion
            datasets = self._run_data_ingestion()

            if not datasets:
                raise Exception("Data ingestion failed")

            # Stage 3: Data Loading and Merging
            train_df, test_df = self._run_data_loading(datasets)

            # Stage 4: Data Validation
            self._run_data_validation(train_df, test_df)

            # Stage 5: Data Preprocessing
            train_clean, test_clean = self._run_preprocessing(train_df, test_df)

            # Stage 6: Feature Engineering
            train_features, test_features = self._run_feature_engineering(train_clean, test_clean)

            # Stage 7: Data Normalization
            train_normalized, test_normalized = self._run_normalization(train_features, test_features)

            # Stage 8: Exploratory Data Analysis
            self._run_eda(train_normalized)

            # Stage 9: Feature Importance Analysis
            self._run_feature_importance(train_normalized)

            # Stage 10: Save Processed Data
            self._save_processed_data(train_normalized, test_normalized)

            # Pipeline completed successfully
            duration = time.time() - start_time
            self.monitor.end_execution("SUCCESS")
            self.pipeline_logger.log_pipeline_end("Predictive Maintenance Pipeline", "SUCCESS", duration)

            # Generate dashboard
            self._generate_dashboard()

            self.logger.info("✅ Pipeline completed successfully!")
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.monitor.end_execution("FAILED")
            self.pipeline_logger.log_pipeline_end("Predictive Maintenance Pipeline", "FAILED", duration)
            self.logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            return False

    def _run_business_understanding(self):
        """Run business understanding stage"""
        self.pipeline_logger.log_stage_start("Business Understanding")
        self.monitor.start_stage("Business Understanding")
        start_time = time.time()

        try:
            business_def = BusinessProblemDefinition()

            # Create docs directory
            docs_dir = BASE_DIR / "docs"
            docs_dir.mkdir(exist_ok=True)

            # Save reports
            business_def.save_report(docs_dir / "business_problem_definition.txt")
            business_def.export_json(docs_dir / "business_problem_definition.json")

            self.logger.info("Business problem definition created")

            duration = time.time() - start_time
            self.monitor.end_stage("Business Understanding", "SUCCESS")
            self.pipeline_logger.log_stage_end("Business Understanding", "SUCCESS", duration)

        except Exception as e:
            duration = time.time() - start_time
            self.monitor.end_stage("Business Understanding", "FAILED")
            self.pipeline_logger.log_stage_end("Business Understanding", "FAILED", duration)
            raise

    def _run_data_ingestion(self):
        """Run data ingestion stage"""
        self.pipeline_logger.log_stage_start("Data Ingestion")
        self.monitor.start_stage("Data Ingestion")
        start_time = time.time()

        try:
            # Download dataset
            success = self.downloader.download_dataset()

            if not success:
                self.logger.warning("Dataset download failed or skipped - using existing files")

            # Get file info
            file_info = self.downloader.get_file_info()
            self.logger.info(f"Available data files: {file_info['total_files']}")

            for file in file_info['files']:
                self.logger.info(f"  - {file['name']} ({file['size_mb']} MB)")

            duration = time.time() - start_time
            self.monitor.record_metric("files_downloaded", file_info['total_files'], "Data Ingestion")
            self.monitor.end_stage("Data Ingestion", "SUCCESS")
            self.pipeline_logger.log_stage_end("Data Ingestion", "SUCCESS", duration)

            return True

        except Exception as e:
            duration = time.time() - start_time
            self.monitor.end_stage("Data Ingestion", "FAILED")
            self.pipeline_logger.log_stage_end("Data Ingestion", "FAILED", duration)
            raise

    def _run_data_loading(self, datasets):
        """Run data loading stage"""
        self.pipeline_logger.log_stage_start("Data Loading")
        self.monitor.start_stage("Data Loading")
        start_time = time.time()

        try:
            # Load all datasets
            loaded_datasets = self.loader.load_all_datasets()

            # Merge datasets
            train_df, test_df = self.loader.merge_datasets(loaded_datasets)

            # Add failure labels
            train_df = self.loader.create_failure_labels(train_df, threshold=30)
            test_df = self.loader.create_failure_labels(test_df, threshold=30)

            # Log statistics
            train_summary = self.loader.get_dataset_summary(train_df, "Training")
            test_summary = self.loader.get_dataset_summary(test_df, "Test")

            self.logger.info(f"Training data: {train_summary['total_records']:,} records, "
                           f"{train_summary['total_engines']} engines")
            self.logger.info(f"Test data: {test_summary['total_records']:,} records, "
                           f"{test_summary['total_engines']} engines")

            duration = time.time() - start_time
            self.monitor.record_metric("train_records", train_summary['total_records'], "Data Loading")
            self.monitor.record_metric("test_records", test_summary['total_records'], "Data Loading")
            self.monitor.end_stage("Data Loading", "SUCCESS")
            self.pipeline_logger.log_stage_end("Data Loading", "SUCCESS", duration)

            return train_df, test_df

        except Exception as e:
            duration = time.time() - start_time
            self.monitor.end_stage("Data Loading", "FAILED")
            self.pipeline_logger.log_stage_end("Data Loading", "FAILED", duration)
            raise

    def _run_data_validation(self, train_df, test_df):
        """Run data validation stage"""
        self.pipeline_logger.log_stage_start("Data Validation")
        self.monitor.start_stage("Data Validation")
        start_time = time.time()

        try:
            # Validate training data
            train_results = self.validator.validate_dataframe(train_df, "Training")
            test_results = self.validator.validate_dataframe(test_df, "Test")

            # Save validation report
            self.validator.save_validation_report(LOGS_DIR / "data_validation_report.txt")

            # Record quality scores
            self.monitor.record_data_quality(train_results['quality_score'], train_results)

            self.logger.info(f"Training data quality: {train_results['quality_score']:.2%}")
            self.logger.info(f"Test data quality: {test_results['quality_score']:.2%}")

            duration = time.time() - start_time
            self.monitor.end_stage("Data Validation", "SUCCESS")
            self.pipeline_logger.log_stage_end("Data Validation", "SUCCESS", duration)

        except Exception as e:
            duration = time.time() - start_time
            self.monitor.end_stage("Data Validation", "FAILED")
            self.pipeline_logger.log_stage_end("Data Validation", "FAILED", duration)
            raise

    def _run_preprocessing(self, train_df, test_df):
        """Run preprocessing stage"""
        self.pipeline_logger.log_stage_start("Data Preprocessing")
        self.monitor.start_stage("Data Preprocessing")
        start_time = time.time()

        try:
            # Clean training data
            train_clean = self.cleaner.clean_dataset(train_df, "Training")

            # Clean test data
            test_clean = self.cleaner.clean_dataset(test_df, "Test")

            self.logger.info(f"Cleaned training data: {train_clean.shape}")
            self.logger.info(f"Cleaned test data: {test_clean.shape}")

            duration = time.time() - start_time
            self.monitor.end_stage("Data Preprocessing", "SUCCESS")
            self.pipeline_logger.log_stage_end("Data Preprocessing", "SUCCESS", duration)

            return train_clean, test_clean

        except Exception as e:
            duration = time.time() - start_time
            self.monitor.end_stage("Data Preprocessing", "FAILED")
            self.pipeline_logger.log_stage_end("Data Preprocessing", "FAILED", duration)
            raise

    def _run_feature_engineering(self, train_df, test_df):
        """Run feature engineering stage"""
        self.pipeline_logger.log_stage_start("Feature Engineering")
        self.monitor.start_stage("Feature Engineering")
        start_time = time.time()

        try:
            # Engineer features for training data
            train_features = self.feature_engineer.engineer_features(train_df, "Training")

            # Engineer features for test data
            test_features = self.feature_engineer.engineer_features(test_df, "Test")

            # Get feature summary
            train_summary = self.feature_engineer.get_feature_summary(train_features)

            self.logger.info(f"Feature engineering complete:")
            self.logger.info(f"  Total features: {train_summary['total_features']}")
            self.logger.info(f"  Rolling features: {train_summary['rolling_features']}")
            self.logger.info(f"  Lag features: {train_summary['lag_features']}")
            self.logger.info(f"  Trend features: {train_summary['trend_features']}")

            duration = time.time() - start_time
            self.monitor.record_metric("total_features", train_summary['total_features'],
                                      "Feature Engineering")
            self.monitor.end_stage("Feature Engineering", "SUCCESS")
            self.pipeline_logger.log_stage_end("Feature Engineering", "SUCCESS", duration)

            return train_features, test_features

        except Exception as e:
            duration = time.time() - start_time
            self.monitor.end_stage("Feature Engineering", "FAILED")
            self.pipeline_logger.log_stage_end("Feature Engineering", "FAILED", duration)
            raise

    def _run_normalization(self, train_df, test_df):
        """Run normalization stage"""
        self.pipeline_logger.log_stage_start("Data Normalization")
        self.monitor.start_stage("Data Normalization")
        start_time = time.time()

        try:
            # Get feature columns
            feature_cols = self.feature_engineer.get_feature_importance_candidates(train_df)

            # Normalize training data
            train_normalized = self.normalizer.fit_transform(train_df, feature_cols, "Training")

            # Normalize test data using training scaler
            test_normalized = self.normalizer.transform(test_df, feature_cols, "Training")

            # Save scalers
            self.normalizer.save_scalers(PROCESSED_DATA_DIR / "scalers.pkl")

            self.logger.info(f"Normalized {len(feature_cols)} features")

            duration = time.time() - start_time
            self.monitor.end_stage("Data Normalization", "SUCCESS")
            self.pipeline_logger.log_stage_end("Data Normalization", "SUCCESS", duration)

            return train_normalized, test_normalized

        except Exception as e:
            duration = time.time() - start_time
            self.monitor.end_stage("Data Normalization", "FAILED")
            self.pipeline_logger.log_stage_end("Data Normalization", "FAILED", duration)
            raise

    def _run_eda(self, train_df):
        """Run exploratory data analysis stage"""
        self.pipeline_logger.log_stage_start("Exploratory Data Analysis")
        self.monitor.start_stage("EDA")
        start_time = time.time()

        try:
            # Create visualizations directory
            viz_dir = BASE_DIR / "visualizations"
            viz_dir.mkdir(exist_ok=True)

            # OPTIMIZATION: Only analyze key features, not all 7,576 features
            # Select only original sensors + RUL + key metadata for EDA
            key_columns = ['unit_id', 'time_cycles', 'RUL', 'failure_label', 'dataset']
            sensor_cols = [col for col in train_df.columns if col.startswith('sensor_') and '_' not in col[7:]]  # Original sensors only
            key_columns.extend(sensor_cols[:21])  # Max 21 original sensors

            # Create a subset for EDA (much faster)
            train_eda = train_df[key_columns].copy()
            self.logger.info(f"EDA analyzing {len(train_eda.columns)} key features (out of {len(train_df.columns)} total)")

            # Statistical analysis (on subset)
            stat_results = self.stat_analyzer.analyze_dataset(train_eda, "Training")
            self.stat_analyzer.save_report(LOGS_DIR / "statistical_analysis.txt")
            self.stat_analyzer.export_json(LOGS_DIR / "statistical_analysis.json")

            # Correlation analysis (on subset)
            corr_results = self.corr_analyzer.analyze_correlations(train_eda, "Training")
            self.corr_analyzer.save_report(LOGS_DIR / "correlation_analysis.txt")
            self.corr_analyzer.export_json(LOGS_DIR / "correlation_analysis.json")

            # Create visualizations (using subset)
            self.visualizer.create_correlation_heatmap(train_eda, viz_dir / "correlation_heatmap.png")
            self.visualizer.create_rul_distribution(train_eda, viz_dir / "rul_distribution.png")
            self.visualizer.create_sensor_distributions(train_eda, viz_dir / "sensor_distributions.png")
            self.visualizer.create_degradation_patterns(train_eda, viz_dir / "degradation_patterns.png")
            self.visualizer.create_rul_vs_sensors(train_eda, viz_dir / "rul_vs_sensors.png")
            self.visualizer.create_feature_correlation_with_rul(train_eda,
                                                               viz_dir / "feature_rul_correlation.png")

            self.logger.info("EDA completed - reports and visualizations generated")

            duration = time.time() - start_time
            self.monitor.end_stage("EDA", "SUCCESS")
            self.pipeline_logger.log_stage_end("EDA", "SUCCESS", duration)

        except Exception as e:
            duration = time.time() - start_time
            self.monitor.end_stage("EDA", "FAILED")
            self.pipeline_logger.log_stage_end("EDA", "FAILED", duration)
            raise

    def _run_feature_importance(self, train_df):
        """Run feature importance analysis"""
        self.pipeline_logger.log_stage_start("Feature Importance Analysis")
        self.monitor.start_stage("Feature Importance")
        start_time = time.time()

        try:
            # Analyze feature importance
            importance_results = self.importance_analyzer.analyze_importance(train_df, 'RUL', "Training")

            # Save reports
            self.importance_analyzer.save_report(LOGS_DIR / "feature_importance.txt")
            self.importance_analyzer.export_json(LOGS_DIR / "feature_importance.json")

            self.logger.info("Feature importance analysis completed")

            duration = time.time() - start_time
            self.monitor.end_stage("Feature Importance", "SUCCESS")
            self.pipeline_logger.log_stage_end("Feature Importance", "SUCCESS", duration)

        except Exception as e:
            duration = time.time() - start_time
            self.monitor.end_stage("Feature Importance", "FAILED")
            self.pipeline_logger.log_stage_end("Feature Importance", "FAILED", duration)
            raise

    def _save_processed_data(self, train_df, test_df):
        """Save processed data"""
        self.pipeline_logger.log_stage_start("Save Processed Data")
        self.monitor.start_stage("Save Data")
        start_time = time.time()

        try:
            self.logger.info("Saving processed data (this may take 2-3 minutes)...")

            # Save to parquet ONLY (much faster than CSV - 100x speed improvement)
            self.logger.info("Saving training data to parquet...")
            train_df.to_parquet(PROCESSED_DATA_DIR / "train_processed.parquet",
                              index=False,
                              engine='pyarrow',
                              compression='snappy')

            self.logger.info("Saving test data to parquet...")
            test_df.to_parquet(PROCESSED_DATA_DIR / "test_processed.parquet",
                             index=False,
                             engine='pyarrow',
                             compression='snappy')

            # Optional: Save smaller CSV with only key features for easy viewing
            key_cols = ['unit_id', 'time_cycles', 'RUL', 'failure_label'] + \
                       [col for col in train_df.columns if col.startswith('sensor_') and '_' not in col[7:]][:10]

            self.logger.info("Saving sample CSV with key features...")
            train_df[key_cols].to_csv(PROCESSED_DATA_DIR / "train_sample.csv", index=False)
            test_df[key_cols].to_csv(PROCESSED_DATA_DIR / "test_sample.csv", index=False)

            self.logger.info("✓ Processed data saved successfully (parquet + sample CSV)")

            duration = time.time() - start_time
            self.monitor.end_stage("Save Data", "SUCCESS")
            self.pipeline_logger.log_stage_end("Save Data", "SUCCESS", duration)

        except Exception as e:
            duration = time.time() - start_time
            self.monitor.end_stage("Save Data", "FAILED")
            self.pipeline_logger.log_stage_end("Save Data", "FAILED", duration)
            raise

    def _generate_dashboard(self):
        """Generate monitoring dashboard"""
        try:
            # Export metrics
            self.monitor.export_metrics(LOGS_DIR / "pipeline_metrics.json")

            # Get data for dashboard
            import json
            with open(LOGS_DIR / "pipeline_metrics.json", 'r') as f:
                monitor_data = json.load(f)

            scheduler_status = {
                'is_running': False,
                'execution_count': monitor_data['performance_summary']['total_executions'],
                'schedule_interval_minutes': DATAOPS_CONFIG['pipeline_schedule_minutes'],
                'last_execution_time': datetime.now().isoformat(),
                'next_run': 'Manual execution'
            }

            # Generate dashboard
            self.dashboard.generate_dashboard(monitor_data, scheduler_status)

            self.logger.info("Dashboard generated successfully")

        except Exception as e:
            self.logger.warning(f"Failed to generate dashboard: {e}")


def main():
    """Main entry point"""
    print("="*80)
    print("PREDICTIVE MAINTENANCE ML PIPELINE")
    print("NASA C-MAPSS Turbofan Engine Dataset")
    print("Author: ramkumarjayakumar")
    print("Date: 2025-10-18")
    print("="*80)
    print()

    # Create pipeline
    pipeline = PredictiveMaintenancePipeline()

    # Run pipeline
    success = pipeline.run_full_pipeline()

    if success:
        print("\n✅ Pipeline execution completed successfully!")
        print(f"\nOutputs:")
        print(f"  - Logs: {LOGS_DIR}")
        print(f"  - Processed Data: {PROCESSED_DATA_DIR}")
        print(f"  - Visualizations: {BASE_DIR / 'visualizations'}")
        print(f"  - Dashboard: {BASE_DIR / 'dashboards'}")
        print(f"  - Documentation: {BASE_DIR / 'docs'}")
    else:
        print("\n❌ Pipeline execution failed. Check logs for details.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
