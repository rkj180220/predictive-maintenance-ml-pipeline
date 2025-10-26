"""
Complete Machine Learning Pipeline Orchestrator
Author: ramkumarjayakumar
Date: 2025-10-26

Sub-Objective 2: Complete ML Pipeline Implementation
Orchestrates: Model Preparation → Training → Evaluation → MLOps Monitoring
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml_pipeline.model_preparation import ModelPreparation
from src.ml_pipeline.model_training import ModelTrainer
from src.ml_pipeline.model_evaluation import ModelEvaluator
from src.ml_pipeline.mlops_monitoring import MLOpsMonitor

# Ensure logs directory exists before setting up logging
logs_dir = project_root / 'logs'
logs_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / f'ml_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class MLPipelineOrchestrator:
    """
    Orchestrates the complete ML pipeline for predictive maintenance

    Pipeline Stages:
    1. Model Preparation (2.1)
    2. Model Training (2.2)
    3. Model Evaluation (2.3)
    4. MLOps Monitoring (2.4)
    """

    def __init__(self, project_root: Path):
        """
        Initialize ML Pipeline Orchestrator

        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / 'data' / 'processed'
        self.models_dir = self.project_root / 'models'
        self.logs_dir = self.project_root / 'logs'
        self.visualizations_dir = self.project_root / 'visualizations'

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)

        logger.info("ML Pipeline Orchestrator initialized")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Models directory: {self.models_dir}")

    def run_complete_pipeline(self, enable_cloudwatch: bool = False):
        """
        Execute the complete ML pipeline

        Args:
            enable_cloudwatch: Enable AWS CloudWatch integration

        Returns:
            Dictionary with all pipeline results
        """
        logger.info("\n" + "🚀"*50)
        logger.info("STARTING COMPLETE MACHINE LEARNING PIPELINE")
        logger.info("Sub-Objective 2: ML Pipeline for Predictive Maintenance")
        logger.info("🚀"*50 + "\n")

        pipeline_start_time = datetime.now()

        try:
            # ============================================================
            # STAGE 1: Model Preparation (Activity 2.1)
            # ============================================================
            logger.info("\n" + "="*80)
            logger.info("STAGE 1: MODEL PREPARATION (Activity 2.1)")
            logger.info("="*80)

            prep = ModelPreparation()
            prep.print_selection_report()
            rf_model, xgb_model = prep.initialize_models()

            logger.info("✅ Stage 1 Complete: Models prepared and initialized\n")

            # ============================================================
            # STAGE 2: Model Training (Activity 2.2)
            # ============================================================
            logger.info("\n" + "="*80)
            logger.info("STAGE 2: MODEL TRAINING (Activity 2.2)")
            logger.info("="*80)

            trainer = ModelTrainer(test_size=0.3, random_state=42)

            # Find processed data
            data_path = self.data_dir / 'train_processed.parquet'
            if not data_path.exists():
                data_path = self.data_dir / 'train_processed.csv'

            if not data_path.exists():
                raise FileNotFoundError(f"Processed data not found in {self.data_dir}")

            logger.info(f"Using data from: {data_path}")

            # Train models
            training_results = trainer.full_training_pipeline(
                data_path=data_path,
                rf_model=rf_model,
                xgb_model=xgb_model,
                output_dir=self.models_dir
            )

            logger.info("✅ Stage 2 Complete: Models trained successfully\n")

            # ============================================================
            # STAGE 3: Model Evaluation (Activity 2.3)
            # ============================================================
            logger.info("\n" + "="*80)
            logger.info("STAGE 3: MODEL EVALUATION (Activity 2.3)")
            logger.info("="*80)

            evaluator = ModelEvaluator()

            # Evaluate Random Forest
            rf_metrics = evaluator.evaluate_model(
                model=training_results['random_forest']['model'],
                X_test=training_results['data_split']['X_test'],
                y_test=training_results['data_split']['y_test'],
                model_name='Random Forest'
            )

            # Evaluate XGBoost
            xgb_metrics = evaluator.evaluate_model(
                model=training_results['xgboost']['model'],
                X_test=training_results['data_split']['X_test'],
                y_test=training_results['data_split']['y_test'],
                model_name='XGBoost'
            )

            # Compare models
            comparison_df = evaluator.compare_models(training_results)

            # Generate visualizations
            evaluator.plot_confusion_matrices(self.visualizations_dir)
            evaluator.plot_metrics_comparison(comparison_df, self.visualizations_dir)

            # Save results
            evaluator.save_evaluation_results(self.logs_dir)
            evaluator.generate_evaluation_report(self.logs_dir)

            logger.info("✅ Stage 3 Complete: Models evaluated successfully\n")

            # ============================================================
            # STAGE 4: MLOps Monitoring (Activity 2.4)
            # ============================================================
            logger.info("\n" + "="*80)
            logger.info("STAGE 4: MLOPS MONITORING (Activity 2.4)")
            logger.info("="*80)

            mlops_monitor = MLOpsMonitor(
                log_dir=self.logs_dir,
                enable_cloudwatch=enable_cloudwatch
            )

            # Log metrics for Random Forest
            rf_mlops_metrics = mlops_monitor.log_training_metrics(
                model_name='Random Forest',
                training_metadata=training_results['random_forest']['metadata'],
                evaluation_metrics=rf_metrics
            )

            # Log metrics for XGBoost
            xgb_mlops_metrics = mlops_monitor.log_training_metrics(
                model_name='XGBoost',
                training_metadata=training_results['xgboost']['metadata'],
                evaluation_metrics=xgb_metrics
            )

            # Monitor data drift
            drift_metrics = mlops_monitor.monitor_data_drift(
                X_train=training_results['data_split']['X_train'],
                X_test=training_results['data_split']['X_test']
            )

            # Generate dashboard data
            dashboard_data = mlops_monitor.generate_mlops_dashboard_data(
                self.logs_dir / 'mlops_dashboard_data.json'
            )

            # Save all metrics
            all_metrics_path = mlops_monitor.save_all_metrics()

            logger.info("✅ Stage 4 Complete: MLOps monitoring configured\n")

            # ============================================================
            # PIPELINE COMPLETION
            # ============================================================
            pipeline_end_time = datetime.now()
            pipeline_duration = (pipeline_end_time - pipeline_start_time).total_seconds()

            logger.info("\n" + "🎉"*50)
            logger.info("MACHINE LEARNING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("🎉"*50)

            logger.info(f"\n⏱️  Total Pipeline Duration: {pipeline_duration:.2f} seconds ({pipeline_duration/60:.2f} minutes)")
            logger.info(f"📅 Completed at: {pipeline_end_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Summary of results
            logger.info("\n📊 PIPELINE SUMMARY:")
            logger.info("="*80)
            logger.info(f"✓ Models Trained: 2 (Random Forest, XGBoost)")
            logger.info(f"✓ Training Samples: {training_results['random_forest']['metadata']['training_samples']:,}")
            logger.info(f"✓ Test Samples: {len(training_results['data_split']['X_test']):,}")
            logger.info(f"✓ Features Used: {training_results['random_forest']['metadata']['n_features']}")
            logger.info(f"\n📈 BEST MODEL PERFORMANCE:")

            # Determine best model
            if rf_metrics['f1_score'] >= xgb_metrics['f1_score']:
                best_model = 'Random Forest'
                best_metrics = rf_metrics
            else:
                best_model = 'XGBoost'
                best_metrics = xgb_metrics

            logger.info(f"   🏆 Best Model: {best_model}")
            logger.info(f"   • Accuracy:  {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
            logger.info(f"   • Precision: {best_metrics['precision']:.4f} ({best_metrics['precision']*100:.2f}%)")
            logger.info(f"   • Recall:    {best_metrics['recall']:.4f} ({best_metrics['recall']*100:.2f}%)")
            logger.info(f"   • F1-Score:  {best_metrics['f1_score']:.4f} ({best_metrics['f1_score']*100:.2f}%)")

            logger.info(f"\n📁 OUTPUT FILES:")
            logger.info(f"   • Models: {self.models_dir}")
            logger.info(f"   • Logs: {self.logs_dir}")
            logger.info(f"   • Visualizations: {self.visualizations_dir}")

            logger.info("\n" + "="*80)
            logger.info("🎯 SUB-OBJECTIVE 2 COMPLETED")
            logger.info("All ML pipeline activities (2.1-2.4) executed successfully")
            logger.info("="*80 + "\n")

            # Return comprehensive results
            return {
                'status': 'SUCCESS',
                'pipeline_duration_seconds': pipeline_duration,
                'timestamp': pipeline_end_time.isoformat(),
                'model_preparation': {
                    'algorithms_selected': ['Random Forest', 'XGBoost'],
                    'report_generated': True
                },
                'model_training': {
                    'random_forest': training_results['random_forest'],
                    'xgboost': training_results['xgboost'],
                    'data_split': {
                        'train_size': len(training_results['data_split']['X_train']),
                        'test_size': len(training_results['data_split']['X_test']),
                        'n_features': training_results['data_split']['X_train'].shape[1]
                    }
                },
                'model_evaluation': {
                    'random_forest': rf_metrics,
                    'xgboost': xgb_metrics,
                    'comparison': comparison_df.to_dict('records'),
                    'best_model': best_model
                },
                'mlops_monitoring': {
                    'random_forest_mlops': rf_mlops_metrics,
                    'xgboost_mlops': xgb_mlops_metrics,
                    'data_drift': drift_metrics,
                    'dashboard_data': dashboard_data,
                    'cloudwatch_enabled': enable_cloudwatch
                },
                'output_paths': {
                    'models_dir': str(self.models_dir),
                    'logs_dir': str(self.logs_dir),
                    'visualizations_dir': str(self.visualizations_dir),
                    'mlops_metrics': str(all_metrics_path)
                }
            }

        except Exception as e:
            logger.error(f"❌ Pipeline failed with error: {str(e)}", exc_info=True)
            return {
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


def main():
    """Main entry point for ML pipeline"""

    # Initialize orchestrator
    project_root = Path(__file__).parent.parent.parent
    orchestrator = MLPipelineOrchestrator(project_root)

    # Run complete pipeline
    results = orchestrator.run_complete_pipeline(enable_cloudwatch=False)

    if results['status'] == 'SUCCESS':
        logger.info("\n✅ ML Pipeline executed successfully!")
        logger.info(f"Check logs at: {results['output_paths']['logs_dir']}")
        logger.info(f"Check models at: {results['output_paths']['models_dir']}")
        logger.info(f"Check visualizations at: {results['output_paths']['visualizations_dir']}")
    else:
        logger.error("\n❌ ML Pipeline failed!")
        logger.error(f"Error: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()

