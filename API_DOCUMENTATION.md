"""
Predictive Maintenance ML Pipeline - Comprehensive API Documentation
Author: ramkumarjayakumar
Date: 2025-10-27

This document provides a complete guide to all available API endpoints
for accessing data pipeline and ML pipeline information from logs.
"""

# ============================================================================
# API STRUCTURE
# ============================================================================

"""
The API is organized into three main categories:

1. APPLICATION DETAILS ENDPOINTS (/api/v1/*)
   - General application information
   - Pipeline and model metadata
   - System information
   
2. DATA PIPELINE ENDPOINTS (/api/v1/data/*)
   - Correlation analysis
   - Statistical analysis
   - Data validation reports
   - Data drift detection
   - Pipeline metrics
   
3. ML PIPELINE ENDPOINTS (/api/v1/ml/*)
   - Model evaluation results
   - Feature importance
   - MLOps monitoring metrics
   - Model comparison
   - Model health status
"""

# ============================================================================
# APPLICATION DETAILS ENDPOINTS
# ============================================================================

"""
GET /api/v1/health
- Health check of the API
- Response: {status: "healthy", message: "...", timestamp: "..."}

GET /api/v1/application-details
- Get all 4+ required application details
- Returns: Pipeline info + Models + Metrics + Data Pipeline + System

GET /api/v1/pipeline/info
- Get pipeline metadata and status
- Returns: Pipeline name, status, last run time, version

GET /api/v1/models/info
- Get available models and their metadata
- Returns: Best model, models available, performance

GET /api/v1/metrics
- Get performance metrics
- Returns: Accuracy, precision, recall, F1-score, ROC-AUC

GET /api/v1/data-pipeline
- Get data pipeline execution status
- Returns: Records processed, features engineered, data quality

GET /api/v1/system
- Get system information
- Returns: Project name, directories, framework, deployment status
"""

# ============================================================================
# DATA PIPELINE ENDPOINTS (/api/v1/data/*)
# ============================================================================

"""
GET /api/v1/data/correlation-analysis
- Fetch correlation analysis from correlation_analysis.json
- Returns feature correlation matrices and coefficients
- Example response:
  {
    "status": "success",
    "data": {
      "Training": {
        "dataset_name": "Training",
        "feature_correlations": {
          "matrix": {...correlation matrix...}
        }
      }
    }
  }

GET /api/v1/data/statistical-analysis
- Fetch statistical analysis from statistical_analysis.json
- Returns distribution stats, basic stats for all features
- Example response:
  {
    "status": "success",
    "data": {
      "Training": {
        "basic_stats": {...},
        "distribution_stats": {...}
      }
    }
  }

GET /api/v1/data/validation-report
- Fetch data validation report from data_validation_report.txt
- Returns data quality validation results
- Example response:
  {
    "status": "success",
    "report": "...validation details..."
  }

GET /api/v1/data/drift-analysis
- Fetch latest data drift analysis from data_drift_*.json
- Returns data drift metrics and statistics
- Example response:
  {
    "status": "success",
    "data": {...drift analysis...},
    "latest_file": "data_drift_20251027_122310.json"
  }

GET /api/v1/data/drift-history
- Fetch complete data drift history (all data_drift_*.json files)
- Returns list of all drift analyses in reverse chronological order
- Example response:
  {
    "status": "success",
    "count": 4,
    "data": [
      {
        "timestamp": "20251027_122310",
        "file": "data_drift_20251027_122310.json",
        "data": {...}
      },
      ...
    ]
  }

GET /api/v1/data/pipeline-metrics
- Fetch data pipeline execution metrics from pipeline_metrics.json
- Returns execution times, stages, status for all runs
- Example response:
  {
    "status": "success",
    "data": {
      "executions": [
        {
          "execution_id": "...",
          "stages": [...],
          "status": "SUCCESS"
        }
      ]
    }
  }

GET /api/v1/data/quality-summary
- Get comprehensive data quality summary
- Returns combined assessment from all data analyses
- Example response:
  {
    "status": "success",
    "data": {
      "statistical_analysis": "Available",
      "validation_report": "Available",
      "drift_analysis": "Available",
      "details": {...all data...}
    }
  }

GET /api/v1/data/all-details
- Get ALL data pipeline information in one response
- Returns correlation, statistical, validation, drift, and metrics data
- Most comprehensive data pipeline endpoint
"""

# ============================================================================
# ML PIPELINE ENDPOINTS (/api/v1/ml/*)
# ============================================================================

"""
GET /api/v1/ml/evaluation-results
- Fetch model evaluation results from evaluation_results.json
- Returns accuracy, precision, recall, F1-score, confusion matrix for all models
- Example response:
  {
    "status": "success",
    "data": {
      "XGBoost": {
        "accuracy": 0.9961,
        "precision": 0.9782,
        "recall": 0.9939,
        "f1_score": 0.9860,
        "roc_auc": 0.9999,
        "confusion_matrix": [...]
      },
      "Random Forest": {...}
    }
  }

GET /api/v1/ml/feature-importance
- Fetch feature importance scores from feature_importance.json
- Returns ranked features by importance for each model and dataset
- Example response:
  {
    "status": "success",
    "data": {
      "Training": {
        "random_forest": {
          "importances": {
            "sensor_11_lag_1_cumulative_max": 0.1518,
            "sensor_2_lag_1_cumulative_max": 0.0793,
            ...
          }
        },
        "xgboost": {...}
      }
    }
  }

GET /api/v1/ml/mlops-metrics?model=<optional>
- Fetch MLOps monitoring metrics from mlops_metrics_*.json files
- Optional query parameter "model" to filter by model name (xgboost, random_forest)
- Returns latest metrics for specified or all models
- Example URLs:
  - /api/v1/ml/mlops-metrics (all models)
  - /api/v1/ml/mlops-metrics?model=xgboost
  - /api/v1/ml/mlops-metrics?model=random_forest

GET /api/v1/ml/mlops-all-metrics
- Fetch aggregated MLOps metrics from mlops_all_metrics_*.json
- Returns latest aggregated metrics for all models
- Example response:
  {
    "status": "success",
    "latest_file": "mlops_all_metrics_20251027_122310.json",
    "data": {...aggregated metrics...}
  }

GET /api/v1/ml/dashboard-data
- Fetch MLOps dashboard data from mlops_dashboard_data.json
- Returns model health scores, status, and dashboard metrics
- Example response:
  {
    "status": "success",
    "data": {
      "last_updated": "2025-10-27T12:23:10.476130",
      "models": [
        {
          "model_name": "XGBoost",
          "accuracy": 0.9961,
          "health_status": "EXCELLENT",
          "health_score": 100
        }
      ]
    }
  }

GET /api/v1/ml/model-comparison
- Get side-by-side comparison of all trained models
- Returns best model identified and metric comparison
- Example response:
  {
    "status": "success",
    "best_model": "XGBoost",
    "comparison": {
      "XGBoost": {
        "accuracy": 0.9961,
        "f1_score": 0.9860,
        ...
      },
      "Random Forest": {...}
    }
  }

GET /api/v1/ml/pipeline-logs?limit=10
- Fetch recent ML pipeline execution logs
- Optional query parameter "limit" (default: 10, max: 50)
- Returns recent logs with timestamps and preview
- Example URLs:
  - /api/v1/ml/pipeline-logs (last 10 logs)
  - /api/v1/ml/pipeline-logs?limit=5
  - /api/v1/ml/pipeline-logs?limit=50

GET /api/v1/ml/model-health
- Get current model health status for all models
- Returns health scores and recommendations
- Example response:
  {
    "status": "success",
    "data": {
      "last_updated": "...",
      "total_models": 2,
      "models": [
        {
          "name": "XGBoost",
          "health_status": "EXCELLENT",
          "health_score": 100,
          "accuracy": 0.9961
        }
      ]
    }
  }

GET /api/v1/ml/all-details
- Get ALL ML pipeline information in one response
- Returns evaluation results, features, comparison, health, logs, and metrics
- Most comprehensive ML pipeline endpoint
"""

# ============================================================================
# DATA SOURCE INFORMATION
# ============================================================================

"""
All endpoints fetch data from the following log files:

DATA PIPELINE SOURCES:
- correlation_analysis.json - Feature correlation matrices
- statistical_analysis.json - Distribution statistics
- data_validation_report.txt - Data quality validation
- data_drift_*.json - Data drift detection results (multiple timestamps)
- pipeline_metrics.json - Pipeline execution metrics
- pipeline_run.log - Pipeline execution logs

ML PIPELINE SOURCES:
- evaluation_results.json - Model evaluation metrics
- feature_importance.json - Feature importance scores
- mlops_metrics_*.json - Per-model monitoring metrics
- mlops_all_metrics_*.json - Aggregated metrics for all models
- mlops_dashboard_data.json - Dashboard display data
- ml_pipeline_*.log - Pipeline execution logs
- pipeline_metrics.json - Shared execution metrics
"""

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
EXAMPLE 1: Get all application details
curl http://127.0.0.1:8000/api/v1/application-details

EXAMPLE 2: Get data correlation analysis
curl http://127.0.0.1:8000/api/v1/data/correlation-analysis

EXAMPLE 3: Get ML model evaluation results
curl http://127.0.0.1:8000/api/v1/ml/evaluation-results

EXAMPLE 4: Get XGBoost MLOps metrics specifically
curl http://127.0.0.1:8000/api/v1/ml/mlops-metrics?model=xgboost

EXAMPLE 5: Get top 5 pipeline logs
curl http://127.0.0.1:8000/api/v1/ml/pipeline-logs?limit=5

EXAMPLE 6: Get all data pipeline details
curl http://127.0.0.1:8000/api/v1/data/all-details

EXAMPLE 7: Get all ML pipeline details
curl http://127.0.0.1:8000/api/v1/ml/all-details

EXAMPLE 8: Get data drift history
curl http://127.0.0.1:8000/api/v1/data/drift-history

EXAMPLE 9: Get model health status
curl http://127.0.0.1:8000/api/v1/ml/model-health

EXAMPLE 10: Get model comparison
curl http://127.0.0.1:8000/api/v1/ml/model-comparison
"""

# ============================================================================
# API RESPONSE STRUCTURE
# ============================================================================

"""
All endpoints return JSON responses with the following structure:

SUCCESS RESPONSE:
{
  "status": "success",
  "data": {
    ... endpoint-specific data ...
  },
  "timestamp": "2025-10-27T12:30:00.000000"
}

ERROR RESPONSE:
{
  "status": "error",
  "message": "Error description",
  "timestamp": "2025-10-27T12:30:00.000000"
}
"""

# ============================================================================
# ENDPOINT ORGANIZATION
# ============================================================================

"""
PATH PREFIXES:
- /api/v1/ - All application detail endpoints (general info)
- /api/v1/data/ - All data pipeline endpoints (EDA, validation, drift)
- /api/v1/ml/ - All ML pipeline endpoints (evaluation, metrics, health)

ENDPOINT COUNT:
- Application Details: 7 endpoints
- Data Pipeline: 8 endpoints
- ML Pipeline: 9 endpoints
- Total: 24+ endpoints

TOTAL DATA SOURCES:
- 30+ JSON log files
- Multiple timestamp-based drift/metrics tracking
- Real-time data fetching (no caching)
"""

