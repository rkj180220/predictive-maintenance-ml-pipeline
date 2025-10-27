"""
ML Pipeline Routes - Endpoints for ML pipeline information
Author: ramkumarjayakumar
Date: 2025-10-27
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from datetime import datetime

from src.api.services.ml_pipeline_service import MLPipelineService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/ml", tags=["ML Pipeline"])

# Initialize service
project_root = Path(__file__).parent.parent.parent.parent
ml_service = MLPipelineService(project_root)


@router.get("/evaluation-results")
async def get_evaluation_results():
    """
    Get model evaluation results including all performance metrics

    Returns:
        Evaluation metrics for all trained models (accuracy, precision, recall, F1, ROC-AUC, etc.)
    """
    logger.info("Retrieving ML evaluation results")

    try:
        data = ml_service.get_evaluation_results()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving evaluation results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance")
async def get_feature_importance():
    """
    Get feature importance scores for all models

    Returns:
        Top features ranked by importance for training and test datasets
    """
    logger.info("Retrieving feature importance data")

    try:
        data = ml_service.get_feature_importance()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mlops-metrics")
async def get_mlops_metrics(model: str = Query(None, description="Optional model name to filter (e.g., 'xgboost', 'random_forest')")):
    """
    Get MLOps monitoring metrics for models

    Args:
        model: Optional model name to filter metrics

    Returns:
        MLOps metrics for specified model or all models with timestamps
    """
    logger.info(f"Retrieving MLOps metrics for model: {model or 'all'}")

    try:
        data = ml_service.get_mlops_metrics(model_name=model)
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving MLOps metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mlops-all-metrics")
async def get_mlops_all_metrics():
    """
    Get aggregated MLOps metrics for all models

    Returns:
        Latest aggregated metrics from all model monitoring sessions
    """
    logger.info("Retrieving aggregated MLOps metrics")

    try:
        data = ml_service.get_mlops_all_metrics()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving aggregated MLOps metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard-data")
async def get_mlops_dashboard():
    """
    Get MLOps dashboard data with model health and metrics

    Returns:
        Dashboard data including model status and key performance indicators
    """
    logger.info("Retrieving MLOps dashboard data")

    try:
        data = ml_service.get_mlops_dashboard_data()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-comparison")
async def get_model_comparison():
    """
    Get comparison between all trained models

    Returns:
        Side-by-side comparison of model performance metrics with best model identified
    """
    logger.info("Retrieving model comparison")

    try:
        data = ml_service.get_model_comparison()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving model comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline-logs")
async def get_pipeline_logs(limit: int = Query(10, ge=1, le=50, description="Number of recent logs to retrieve")):
    """
    Get recent ML pipeline execution logs

    Args:
        limit: Number of recent logs to retrieve (default: 10, max: 50)

    Returns:
        Recent pipeline execution logs with timestamps and details
    """
    logger.info(f"Retrieving pipeline logs (limit: {limit})")

    try:
        data = ml_service.get_pipeline_logs(limit=limit)
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving pipeline logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-health")
async def get_model_health():
    """
    Get current model health status

    Returns:
        Model health scores, status, and recommendations for all models
    """
    logger.info("Retrieving model health status")

    try:
        data = ml_service.get_model_health_status()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving model health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all-details")
async def get_all_ml_details():
    """
    Get all ML pipeline information in a single response

    Returns:
        Complete ML pipeline details including evaluation, features, comparison, health, and logs
    """
    logger.info("Retrieving all ML pipeline details")

    try:
        data = ml_service.get_all_ml_pipeline_details()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving all ML details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

