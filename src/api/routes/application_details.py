"""
Application Details Routes
Author: ramkumarjayakumar
Date: 2025-10-27

Activity 3.1: Retrieve Key Application Details
Activity 3.2: Display Application Details (4+ details)
"""

import logging
from fastapi import APIRouter, HTTPException
from pathlib import Path
from datetime import datetime

from src.api.services.pipeline_service import PipelineService
from src.api.schemas import HealthResponse

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["Application Details"])

# Initialize services
project_root = Path(__file__).parent.parent.parent.parent
pipeline_service = PipelineService(project_root)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    API Health Check
    Returns:
        Health status of the API
    """
    logger.info("Health check requested")
    return HealthResponse(
        status="healthy",
        message="Predictive Maintenance API is running",
        timestamp=datetime.now().isoformat()
    )


@router.get("/application-details")
async def get_all_application_details():
    """
    Activity 3.2: Display Application Details

    Returns all 4+ required application details in one response:
    1. Pipeline Information
    2. Model Information
    3. Performance Metrics
    4. Data Pipeline Status
    5. System Information (Bonus)
    """
    logger.info("Retrieving all application details")

    try:
        details = pipeline_service.get_all_application_details()

        return {
            "status": "success",
            "data": {
                "detail_1_pipeline": details['pipeline'],
                "detail_2_models": details['models'],
                "detail_3_metrics": details['metrics'],
                "detail_4_data_pipeline": details['data_pipeline'],
                "detail_5_system": details['system']
            },
            "timestamp": datetime.now().isoformat(),
            "message": "All application details retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error retrieving application details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/info")
async def get_pipeline_info():
    """
    Activity 3.1: Retrieve Pipeline Information

    Returns:
        Pipeline metadata, status, and configuration
    """
    logger.info("Retrieving pipeline information")

    try:
        info = pipeline_service.get_pipeline_info()
        return {
            "status": "success",
            "data": info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving pipeline info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/info")
async def get_models_info():
    """
    Activity 3.1: Retrieve Model Information

    Returns:
        Available models, best model, and performance comparison
    """
    logger.info("Retrieving model information")

    try:
        info = pipeline_service.get_model_info()
        return {
            "status": "success",
            "data": info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_performance_metrics():
    """
    Activity 3.1: Retrieve Performance Metrics

    Returns:
        Model accuracy, precision, recall, F1-score, business impact
    """
    logger.info("Retrieving performance metrics")

    try:
        metrics = pipeline_service.get_performance_metrics()
        return {
            "status": "success",
            "data": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-pipeline")
async def get_data_pipeline_status():
    """
    Activity 3.1: Retrieve Data Pipeline Status

    Returns:
        Data pipeline status, activities, records processed, features engineered
    """
    logger.info("Retrieving data pipeline status")

    try:
        status = pipeline_service.get_data_pipeline_status()
        return {
            "status": "success",
            "data": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving data pipeline status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system")
async def get_system_information():
    """
    Activity 3.1: Retrieve System Information (Bonus)

    Returns:
        Project structure, directories, deployment status
    """
    logger.info("Retrieving system information")

    try:
        info = pipeline_service.get_system_information()
        return {
            "status": "success",
            "data": info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving system info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def root():
    """
    Root endpoint - API information
    """
    logger.info("Root endpoint accessed")

    return {
        "service": "Predictive Maintenance ML Pipeline API",
        "version": "1.0.0",
        "sub_objectives": {
            "1": "Data Pipeline (Completed)",
            "2": "ML Pipeline (Completed)",
            "3": "API Access (Current)"
        },
        "activities": {
            "3.1": "Retrieve Key Application Details - ✓ Implemented",
            "3.2": "Display Application Details (4+ details) - ✓ Implemented"
        },
        "available_endpoints": {
            "health": "GET /api/v1/health",
            "all_details": "GET /api/v1/application-details",
            "pipeline_info": "GET /api/v1/pipeline/info",
            "models_info": "GET /api/v1/models/info",
            "metrics": "GET /api/v1/metrics",
            "data_pipeline": "GET /api/v1/data-pipeline",
            "system": "GET /api/v1/system",
            "docs": "/docs (Swagger UI)",
            "redoc": "/redoc (ReDoc)"
        },
        "timestamp": datetime.now().isoformat()
    }

