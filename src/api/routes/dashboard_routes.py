"""
Dashboard Routes - API endpoints for dashboard operations
Author: ramkumarjayakumar
Date: 2025-10-27
"""

import logging
from fastapi import APIRouter, HTTPException
from pathlib import Path
from datetime import datetime

from src.api.services.dashboard_service import DashboardService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/dashboard", tags=["Dashboard"])

# Initialize service
project_root = Path(__file__).parent.parent.parent.parent
dashboard_service = DashboardService(project_root)


@router.get("/overview")
async def get_dashboard_overview():
    """
    Get complete dashboard overview with all key metrics

    Returns:
        Dashboard overview including system health, data pipeline status, and ML metrics
    """
    logger.info("Retrieving dashboard overview")

    try:
        data = dashboard_service.get_dashboard_overview()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving dashboard overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger/data-pipeline")
async def trigger_data_pipeline():
    """
    Trigger data pipeline execution

    Returns:
        Execution status and process ID
    """
    logger.info("Triggering data pipeline from dashboard")

    try:
        result = dashboard_service.trigger_data_pipeline()
        return {
            "status": result.get("status"),
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error triggering data pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger/ml-pipeline")
async def trigger_ml_pipeline():
    """
    Trigger ML pipeline execution

    Returns:
        Execution status and process ID
    """
    logger.info("Triggering ML pipeline from dashboard")

    try:
        result = dashboard_service.trigger_ml_pipeline()
        return {
            "status": result.get("status"),
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error triggering ML pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system-stats")
async def get_system_stats():
    """
    Get system statistics and resource usage

    Returns:
        CPU, memory, and disk usage statistics
    """
    logger.info("Retrieving system statistics")

    try:
        data = dashboard_service.get_system_stats()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving system stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

