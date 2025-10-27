"""
Data Pipeline Routes - Endpoints for data pipeline information
Author: ramkumarjayakumar
Date: 2025-10-27
"""

import logging
from fastapi import APIRouter, HTTPException
from pathlib import Path
from datetime import datetime

from src.api.services.data_pipeline_service import DataPipelineService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/data", tags=["Data Pipeline"])

# Initialize service
project_root = Path(__file__).parent.parent.parent.parent
data_service = DataPipelineService(project_root)


@router.get("/correlation-analysis")
async def get_correlation_analysis():
    """
    Get correlation analysis data from logs

    Returns:
        Correlation coefficients between all features
    """
    logger.info("Retrieving correlation analysis data")

    try:
        data = data_service.get_correlation_analysis()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving correlation analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistical-analysis")
async def get_statistical_analysis():
    """
    Get statistical analysis including distribution statistics

    Returns:
        Statistical summaries for all features
    """
    logger.info("Retrieving statistical analysis data")

    try:
        data = data_service.get_statistical_analysis()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving statistical analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validation-report")
async def get_data_validation():
    """
    Get data validation report

    Returns:
        Data quality and validation details
    """
    logger.info("Retrieving data validation report")

    try:
        data = data_service.get_data_validation_report()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving validation report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift-analysis")
async def get_data_drift():
    """
    Get latest data drift analysis

    Returns:
        Data drift metrics and statistics
    """
    logger.info("Retrieving data drift analysis")

    try:
        data = data_service.get_data_drift_analysis()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving data drift analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift-history")
async def get_drift_history():
    """
    Get complete data drift analysis history

    Returns:
        All data drift analyses in chronological order
    """
    logger.info("Retrieving data drift history")

    try:
        data = data_service.get_all_data_drift_history()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving drift history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline-metrics")
async def get_pipeline_metrics():
    """
    Get data pipeline execution metrics

    Returns:
        Pipeline execution statistics and timing information
    """
    logger.info("Retrieving pipeline metrics")

    try:
        data = data_service.get_pipeline_execution_metrics()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving pipeline metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quality-summary")
async def get_data_quality_summary():
    """
    Get data quality summary combining multiple analyses

    Returns:
        Comprehensive data quality assessment
    """
    logger.info("Retrieving data quality summary")

    try:
        data = data_service.get_data_quality_summary()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving quality summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all-details")
async def get_all_data_details():
    """
    Get all data pipeline information in a single response

    Returns:
        Complete data pipeline details including all analyses
    """
    logger.info("Retrieving all data pipeline details")

    try:
        data = data_service.get_all_data_pipeline_details()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving all data details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

