"""
Pydantic schemas for API requests and responses
Author: ramkumarjayakumar
Date: 2025-10-27
"""

from pydantic import BaseModel
from typing import Dict, List, Optional


class PipelineInfo(BaseModel):
    """Pipeline information schema"""
    pipeline_name: str
    sub_objectives: List[str]
    status: str
    last_run: str


class ModelMetrics(BaseModel):
    """Model performance metrics schema"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float] = None


class ModelInfo(BaseModel):
    """Model information schema"""
    best_model: str
    models_available: List[str]
    model_metrics: Dict[str, float]
    model_location: str


class DataPipelineStatus(BaseModel):
    """Data pipeline status schema"""
    status: str
    activities: List[str]
    records_processed: int
    features_engineered: int
    datasets_available: List[str]


class SystemInfo(BaseModel):
    """System information schema"""
    project_root: str
    models_dir: str
    logs_dir: str
    visualizations_dir: str
    project_status: str


class ApplicationDetails(BaseModel):
    """Complete application details schema"""
    pipeline: PipelineInfo
    models: ModelInfo
    metrics: ModelMetrics
    data_pipeline: DataPipelineStatus
    system: SystemInfo
    timestamp: str


class PredictionRequest(BaseModel):
    """Prediction request schema"""
    features: List[float]
    model_name: Optional[str] = "xgboost"


class PredictionResponse(BaseModel):
    """Prediction response schema"""
    prediction: int  # 0 = No Failure, 1 = Failure
    probability: float
    confidence: str
    model_used: str
    timestamp: str


class HealthResponse(BaseModel):
    """API health check response"""
    status: str
    message: str
    timestamp: str

