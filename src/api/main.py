"""
FastAPI Application for Sub-Objective 3: API Access
Author: ramkumarjayakumar
Date: 2025-10-27

Main FastAPI server for the Predictive Maintenance ML Pipeline
Activities 3.1 & 3.2: Retrieve and Display Application Details via APIs
"""

import logging
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.routes import router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Predictive Maintenance ML Pipeline API",
    description="""
    Complete API for the Predictive Maintenance System built with NASA C-MAPSS Dataset
    
    ## Sub-Objectives Completed:
    - ✅ Sub-Objective 1: Data Pipeline (Data Ingestion → EDA → DataOps)
    - ✅ Sub-Objective 2: ML Pipeline (Model Training → Evaluation → MLOps)
    - ✅ Sub-Objective 3: API Access (Retrieve & Display Application Details)
    
    ## Activities 3.1 & 3.2:
    - Activity 3.1: Retrieve Key Application Details via APIs
    - Activity 3.2: Display 4+ Application Details
    
    ## Key Features:
    - Retrieve pipeline flow information
    - Access model metadata and performance metrics
    - Get data pipeline status
    - System deployment information
    - Interactive API documentation (Swagger/OpenAPI)
    - Complete Frontend Dashboard
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

# Mount static files
static_dir = Path(__file__).parent / 'static'
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Serve dashboard
@app.get("/dashboard")
async def serve_dashboard():
    """Serve the main dashboard HTML"""
    dashboard_path = Path(__file__).parent / 'templates' / 'dashboard.html'
    if dashboard_path.exists():
        return FileResponse(str(dashboard_path), media_type="text/html")
    return {"error": "Dashboard not found"}


@app.get("/")
async def root():
    """
    Root endpoint - redirects to API documentation and dashboard
    """
    return {
        "message": "🚀 Predictive Maintenance ML Pipeline API",
        "version": "1.0.0",
        "status": "Running",
        "dashboard": "http://127.0.0.1:8000/dashboard",
        "documentation": "http://127.0.0.1:8000/docs",
        "redoc": "http://127.0.0.1:8000/redoc",
        "api_endpoints": {
            "health_check": "GET /api/v1/health",
            "application_details": {
                "all_details": "GET /api/v1/application-details",
                "pipeline_info": "GET /api/v1/pipeline/info",
                "models_info": "GET /api/v1/models/info",
                "metrics": "GET /api/v1/metrics",
                "data_pipeline": "GET /api/v1/data-pipeline",
                "system": "GET /api/v1/system"
            },
            "data_pipeline_endpoints": {
                "correlation_analysis": "GET /api/v1/data/correlation-analysis",
                "statistical_analysis": "GET /api/v1/data/statistical-analysis",
                "validation_report": "GET /api/v1/data/validation-report",
                "drift_analysis_latest": "GET /api/v1/data/drift-analysis",
                "drift_history": "GET /api/v1/data/drift-history",
                "pipeline_metrics": "GET /api/v1/data/pipeline-metrics",
                "quality_summary": "GET /api/v1/data/quality-summary",
                "all_data_details": "GET /api/v1/data/all-details"
            },
            "ml_pipeline_endpoints": {
                "evaluation_results": "GET /api/v1/ml/evaluation-results",
                "feature_importance": "GET /api/v1/ml/feature-importance",
                "mlops_metrics": "GET /api/v1/ml/mlops-metrics?model=<optional>",
                "mlops_all_metrics": "GET /api/v1/ml/mlops-all-metrics",
                "dashboard_data": "GET /api/v1/ml/dashboard-data",
                "model_comparison": "GET /api/v1/ml/model-comparison",
                "pipeline_logs": "GET /api/v1/ml/pipeline-logs?limit=10",
                "model_health": "GET /api/v1/ml/model-health",
                "all_ml_details": "GET /api/v1/ml/all-details"
            },
            "dashboard_endpoints": {
                "dashboard_overview": "GET /api/v1/dashboard/overview",
                "trigger_data_pipeline": "POST /api/v1/dashboard/trigger/data-pipeline",
                "trigger_ml_pipeline": "POST /api/v1/dashboard/trigger/ml-pipeline",
                "system_stats": "GET /api/v1/dashboard/system-stats"
            }
        },
        "quick_start": {
            "view_dashboard": "Visit http://127.0.0.1:8000/dashboard",
            "api_docs": "Visit http://127.0.0.1:8000/docs",
            "redoc_docs": "Visit http://127.0.0.1:8000/redoc"
        },
        "note": "Use the dashboard for interactive UI or visit /docs for API testing"
    }


@app.on_event("startup")
async def startup_event():
    """Startup event - log initialization"""
    logger.info("="*80)
    logger.info("🚀 PREDICTIVE MAINTENANCE ML PIPELINE API STARTING")
    logger.info("="*80)
    logger.info(f"Project Root: {project_root}")
    logger.info(f"API Version: 1.0.0")
    logger.info(f"Objectives: Sub-Obj 1 ✓ | Sub-Obj 2 ✓ | Sub-Obj 3 ✓")
    logger.info(f"Dashboard: http://127.0.0.1:8000/dashboard")
    logger.info("="*80)


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - log cleanup"""
    logger.info("🛑 API Server Shutdown")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server...")
    logger.info("📍 API will be available at http://127.0.0.1:8000")
    logger.info("📊 Dashboard at http://127.0.0.1:8000/dashboard")
    logger.info("📖 Documentation at http://127.0.0.1:8000/docs")
    logger.info("📚 ReDoc at http://127.0.0.1:8000/redoc")

    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )


@app.get("/")
async def root():
    """
    Root endpoint - redirects to API documentation
    """
    return {
        "message": "🚀 Predictive Maintenance ML Pipeline API",
        "version": "1.0.0",
        "status": "Running",
        "documentation": "http://127.0.0.1:8000/docs",
        "redoc": "http://127.0.0.1:8000/redoc",
        "api_endpoints": {
            "health_check": "GET /api/v1/health",
            "application_details": {
                "all_details": "GET /api/v1/application-details",
                "pipeline_info": "GET /api/v1/pipeline/info",
                "models_info": "GET /api/v1/models/info",
                "metrics": "GET /api/v1/metrics",
                "data_pipeline": "GET /api/v1/data-pipeline",
                "system": "GET /api/v1/system"
            },
            "data_pipeline_endpoints": {
                "correlation_analysis": "GET /api/v1/data/correlation-analysis",
                "statistical_analysis": "GET /api/v1/data/statistical-analysis",
                "validation_report": "GET /api/v1/data/validation-report",
                "drift_analysis_latest": "GET /api/v1/data/drift-analysis",
                "drift_history": "GET /api/v1/data/drift-history",
                "pipeline_metrics": "GET /api/v1/data/pipeline-metrics",
                "quality_summary": "GET /api/v1/data/quality-summary",
                "all_data_details": "GET /api/v1/data/all-details"
            },
            "ml_pipeline_endpoints": {
                "evaluation_results": "GET /api/v1/ml/evaluation-results",
                "feature_importance": "GET /api/v1/ml/feature-importance",
                "mlops_metrics": "GET /api/v1/ml/mlops-metrics?model=<optional>",
                "mlops_all_metrics": "GET /api/v1/ml/mlops-all-metrics",
                "dashboard_data": "GET /api/v1/ml/dashboard-data",
                "model_comparison": "GET /api/v1/ml/model-comparison",
                "pipeline_logs": "GET /api/v1/ml/pipeline-logs?limit=10",
                "model_health": "GET /api/v1/ml/model-health",
                "all_ml_details": "GET /api/v1/ml/all-details"
            }
        },
        "note": "Visit /docs or /redoc for interactive API documentation"
    }


@app.on_event("startup")
async def startup_event():
    """Startup event - log initialization"""
    logger.info("="*80)
    logger.info("🚀 PREDICTIVE MAINTENANCE ML PIPELINE API STARTING")
    logger.info("="*80)
    logger.info(f"Project Root: {project_root}")
    logger.info(f"API Version: 1.0.0")
    logger.info(f"Objectives: Sub-Obj 1 ✓ | Sub-Obj 2 ✓ | Sub-Obj 3 ✓")
    logger.info("="*80)


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - log cleanup"""
    logger.info("🛑 API Server Shutdown")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server...")
    logger.info("📍 API will be available at http://127.0.0.1:8000")
    logger.info("📖 Documentation at http://127.0.0.1:8000/docs")
    logger.info("📚 ReDoc at http://127.0.0.1:8000/redoc")

    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )

