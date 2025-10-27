"""
Routes module initialization
"""

from fastapi import APIRouter
from .application_details import router as app_details_router
from .data_pipeline_routes import router as data_router
from .ml_pipeline_routes import router as ml_router
from .dashboard_routes import router as dashboard_router

# Create main router
router = APIRouter()

# Include all sub-routers
router.include_router(app_details_router)
router.include_router(data_router)
router.include_router(ml_router)
router.include_router(dashboard_router)

__all__ = ["router"]
"""
API module initialization
"""

__version__ = "1.0.0"

