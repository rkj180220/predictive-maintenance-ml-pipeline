"""
API Server Runner - Sub-Objective 3
Author: ramkumarjayakumar
Date: 2025-10-27

Quick runner script to start the Predictive Maintenance ML Pipeline API
Activities 3.1 & 3.2: Retrieve and Display Application Details
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Main entry point for API server"""
    import uvicorn

    print("="*80)
    print("🚀 PREDICTIVE MAINTENANCE ML PIPELINE - API SERVER")
    print("="*80)
    print("\n📊 Sub-Objectives:")
    print("   ✅ Sub-Objective 1: Data Pipeline (Completed)")
    print("   ✅ Sub-Objective 2: ML Pipeline (Completed)")
    print("   ✅ Sub-Objective 3: API Access (Current)")
    print("\n📋 Activities 3.1 & 3.2:")
    print("   ✅ Activity 3.1: Retrieve Key Application Details via APIs")
    print("   ✅ Activity 3.2: Display 4+ Application Details")
    print("\n" + "="*80)

    print("\n🌐 API ENDPOINTS:")
    print("   Base URL: http://127.0.0.1:8000")
    print("\n   📍 Health Check:")
    print("      GET /api/v1/health")
    print("\n   📍 All Application Details (Activity 3.2):")
    print("      GET /api/v1/application-details")
    print("      → Returns all 4+ required details in one response")
    print("\n   📍 Individual Details (Activity 3.1):")
    print("      GET /api/v1/pipeline/info")
    print("      GET /api/v1/models/info")
    print("      GET /api/v1/metrics")
    print("      GET /api/v1/data-pipeline")
    print("      GET /api/v1/system")
    print("\n📖 DOCUMENTATION:")
    print("   Swagger UI: http://127.0.0.1:8000/docs")
    print("   ReDoc: http://127.0.0.1:8000/redoc")
    print("\n" + "="*80)
    print("\n⏳ Starting API Server...")
    print("   Press Ctrl+C to stop the server\n")
    print("="*80 + "\n")

    # Start the API server
    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()

