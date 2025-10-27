
"""
ML Pipeline Runner - Sub-Objective 2
Author: ramkumarjayakumar
Date: 2025-10-26

Quick runner script for the complete ML pipeline
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.ml_pipeline.pipeline_orchestrator import MLPipelineOrchestrator


if __name__ == "__main__":
    print("="*80)
    print("MACHINE LEARNING PIPELINE - SUB-OBJECTIVE 2")
    print("Predictive Maintenance for NASA C-MAPSS Dataset")
    print("="*80)

    # Initialize and run pipeline
    orchestrator = MLPipelineOrchestrator(project_root)
    results = orchestrator.run_complete_pipeline(enable_cloudwatch=False)

    if results['status'] == 'SUCCESS':
        print("\n" + "="*80)
        print("✅ SUCCESS! ML Pipeline completed successfully")
        print("="*80)
        print(f"\n📊 Best Model: {results['model_evaluation']['best_model']}")
        print(f"📁 Models saved to: {results['output_paths']['models_dir']}")
        print(f"📁 Logs saved to: {results['output_paths']['logs_dir']}")
        print(f"📁 Visualizations saved to: {results['output_paths']['visualizations_dir']}")
        print("\n" + "="*80)
    else:
        print("\n❌ FAILED! Pipeline encountered an error")
        print(f"Error: {results.get('error', 'Unknown error')}")
        sys.exit(1)

