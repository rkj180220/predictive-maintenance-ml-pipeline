"""
Predictive Maintenance ML Pipeline - Main Entry Point
Author: ramkumarjayakumar
Date: 2025-10-18

This script provides multiple execution modes:
1. One-time pipeline execution
2. Scheduled execution (every 2 minutes)
3. Background daemon mode
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline_main import PredictiveMaintenancePipeline
from src.dataops.pipeline_scheduler import PipelineScheduler
from src.config.settings import DATAOPS_CONFIG


def run_once():
    """Run pipeline once"""
    print("Running pipeline in one-time execution mode...")
    pipeline = PredictiveMaintenancePipeline()
    return pipeline.run_full_pipeline()


def run_scheduled():
    """Run pipeline on schedule"""
    print(f"Running pipeline in scheduled mode (every {DATAOPS_CONFIG['pipeline_schedule_minutes']} minutes)...")
    print("Press Ctrl+C to stop")

    pipeline = PredictiveMaintenancePipeline()
    scheduler = PipelineScheduler(DATAOPS_CONFIG)

    # Schedule the pipeline
    scheduler.schedule_pipeline(pipeline.run_full_pipeline)

    # Run first execution immediately
    print("\nExecuting first run immediately...")
    scheduler.run_once(pipeline.run_full_pipeline)

    # Start scheduler
    scheduler.start()


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Predictive Maintenance ML Pipeline for NASA C-MAPSS Dataset"
    )

    parser.add_argument(
        '--mode',
        choices=['once', 'scheduled'],
        default='once',
        help='Execution mode: once (single run) or scheduled (every 2 minutes)'
    )

    args = parser.parse_args()

    print("="*80)
    print("PREDICTIVE MAINTENANCE ML PIPELINE")
    print("NASA C-MAPSS Turbofan Engine Dataset")
    print("Author: ramkumarjayakumar")
    print("Date: 2025-10-18")
    print("="*80)
    print()

    if args.mode == 'once':
        success = run_once()
        return 0 if success else 1
    elif args.mode == 'scheduled':
        run_scheduled()
        return 0


if __name__ == "__main__":
    sys.exit(main())
