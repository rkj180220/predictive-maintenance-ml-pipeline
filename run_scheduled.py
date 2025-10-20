"""
Scheduled Pipeline Execution Script
Author: ramkumarjayakumar
Date: 2025-10-18

This script runs the pipeline every 2 minutes for demonstration purposes
"""

import sys
import time
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline_main import PredictiveMaintenancePipeline
from src.dataops.pipeline_scheduler import PipelineScheduler
from src.config.settings import DATAOPS_CONFIG
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n" + "="*80)
    print("🛑 Scheduler stopped by user")
    print("="*80)
    sys.exit(0)


def main():
    """Main execution"""
    print("="*80)
    print("📅 SCHEDULED PIPELINE EXECUTION")
    print("NASA C-MAPSS Predictive Maintenance Pipeline")
    print("="*80)
    print(f"\n⏰ Schedule: Every {DATAOPS_CONFIG['pipeline_schedule_minutes']} minutes")
    print("Press Ctrl+C to stop\n")
    print("="*80)

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize pipeline and scheduler
    pipeline = PredictiveMaintenancePipeline()
    scheduler = PipelineScheduler(DATAOPS_CONFIG)

    # Schedule the pipeline
    scheduler.schedule_pipeline(pipeline.run_full_pipeline)

    # Run first execution immediately
    print("\n🚀 Running first execution immediately...")
    print("-"*80)
    scheduler.run_once(pipeline.run_full_pipeline)

    # Start scheduler
    print("\n✅ Scheduler started successfully!")
    print(f"Next execution in {DATAOPS_CONFIG['pipeline_schedule_minutes']} minutes...")
    print("-"*80 + "\n")

    scheduler.start()


if __name__ == "__main__":
    main()

