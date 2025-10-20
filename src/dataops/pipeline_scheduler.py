"""
Pipeline Scheduler for Predictive Maintenance
Author: ramkumarjayakumar
Date: 2025-10-18
"""

import schedule
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional
import threading
import signal
import sys

logger = logging.getLogger(__name__)


class PipelineScheduler:
    """
    Schedule and execute predictive maintenance pipeline
    Supports periodic execution with configurable intervals
    """

    def __init__(self, config: Dict):
        """
        Initialize pipeline scheduler

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.schedule_minutes = config.get('pipeline_schedule_minutes', 2)
        self.is_running = False
        self.execution_count = 0
        self.last_execution_time = None
        self.last_execution_status = None
        self.scheduler_thread = None

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def schedule_pipeline(self, pipeline_func: Callable, *args, **kwargs):
        """
        Schedule pipeline to run at specified intervals

        Args:
            pipeline_func: Function to execute
            *args: Arguments for pipeline function
            **kwargs: Keyword arguments for pipeline function
        """
        logger.info(f"Scheduling pipeline to run every {self.schedule_minutes} minute(s)")

        # Wrapper to track execution
        def wrapped_pipeline():
            try:
                logger.info(f"=" * 80)
                logger.info(f"SCHEDULED EXECUTION #{self.execution_count + 1}")
                logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"=" * 80)

                start_time = time.time()
                self.last_execution_time = datetime.now()

                # Execute pipeline
                result = pipeline_func(*args, **kwargs)

                duration = time.time() - start_time
                self.execution_count += 1
                self.last_execution_status = "SUCCESS"

                logger.info(f"Execution #{self.execution_count} completed successfully in {duration:.2f}s")

                return result

            except Exception as e:
                self.last_execution_status = "FAILED"
                logger.error(f"Execution #{self.execution_count + 1} failed: {e}", exc_info=True)
                raise

        # Schedule the pipeline
        schedule.every(self.schedule_minutes).minutes.do(wrapped_pipeline)

        logger.info("Pipeline scheduled successfully")

    def run_once(self, pipeline_func: Callable, *args, **kwargs):
        """
        Run pipeline once immediately

        Args:
            pipeline_func: Function to execute
            *args: Arguments for pipeline function
            **kwargs: Keyword arguments for pipeline function
        """
        logger.info("Running pipeline once (immediate execution)")

        try:
            start_time = time.time()
            self.last_execution_time = datetime.now()

            result = pipeline_func(*args, **kwargs)

            duration = time.time() - start_time
            self.execution_count += 1
            self.last_execution_status = "SUCCESS"

            logger.info(f"One-time execution completed successfully in {duration:.2f}s")

            return result

        except Exception as e:
            self.last_execution_status = "FAILED"
            logger.error(f"One-time execution failed: {e}", exc_info=True)
            raise

    def start(self):
        """Start the scheduler in the main thread"""
        logger.info("Starting scheduler...")
        self.is_running = True

        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Scheduler interrupted by user")
        finally:
            self.stop()

    def start_background(self):
        """Start the scheduler in a background thread"""
        logger.info("Starting scheduler in background thread...")

        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()

        logger.info("Scheduler running in background")

    def _run_scheduler(self):
        """Internal method to run scheduler loop"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)

    def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping scheduler...")
        self.is_running = False

        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)

        schedule.clear()
        logger.info("Scheduler stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)

    def get_status(self) -> Dict:
        """
        Get scheduler status

        Returns:
            Dictionary with scheduler status
        """
        return {
            'is_running': self.is_running,
            'execution_count': self.execution_count,
            'last_execution_time': self.last_execution_time.isoformat() if self.last_execution_time else None,
            'last_execution_status': self.last_execution_status,
            'schedule_interval_minutes': self.schedule_minutes,
            'next_run': schedule.next_run().isoformat() if schedule.next_run() else None
        }

    def get_execution_stats(self) -> Dict:
        """
        Get execution statistics

        Returns:
            Dictionary with execution statistics
        """
        return {
            'total_executions': self.execution_count,
            'last_status': self.last_execution_status,
            'last_execution': self.last_execution_time.isoformat() if self.last_execution_time else None,
            'uptime_minutes': (datetime.now() - self.last_execution_time).total_seconds() / 60
                             if self.last_execution_time else 0
        }


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from src.config.settings import DATAOPS_CONFIG

    # Test scheduler
    def test_pipeline():
        print(f"Pipeline executed at {datetime.now()}")
        return "SUCCESS"

    scheduler = PipelineScheduler(DATAOPS_CONFIG)

    # Run once for testing
    scheduler.run_once(test_pipeline)

    print("\nScheduler Status:")
    print(scheduler.get_status())

