"""
Automated Data Sync Scheduler
==============================
Runs in the background and syncs data periodically.

Usage:
    python sync_scheduler.py              # Run continuous sync
    python sync_scheduler.py --once       # Run once and exit
    python sync_scheduler.py --daemon     # Run as daemon (detached)
"""

import schedule
import time
import logging
from datetime import datetime
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fetchers import SyncOrchestrator

# Configure logging
log_path = Path.home() / ".consciousness_research" / "logs"
log_path.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path / "sync.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SyncScheduler:
    """Manages scheduled data syncing."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path.home() / ".consciousness_research" / "consciousness_research.db")
        
        self.orchestrator = SyncOrchestrator(db_path)
        self.last_sync = None
        self.sync_count = 0
    
    def setup(self) -> bool:
        """Initialize authentication for all sources."""
        logger.info("Setting up data source authentication...")
        results = self.orchestrator.setup()
        
        all_success = all(results.values())
        for source, success in results.items():
            status = "✓" if success else "✗"
            logger.info(f"  {status} {source}")
        
        return all_success
    
    def run_sync(self):
        """Execute a sync operation."""
        logger.info("Starting scheduled sync...")
        
        try:
            results = self.orchestrator.sync_all(days_back=3)  # Last 3 days for regular sync
            
            self.last_sync = datetime.now()
            self.sync_count += 1
            
            logger.info(f"Sync completed: {results}")
            
            # Compute daily context for today and yesterday
            today = datetime.now().strftime("%Y-%m-%d")
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            
            for date in [yesterday, today]:
                try:
                    self.orchestrator.compute_daily_context(date)
                    logger.info(f"Computed daily context for {date}")
                except Exception as e:
                    logger.warning(f"Failed to compute context for {date}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return None
    
    def status(self) -> dict:
        """Return current sync status."""
        return {
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "sync_count": self.sync_count,
            "db_path": str(self.orchestrator.data_store.db_path)
        }


def run_scheduler():
    """Run the continuous sync scheduler."""
    from datetime import timedelta
    
    scheduler = SyncScheduler()
    
    # Setup authentication
    if not scheduler.setup():
        logger.warning("Some data sources failed to authenticate. Continuing with available sources.")
    
    # Initial sync
    logger.info("Running initial sync...")
    scheduler.run_sync()
    
    # Schedule regular syncs
    # - Garmin: every 4 hours (data updates aren't that frequent)
    # - Calendar: every hour (more dynamic)
    schedule.every(4).hours.do(scheduler.run_sync)
    
    logger.info("Scheduler started. Press Ctrl+C to stop.")
    logger.info("Sync schedule: every 4 hours")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped.")


def run_once():
    """Run a single sync and exit."""
    scheduler = SyncScheduler()
    
    if not scheduler.setup():
        logger.warning("Some data sources failed to authenticate.")
    
    results = scheduler.run_sync()
    
    if results:
        print(f"\n✓ Sync complete!")
        for source, count in results.items():
            print(f"  {source}: {count} records")
    else:
        print("\n✗ Sync failed. Check logs for details.")


if __name__ == "__main__":
    import argparse
    from datetime import timedelta
    
    parser = argparse.ArgumentParser(description="Consciousness Research Data Sync")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon")
    
    args = parser.parse_args()
    
    if args.once:
        run_once()
    elif args.daemon:
        # Daemonize (Unix only)
        if os.name != 'nt':
            import daemon
            with daemon.DaemonContext():
                run_scheduler()
        else:
            logger.error("Daemon mode not supported on Windows")
    else:
        run_scheduler()
