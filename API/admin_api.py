"""
Admin API for DeepAnalyze API Server
Handles administrative endpoints like thread cleanup and statistics
"""

import time
from fastapi import APIRouter, Query

from config import CLEANUP_TIMEOUT_HOURS
from models import ThreadCleanupRequest, ThreadCleanupResponse, ThreadStatsResponse
from storage import storage


# Create router for admin endpoints
router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.post("/cleanup-threads", response_model=ThreadCleanupResponse)
async def manual_cleanup_threads(
    timeout_hours: int = Query(CLEANUP_TIMEOUT_HOURS, description="Timeout in hours for thread cleanup")
):
    """
    Manual trigger for thread cleanup (Admin API)
    Clean up threads that haven't been accessed for more than timeout_hours
    """
    try:
        cleaned_count = storage.cleanup_expired_threads(timeout_hours=timeout_hours)
        return ThreadCleanupResponse(
            status="success",
            cleaned_threads=cleaned_count,
            timeout_hours=timeout_hours,
            timestamp=int(time.time())
        )
    except Exception as e:
        return ThreadCleanupResponse(
            status="error",
            cleaned_threads=0,
            timeout_hours=timeout_hours,
            timestamp=int(time.time())
        )


@router.get("/threads-stats", response_model=ThreadStatsResponse)
async def get_threads_stats():
    """
    Get statistics about threads (Admin API)
    """
    with storage._lock:
        total_threads = len(storage.threads)
        now = int(time.time())

        # Count threads by age categories
        recent_threads = 0  # < 1 hour
        old_threads = 0     # 1-12 hours
        expired_threads = 0 # > 12 hours

        for thread_data in storage.threads.values():
            last_accessed = thread_data.get("last_accessed_at", thread_data.get("created_at", 0))
            age_hours = (now - last_accessed) / 3600

            if age_hours < 1:
                recent_threads += 1
            elif age_hours <= CLEANUP_TIMEOUT_HOURS:
                old_threads += 1
            else:
                expired_threads += 1

    return ThreadStatsResponse(
        total_threads=total_threads,
        recent_threads=recent_threads,  # < 1 hour
        old_threads=old_threads,        # 1-12 hours
        expired_threads=expired_threads, # > 12 hours
        timeout_hours=CLEANUP_TIMEOUT_HOURS,
        timestamp=int(time.time())
    )