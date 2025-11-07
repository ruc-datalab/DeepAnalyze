"""
Main application entry point for DeepAnalyze API Server
Sets up the FastAPI application and starts the server
"""

import time
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import API_HOST, API_PORT, API_TITLE, API_VERSION, HTTP_SERVER_PORT, CLEANUP_INTERVAL_MINUTES
from models import HealthResponse
from utils import start_http_server
from storage import storage


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(title=API_TITLE, version=API_VERSION)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include all routers
    from file_api import router as file_router
    from assistants_api import router as assistants_router
    from threads_api import threads_router, messages_router
    from runs_api import router as runs_router
    from chat_api import router as chat_router
    from admin_api import router as admin_router

    app.include_router(file_router)
    app.include_router(assistants_router)
    app.include_router(threads_router)
    app.include_router(messages_router)
    app.include_router(runs_router)
    app.include_router(chat_router)
    app.include_router(admin_router)

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        return HealthResponse(
            status="healthy",
            timestamp=int(time.time())
        )

    return app


def schedule_thread_cleanup():
    """Background thread to periodically clean up expired threads"""
    import time as time_module

    while True:
        try:
            cleaned_count = storage.cleanup_expired_threads(timeout_hours=12)
            if cleaned_count > 0:
                print(f"Thread cleanup completed: removed {cleaned_count} expired threads")
        except Exception as e:
            print(f"Thread cleanup error: {e}")

        # Sleep for configured interval
        time_module.sleep(CLEANUP_INTERVAL_MINUTES * 60)


def main():
    """Main entry point to start the API server"""
    print("🚀 Starting DeepAnalyze OpenAI-Compatible API Server...")
    print(f"   - API Server: http://{API_HOST}:{API_PORT}")
    print(f"   - File Server: http://localhost:{HTTP_SERVER_PORT}")
    print(f"   - Workspace: workspace")
    print("\n📖 API Endpoints:")
    print("   - Files API: /v1/files")
    print("   - Assistants API: /v1/assistants")
    print("   - Threads API: /v1/threads")
    print("   - Messages API: /v1/threads/{thread_id}/messages")
    print("   - Runs API: /v1/threads/{thread_id}/runs")
    print("   - Chat API: /v1/chat/completions")
    print("   - Admin API: /v1/admin")
    print("   - Extended: /v1/threads/{thread_id}/files")
    print("   - Extended: /v1/chat/completions (with file_ids)")

    # Start HTTP file server in a separate thread
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()

    # Start thread cleanup scheduler in background thread
    cleanup_thread = threading.Thread(target=schedule_thread_cleanup, daemon=True)
    cleanup_thread.start()
    print(f"Thread cleanup scheduler started (runs every {CLEANUP_INTERVAL_MINUTES} minutes, 12-hour timeout)")

    # Create and start the FastAPI application
    app = create_app()
    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()