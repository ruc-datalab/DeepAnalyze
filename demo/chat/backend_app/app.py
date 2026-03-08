from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.chat import router as chat_router
from .routers.export import router as export_router
from .routers.workspace import router as workspace_router
from .services.workspace import ensure_http_server_started


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_http_server_started()
    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(workspace_router)
    app.include_router(chat_router)
    app.include_router(export_router)
    return app


app = create_app()
