"""
Threads and Messages API for DeepAnalyze API Server
Handles thread creation, management, and message endpoints
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal

from fastapi import APIRouter, HTTPException, Body

from models import ThreadObject, MessageObject
from storage import storage
from utils import get_thread_workspace, uniquify_path


# Create router for threads endpoints
threads_router = APIRouter(prefix="/v1/threads", tags=["threads"])
messages_router = APIRouter(prefix="/v1/threads", tags=["messages"])


@threads_router.post("", response_model=ThreadObject)
async def create_thread(
    metadata: Optional[Dict] = Body(None),
    file_ids: Optional[List[str]] = Body(None),
):
    """Create a thread (OpenAI compatible)"""
    # Validate file_ids
    if file_ids:
        for fid in file_ids:
            if not storage.get_file(fid):
                raise HTTPException(status_code=400, detail=f"File {fid} not found")

    thread = storage.create_thread(metadata=metadata or {}, file_ids=file_ids)
    return thread


@threads_router.get("/{thread_id}", response_model=ThreadObject)
async def retrieve_thread(thread_id: str):
    """Retrieve a thread (OpenAI compatible)"""
    thread = storage.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@threads_router.delete("/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a thread (OpenAI compatible)"""
    success = storage.delete_thread(thread_id)
    if not success:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"id": thread_id, "object": "thread.deleted", "deleted": True}


@messages_router.post("/{thread_id}/messages", response_model=MessageObject)
async def create_message(
    thread_id: str,
    role: Literal["user", "assistant"] = Body(...),
    content: str = Body(...),
    file_ids: Optional[List[str]] = Body(None),
    metadata: Optional[Dict] = Body(None),
):
    """Create a message (OpenAI compatible)"""
    thread = storage.get_thread(thread_id)  # This will update last_accessed_at
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Validate file_ids
    if file_ids:
        for fid in file_ids:
            if not storage.get_file(fid):
                raise HTTPException(status_code=400, detail=f"File {fid} not found")

    # Copy files to thread workspace
    if file_ids:
        workspace_dir = get_thread_workspace(thread_id)
        for fid in file_ids:
            file_obj = storage.get_file(fid)
            if file_obj:
                src_path = storage.files[fid].get("filepath")
                if src_path and os.path.exists(src_path):
                    dst_path = uniquify_path(Path(workspace_dir) / file_obj.filename)
                    shutil.copy2(src_path, dst_path)

    message = storage.create_message(
        thread_id=thread_id,
        role=role,
        content=content,
        file_ids=file_ids,
        metadata=metadata,
    )
    return message


@messages_router.get("/{thread_id}/messages", response_model=dict)
async def list_messages(thread_id: str):
    """List messages in a thread (OpenAI compatible)"""
    thread = storage.get_thread(thread_id)  # This will update last_accessed_at
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    messages = storage.list_messages(thread_id)
    return {"object": "list", "data": [m.dict() for m in reversed(messages)]}


@messages_router.get("/{thread_id}/files")
async def get_thread_files(thread_id: str):
    """
    Extended API: Get generated files for a thread.
    Returns list of files with download URLs.
    """
    thread = storage.get_thread(thread_id)  # This will update last_accessed_at
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    workspace_dir = get_thread_workspace(thread_id)
    generated_dir = Path(workspace_dir) / "generated"

    files = []
    if generated_dir.exists():
        for file_path in generated_dir.iterdir():
            if file_path.is_file():
                rel_path = f"generated/{file_path.name}"
                files.append(
                    {
                        "name": file_path.name,
                        "size": file_path.stat().st_size,
                        "url": f"http://localhost:8100/{thread_id}/{rel_path}",
                        "created_at": int(file_path.stat().st_mtime),
                    }
                )

    return {"object": "list", "data": files}