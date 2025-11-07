"""
Assistants API for DeepAnalyze API Server
Handles assistant creation, management, and configuration endpoints
"""

from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Body

from config import SUPPORTED_TOOLS
from models import AssistantObject
from storage import storage


# Create router for assistants endpoints
router = APIRouter(prefix="/v1/assistants", tags=["assistants"])


@router.post("", response_model=AssistantObject)
async def create_assistant(
    model: str = Body(...),
    name: Optional[str] = Body(None),
    description: Optional[str] = Body(None),
    instructions: Optional[str] = Body(None),
    tools: Optional[List[Dict]] = Body(None),
    file_ids: Optional[List[str]] = Body(None),
    metadata: Optional[Dict] = Body(None),
):
    """Create an assistant (OpenAI compatible)"""
    # Validate tools - only allow supported tools for now
    if tools:
        for tool in tools:
            if tool.get("type") not in SUPPORTED_TOOLS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported tool type: {tool.get('type')}. Only supported tools are: {SUPPORTED_TOOLS}",
                )

    # Validate file_ids
    if file_ids:
        for fid in file_ids:
            if not storage.get_file(fid):
                raise HTTPException(status_code=400, detail=f"File {fid} not found")

    assistant = storage.create_assistant(
        model=model,
        name=name,
        description=description,
        instructions=instructions,
        tools=tools,
        file_ids=file_ids,
        metadata=metadata,
    )
    return assistant


@router.get("", response_model=dict)
async def list_assistants():
    """List assistants (OpenAI compatible)"""
    assistants = storage.list_assistants()
    return {"object": "list", "data": [a.dict() for a in assistants]}


@router.get("/{assistant_id}", response_model=AssistantObject)
async def retrieve_assistant(assistant_id: str):
    """Retrieve an assistant (OpenAI compatible)"""
    assistant = storage.get_assistant(assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return assistant


@router.delete("/{assistant_id}")
async def delete_assistant(assistant_id: str):
    """Delete an assistant (OpenAI compatible)"""
    success = storage.delete_assistant(assistant_id)
    if not success:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return {"id": assistant_id, "object": "assistant.deleted", "deleted": True}