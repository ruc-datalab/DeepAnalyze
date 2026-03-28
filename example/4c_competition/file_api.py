"""
File Management API for DeepAnalyze API Server
Handles file upload, download, and management endpoints
"""

import os
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query, Response
from fastapi.responses import JSONResponse

from config import VALID_FILE_PURPOSES, FILE_STORAGE_DIR
from models import FileObject, FileDeleteResponse
from storage import storage


# Create router for file endpoints
router = APIRouter(prefix="/v1/files", tags=["files"])


@router.post("", response_model=FileObject)
async def create_file(
    file: UploadFile = File(...),
    purpose: str = Form("file-extract")
):
    """Upload a file (OpenAI compatible)"""
    # Validate purpose
    if purpose not in VALID_FILE_PURPOSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid purpose. Must be one of {VALID_FILE_PURPOSES}"
        )

    # Save file to a persistent location
    os.makedirs(FILE_STORAGE_DIR, exist_ok=True)
    file_id = f"file-{file.filename.replace('.', '-').replace('_', '-')[:8]}-{os.urandom(4).hex()}"
    file_path = os.path.join(FILE_STORAGE_DIR, file_id)

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_obj = storage.create_file(file.filename, file_path, purpose)
        return file_obj
    except Exception as e:
        # Clean up file if creation failed
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=dict)
async def list_files(purpose: Optional[str] = Query(None)):
    """List files (OpenAI compatible)"""
    files = storage.list_files(purpose=purpose)
    return {"object": "list", "data": [f.dict() for f in files]}


@router.get("/{file_id}", response_model=FileObject)
async def retrieve_file(file_id: str):
    """Retrieve file metadata (OpenAI compatible)"""
    file_obj = storage.get_file(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail="File not found")
    return file_obj


@router.delete("/{file_id}", response_model=FileDeleteResponse)
async def delete_file(file_id: str):
    """Delete a file (OpenAI compatible)"""
    success = storage.delete_file(file_id)
    if not success:
        raise HTTPException(status_code=404, detail="File not found")
    return FileDeleteResponse(id=file_id, object="file", deleted=True)


@router.get("/{file_id}/content")
async def download_file(file_id: str):
    """Download file content (OpenAI compatible)"""
    file_obj = storage.get_file(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail="File not found")

    filepath = storage.files[file_id].get("filepath")
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File content not found")

    with open(filepath, "rb") as f:
        content = f.read()

    return Response(content=content, media_type="application/octet-stream")