from __future__ import annotations

from fastapi import APIRouter, File, Query, UploadFile

from ..services import workspace as workspace_service


router = APIRouter()


@router.get("/workspace/files")
async def get_workspace_files(session_id: str = Query("default")):
    return {"files": workspace_service.list_workspace_files(session_id)}


@router.get("/workspace/tree")
async def workspace_tree(session_id: str = Query("default")):
    workspace_root = workspace_service.resolve_workspace_root(session_id)
    return workspace_service.build_tree(workspace_root, workspace_root, session_id)


@router.get("/workspace/download-bundle")
async def download_workspace_bundle(
    category: str = Query("all", description="all/table/image/other"),
    session_id: str = Query("default"),
):
    return workspace_service.download_generated_bundle(session_id, category)


@router.get("/workspace/preview")
async def preview_workspace_file(
    path: str = Query(..., description="relative path under workspace"),
    session_id: str = Query("default"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    table_name: str = Query("", description="sqlite table name"),
    sheet_name: str = Query("", description="excel sheet name"),
):
    return workspace_service.preview_workspace_file(
        session_id,
        path,
        page=page,
        page_size=page_size,
        table_name=table_name,
        sheet_name=sheet_name,
    )


@router.delete("/workspace/file")
async def delete_workspace_file(
    path: str = Query(..., description="relative path under workspace"),
    session_id: str = Query("default"),
):
    return workspace_service.delete_workspace_file(session_id, path)


@router.post("/workspace/move")
async def move_path(
    src: str = Query(..., description="relative source path under workspace"),
    dst_dir: str = Query("", description="relative target directory under workspace"),
    session_id: str = Query("default"),
):
    return workspace_service.move_workspace_path(session_id, src, dst_dir)


@router.delete("/workspace/dir")
async def delete_workspace_dir(
    path: str = Query(..., description="relative directory under workspace"),
    recursive: bool = Query(True, description="delete directory recursively"),
    session_id: str = Query("default"),
):
    return workspace_service.delete_workspace_dir(session_id, path, recursive)


@router.get("/proxy")
async def proxy(url: str):
    return await workspace_service.proxy_external_file(url)


@router.post("/workspace/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    session_id: str = Query("default"),
):
    return await workspace_service.upload_files_to_workspace(session_id, files)


@router.delete("/workspace/clear")
async def clear_workspace(session_id: str = Query("default")):
    return workspace_service.clear_workspace(session_id)


@router.post("/workspace/upload-to")
async def upload_to_dir(
    dir: str = Query("", description="relative directory under workspace"),
    files: list[UploadFile] = File(...),
    session_id: str = Query("default"),
):
    return await workspace_service.upload_files_to_dir(session_id, dir, files)
