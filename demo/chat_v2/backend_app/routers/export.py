from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse

from ..services.exporter import export_report_from_body


router = APIRouter()


@router.post("/export/report")
async def export_report(body: dict = Body(...)):
    try:
        return JSONResponse(export_report_from_body(body))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
