"""
routers/inspect.py
──────────────────
FAISS index inspector router.
Handles:
  GET  /inspect/classes
  GET  /inspect/{class_name}
  GET  /inspect/{class_name}/html
  GET  /student/photo/{student_id}
  GET  /ui/inspect
"""

import asyncio
import os

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from auth_deps import get_current_user

from inspect_index import load_class_data, build_html, list_available

router = APIRouter()

FRONTEND = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "frontend")
)


def _serve(fname: str) -> HTMLResponse:
    p = os.path.join(FRONTEND, fname)
    try:
        return HTMLResponse(open(p, encoding="utf-8").read())
    except FileNotFoundError:
        return HTMLResponse(f"<h2>Missing: {fname}</h2>", status_code=404)


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/ui/inspect", tags=["ui"], response_class=HTMLResponse)
async def ui_inspect():
    """Rich interactive FAISS inspector — student cards with photos, filters, heatmap."""
    return _serve("inspect.html")




@router.get("/inspect/classes", tags=["inspect"])
async def inspect_list_classes(user: dict = Depends(get_current_user)):
    """List all classes that have a FAISS index file (filtered by school for teachers)."""
    loop = asyncio.get_event_loop()
    try:
        # Teachers/Admins only see classes for THEIR school
        sn = user.get("school_name")
        classes = await loop.run_in_executor(None, list_available, sn)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"classes": classes}


@router.get("/inspect/{class_name:path}", tags=["inspect"])
async def inspect_class(class_name: str, user: dict = Depends(get_current_user)):
    """Return full class inspection data as JSON (enforces school boundary)."""
    sn = user.get("school_name", "")
    
    # Enforce school boundary: prepend school name folder if not present
    if not class_name.startswith(sn + "/"):
        if "/" not in class_name:
            class_name = f"{sn}/{class_name}"
        else:
            # If they try to access another school's folder, 403
            raise HTTPException(status_code=403, detail="Access denied to this school's records")

    loop = asyncio.get_event_loop()
    try:
        data = await asyncio.wait_for(
            loop.run_in_executor(None, load_class_data, class_name),
            timeout=30.0,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Index not found: {class_name}")

    if data and "students" in data:
        for s in data["students"]:
            s.pop("embeddings", None)   # strip raw blobs
    return data


@router.get("/inspect/{class_name:path}/html", tags=["inspect"])
async def inspect_class_html(class_name: str, user: dict = Depends(get_current_user)):
    """Return the HTML inspection report."""
    loop = asyncio.get_event_loop()
    try:
        data = await asyncio.wait_for(
            loop.run_in_executor(None, load_class_data, class_name),
            timeout=30.0,
        )
    except SystemExit:
        raise HTTPException(status_code=404, detail=f"Class '{class_name}' not found")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="inspect timed out")
    html = build_html(data)
    return HTMLResponse(content=html)


@router.get("/student/photo/{student_id}", tags=["ui"])
async def student_photo(student_id: int, user: dict = Depends(get_current_user)):
    """Serve a student's enrollment photo by student ID."""
    loop = asyncio.get_event_loop()

    def _get_path():
        import mysql.connector
        from inspect_index import DB_CONFIG
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cur  = conn.cursor()
            cur.execute("SELECT photo_path FROM students WHERE id = %s", (student_id,))
            row  = cur.fetchone()
            cur.close()
            conn.close()
            return row[0] if row else None
        except Exception:
            return None

    photo_path = await loop.run_in_executor(None, _get_path)
    if not photo_path or photo_path == "—":
        raise HTTPException(status_code=404, detail="No photo")
    if not os.path.isabs(photo_path):
        photo_path = os.path.join(os.getcwd(), photo_path)
    if not os.path.isfile(photo_path):
        raise HTTPException(status_code=404, detail="Photo file not found")
    return FileResponse(photo_path, media_type="image/jpeg")
