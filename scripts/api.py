"""
FastAPI Wrapper — AI Attendance Management System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Thin entry point — all route logic lives in scripts/routers/:

  routers/attendance.py  — live attendance session + dashboard UI
  routers/enroll.py      — student enrollment + enroll UI
  routers/inspect.py     — FAISS index inspector UI + /inspect/* API

Run:
  uvicorn scripts.api:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import os
import sys
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Make sure scripts/ is on sys.path when run as a module ───────────────────
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _p in (_SCRIPTS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("[API] pip install insightface onnxruntime")
    sys.exit(1)

try:
    from live_attendance import MODEL_NAME
    from Enroll_student import IMAGES_DIR, FAISS_DIR
except ImportError as e:
    print(f"[API] Cannot import core scripts: {e}")
    sys.exit(1)

# ── Import routers ────────────────────────────────────────────────────────────
from routers import attendance as _att
from routers import enroll    as _enr
from routers import inspect   as _ins
from routers import auth      as _auth

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_LOAD_TIMEOUT_S = 60

# ── Shared model ──────────────────────────────────────────────────────────────
_insight_app: Optional[FaceAnalysis] = None
_insight_lock = asyncio.Lock()


async def _get_model() -> FaceAnalysis:
    """Return the shared InsightFace model, loading on first call (thread-safe)."""
    global _insight_app
    async with _insight_lock:
        if _insight_app is None:
            loop = asyncio.get_event_loop()

            def _load():
                fa = FaceAnalysis(
                    name=MODEL_NAME,
                    root=_PROJECT_ROOT,
                    providers=["CPUExecutionProvider"],
                )
                fa.prepare(ctx_id=0, det_size=(640, 640))
                warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
                return fa

            print(f"[API] Loading InsightFace {MODEL_NAME} ...")
            _insight_app = await asyncio.wait_for(
                loop.run_in_executor(None, _load),
                timeout=MODEL_LOAD_TIMEOUT_S,
            )
            print("[API] Model ready.")
            # Share model with both routers that need it
            _att.set_model(_insight_app)
            _enr.set_model(_insight_app)
    return _insight_app


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _insight_app

    # ── Startup ──────────────────────────────────────────────────────────────
    loop = asyncio.get_event_loop()
    _enr.set_main_loop(loop)
    print("[API] Starting up …")
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(FAISS_DIR,  exist_ok=True)

    try:
        await _get_model()
    except asyncio.TimeoutError:
        print(f"[API] Model load timed out after {MODEL_LOAD_TIMEOUT_S}s — continuing")
    except Exception as e:
        print(f"[API] Model load failed: {e} — endpoints will retry on first request")

    yield  # ← app runs here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    print("[API] Shutting down …")
    sessions      = _att.get_sessions()
    sessions_lock = _att.get_sessions_lock()
    async with sessions_lock:
        for sess in list(sessions.values()):
            try:
                sess.stop()
                sess.write_queue.put(None)
            except Exception:
                pass

    enroll_jobs      = _enr.get_enroll_jobs()
    enroll_jobs_lock = _enr.get_enroll_jobs_lock()
    async with enroll_jobs_lock:
        for job in list(enroll_jobs.values()):
            job.cancel_event.set()

    print("[API] Shutdown complete.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Attendance Management System API",
    version="1.0.0",
    description=(
        "FastAPI wrapper for live_attendance, inspect_index, and Enroll_student. "
        "All core algorithms remain in the original scripts."
    ),
    lifespan=lifespan,
)

# Mount all routers
app.include_router(_auth.router)
app.include_router(_att.router)
app.include_router(_enr.router)
app.include_router(_ins.router)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health():
    sessions     = _att.get_sessions()
    enroll_jobs  = _enr.get_enroll_jobs()
    return {
        "status":       "ok",
        "model_loaded": _insight_app is not None,
        "sessions":     len(sessions),
        "enroll_jobs":  len(enroll_jobs),
        "timestamp":    datetime.utcnow().isoformat(),
    }
