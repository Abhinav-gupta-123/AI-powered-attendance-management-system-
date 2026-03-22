"""
routers/enroll.py — Enrollment router.
Handles GET /ui/enroll, POST /enroll/start, SSE /enroll/status/{id},
POST /enroll/{id}/trigger, DELETE /enroll/{id}, GET /enroll/{id}.
"""
import asyncio, json as _json, os, threading, time, traceback, uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from auth_deps import get_current_user
from fastapi import Depends

from Enroll_student import (
    db_connect as enroll_db_connect,
    db_get_or_create_class, db_roll_exists, db_insert_student, db_load_all_students,
    build_indexes_from_mysql, faiss_add, faiss_save, find_match_faiss,
    get_best_face, save_face_images,
    SIMILARITY_THRESHOLD, ANGLES_NEEDED, ANGLE_PROMPTS,
    CAPTURE_INTERVAL, NORM_MIN, IMAGES_DIR, FAISS_DIR,
)
from live_attendance import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

router = APIRouter()

# ── constants ─────────────────────────────────────────────────────────────────
MAX_CONCURRENT_ENROLLMENTS = 2
ENROLL_JOB_TTL_S           = 300
SSE_IDLE_TIMEOUT_S         = 120

# ── shared state (lazy asyncio.Lock to avoid module-import crash) ───────────────
_insight_app = None
_enroll_jobs: Dict[str, Any] = {}
_enroll_jobs_lock: Optional[asyncio.Lock] = None
_main_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_lock() -> asyncio.Lock:
    global _enroll_jobs_lock
    if _enroll_jobs_lock is None:
        _enroll_jobs_lock = asyncio.Lock()
    return _enroll_jobs_lock


def set_model(model) -> None:
    global _insight_app; _insight_app = model


def set_main_loop(loop) -> None:
    global _main_loop; _main_loop = loop


# ── frontend path ──────────────────────────────────────────────────────────────
_FRONTEND = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "frontend")
)

def _serve(fname: str) -> HTMLResponse:
    p = os.path.join(_FRONTEND, fname)
    try:
        return HTMLResponse(open(p, encoding="utf-8").read())
    except FileNotFoundError:
        return HTMLResponse(f"<h2>Missing: {fname}</h2>", status_code=404)


def get_enroll_jobs():      return _enroll_jobs
def get_enroll_jobs_lock(): return _get_lock()


# ── EnrollJob ─────────────────────────────────────────────────────────────────
class EnrollJob:
    STATES = ("waiting","capturing","checking","enrolling","done","duplicate","error","cancelled")

    def __init__(self, job_id, class_name, roll_no, name):
        self.job_id = job_id; self.class_name = class_name
        self.roll_no = roll_no; self.name = name
        self.state = "waiting"; self.angle_done = 0
        self.embs: List[np.ndarray] = []; self.crops: List[np.ndarray] = []
        self.message = "Ready — waiting for face..."; self.quality = 0.0; self.reason = ""
        self.created_at = time.time(); self.finished_at: Optional[float] = None
        self.student_id: Optional[int] = None
        self.sse_queue     = asyncio.Queue(maxsize=50)
        self.frame_q       = asyncio.Queue(maxsize=2)
        self.trigger_event = threading.Event()
        self.cancel_event  = threading.Event()

    def push_frame(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok: return
        jpg = buf.tobytes()
        lp = _main_loop
        if lp is None or not lp.is_running(): return
        async def _put():
            if self.frame_q.full():
                try: self.frame_q.get_nowait()
                except: pass
            try: self.frame_q.put_nowait(jpg)
            except: pass
        asyncio.run_coroutine_threadsafe(_put(), lp)

    def push_event(self, event, data):
        payload = {"event": event, "data": data, "ts": datetime.now().strftime("%H:%M:%S")}
        lp = _main_loop
        if lp is None or not lp.is_running(): return
        async def _put():
            try: self.sse_queue.put_nowait(payload)
            except asyncio.QueueFull:
                try: self.sse_queue.get_nowait()
                except: pass
                try: self.sse_queue.put_nowait(payload)
                except: pass
        asyncio.run_coroutine_threadsafe(_put(), lp)

    def is_terminal(self): return self.state in ("done","duplicate","error","cancelled")
    def elapsed_ttl(self): return time.time() - (self.finished_at or self.created_at)





# ── Worker ────────────────────────────────────────────────────────────────────
def _run_enroll_job_sync(job_id: str, loop) -> None:
    job = _enroll_jobs.get(job_id)
    if job is None: return
    def push(event, **kw): job.push_event(event, kw)
    try:
        import sys
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not cap.isOpened():
            job.state = "error"; job.message = "Cannot open camera"
            push("error", message=job.message); return
        conn = enroll_db_connect()
        try:
            cl_id = db_get_or_create_class(conn, job.class_name)
            if db_roll_exists(conn, cl_id, job.roll_no):
                job.state = "error"; job.message = f"Roll '{job.roll_no}' already exists"
                push("error", message=job.message); return
            all_students  = db_load_all_students(conn)
            id_to_record  = {r["student_id"]: r for r in all_students}
            class_indexes = build_indexes_from_mysql(all_students)
        finally: pass
        job.state = "waiting"; push("state_change", state="waiting", message="Camera open — waiting for quality face")
        last_scan = time.time() - CAPTURE_INTERVAL
        while not job.cancel_event.is_set() and not job.is_terminal():
            ret, frame = cap.read()
            if not ret: time.sleep(0.05); continue
            now = time.time(); countdown = max(0.0, CAPTURE_INTERVAL - (now - last_scan))
            faces = []; model = _insight_app
            if model:
                try: faces = model.get(frame)
                except Exception as e: print(f"[ENROLL:{job_id[:8]}] {e}")
            best_face, best_score, best_reason = get_best_face(faces, frame)
            
            # Draw on frame before streaming
            display = frame.copy()
            if best_face is not None:
                bx1, by1, bx2, by2 = [int(v) for v in best_face.bbox]
                bc = (0, 210, 80) if best_score >= 75.0 else (40, 40, 220)
                cv2.rectangle(display, (bx1, by1), (bx2, by2), bc, 2)
            cv2.putText(display, f"Quality: {best_score:.0f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Angle {job.angle_done+1}/{ANGLES_NEEDED}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            job.push_frame(display)

            job.quality = float(best_score); job.reason = best_reason
            push("quality_update", quality=round(best_score,1), reason=best_reason,
                 countdown=round(countdown,1), angle_done=job.angle_done, angles_needed=ANGLES_NEEDED,
                 prompt=ANGLE_PROMPTS[job.angle_done] if job.angle_done < len(ANGLE_PROMPTS) else f"Angle {job.angle_done+1}")
            if countdown <= 0.0 and best_face is not None and best_score >= 75.0:
                job.trigger_event.clear(); job.state = "capturing"
                push("state_change", state="capturing", message=f"Face ready — send POST /enroll/{job_id}/trigger")
                triggered = job.trigger_event.wait(timeout=30.0)
                if not triggered or job.cancel_event.is_set():
                    last_scan = time.time(); job.state = "waiting"; continue
                ret2, frame2 = cap.read()
                if not ret2: last_scan = time.time(); continue
                faces2 = []
                if model:
                    try: faces2 = model.get(frame2)
                    except: pass
                bf2, bs2, br2 = get_best_face(faces2, frame2)
                
                display = frame2.copy()
                if bf2 is not None:
                    bx1, by1, bx2, by2 = [int(v) for v in bf2.bbox]
                    cv2.rectangle(display, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
                job.push_frame(display)

                if bf2 is None or bs2 < 75.0:
                    push("capture_failed", reason=br2 or f"Quality {bs2:.0f}% too low — retry")
                    last_scan = time.time(); job.state = "waiting"; continue
                emb = bf2.embedding
                if emb is None:
                    push("capture_failed", reason="No embedding — retry"); last_scan = time.time(); job.state = "waiting"; continue
                norm = float(np.linalg.norm(emb))
                if norm < NORM_MIN:
                    push("capture_failed", reason=f"Embedding quality low (norm={norm:.1f})"); last_scan = time.time(); job.state = "waiting"; continue
                job.embs.append(emb.copy())
                h2,w2=frame2.shape[:2]; b2=bf2.bbox
                x1,y1,x2,y2=(max(0,int(b2[0])),max(0,int(b2[1])),min(w2,int(b2[2])),min(h2,int(b2[3])))
                job.crops.append(frame2[y1:y2,x1:x2].copy())
                job.angle_done += 1
                push("angle_captured", angle=job.angle_done, angles_needed=ANGLES_NEEDED, norm=round(norm,2), quality=round(bs2,1))
                last_scan = time.time()
                if job.angle_done >= ANGLES_NEEDED: break
                job.state = "waiting"
                push("state_change", state="waiting",
                     message=f"Angle {job.angle_done}/{ANGLES_NEEDED} done — next: {ANGLE_PROMPTS[job.angle_done] if job.angle_done < len(ANGLE_PROMPTS) else 'Angle '+str(job.angle_done+1)}")
            time.sleep(0.05)
        cap.release()
        if job.cancel_event.is_set():
            job.state = "cancelled"; push("state_change", state="cancelled", message="Enrollment cancelled"); conn.close(); return
        if len(job.embs) < ANGLES_NEEDED:
            job.state = "error"; job.message = "Not enough angles captured"; push("error", message=job.message); conn.close(); return
        job.state = "checking"; push("state_change", state="checking", message="Running FAISS duplicate check …")
        matched, sid, dup_name, avg_sim, _ = find_match_faiss(job.embs, class_indexes, id_to_record)
        if matched:
            job.state = "duplicate"; job.message = f"Already enrolled: {dup_name} (sim={avg_sim:.3f})"
            push("duplicate", name=dup_name, sim=round(avg_sim,4), message=job.message); conn.close(); return
        job.state = "enrolling"; push("state_change", state="enrolling", message="Saving to MySQL + FAISS …")
        cl_id2 = db_get_or_create_class(conn, job.class_name)
        student_id = db_insert_student(conn, cl_id2, job.name, job.roll_no, job.embs, "pending")
        save_face_images(student_id, job.crops)
        photo_path = os.path.join(IMAGES_DIR, str(student_id))
        cur = conn.cursor(); cur.execute("UPDATE students SET photo_path=%s WHERE id=%s", (photo_path, student_id))
        conn.commit(); cur.close(); conn.close()
        if job.class_name not in class_indexes:
            import faiss as _faiss
            base = _faiss.IndexFlatIP(512); class_indexes[job.class_name] = _faiss.IndexIDMap(base)
        faiss_add(class_indexes[job.class_name], job.embs, student_id)
        faiss_save(class_indexes[job.class_name], job.class_name, job.school_name)
        job.state = "done"; job.student_id = student_id; job.finished_at = time.time()
        job.message = f"Enrolled {job.name} → id={student_id}, class={job.class_name}, roll={job.roll_no}"
        push("enrolled", student_id=student_id, name=job.name, class_name=job.class_name, roll_no=job.roll_no, message=job.message)
    except Exception as e:
        job.state = "error"; job.message = str(e)
        job.push_event("error", {"message": str(e), "traceback": traceback.format_exc()})
        print(f"[ENROLL:{job_id[:8]}] Error: {e}\n{traceback.format_exc()}")
    finally:
        job.finished_at = job.finished_at or time.time()


# ── Routes ────────────────────────────────────────────────────────────────────
@router.get("/ui/enroll", tags=["ui"], response_class=HTMLResponse)
async def ui_enroll():
    return _serve("enroll.html")


class StartEnrollRequest(BaseModel):
    class_name: str; roll_no: str; name: str
    school_name: Optional[str] = None  # populated from user JWT


@router.post("/enroll/start", tags=["enroll"])
async def enroll_start(req: StartEnrollRequest, user: Dict = Depends(get_current_user)):
    req.school_name = user.get("school_name", "Default School")
    lock = _get_lock()
    async with lock:
        active = sum(1 for j in _enroll_jobs.values() if not j.is_terminal())
        if active >= MAX_CONCURRENT_ENROLLMENTS:
            raise HTTPException(status_code=429, detail=f"Max concurrent enrollment jobs ({MAX_CONCURRENT_ENROLLMENTS}) reached.")
    job_id = str(uuid.uuid4()); job = EnrollJob(job_id, req.class_name, req.roll_no, req.name)
    async with lock: _enroll_jobs[job_id] = job
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_enroll_job_sync, job_id, loop)
    return {"job_id": job_id, "class_name": req.class_name, "roll_no": req.roll_no, "name": req.name,
            "status_url": f"/enroll/status/{job_id}", "trigger_url": f"/enroll/{job_id}/trigger", "cancel_url": f"/enroll/{job_id}"}


@router.get("/enroll/status/{job_id}", tags=["enroll"])
async def enroll_status_stream(job_id: str, user: Dict = Depends(get_current_user)):
    job = _enroll_jobs.get(job_id)
    if job is None: raise HTTPException(status_code=404, detail="Job not found")
    async def generate():
        idle_since = time.time()
        while True:
            try:
                payload = await asyncio.wait_for(job.sse_queue.get(), timeout=2.0)
                idle_since = time.time()
                yield f"data: {_json.dumps(payload)}\n\n"
                if job.is_terminal():
                    yield f"data: {_json.dumps({'event':'stream_end'})}\n\n"; break
            except asyncio.TimeoutError:
                if time.time() - idle_since > SSE_IDLE_TIMEOUT_S: break
                yield ": keepalive\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@router.get("/enroll/{job_id}/stream", tags=["enroll"])
async def enroll_video_stream(job_id: str, user: Dict = Depends(get_current_user)):
    job = _enroll_jobs.get(job_id)
    if job is None: raise HTTPException(status_code=404, detail="Job not found")
    async def generate():
        idle = time.time()
        while not job.is_terminal() and not job.cancel_event.is_set():
            try:
                jpg = await asyncio.wait_for(job.frame_q.get(), timeout=2.0)
                idle = time.time()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+jpg+b"\r\n"
            except asyncio.TimeoutError:
                if time.time() - idle > SSE_IDLE_TIMEOUT_S: break
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@router.post("/enroll/{job_id}/trigger", tags=["enroll"])
async def enroll_trigger(job_id: str, user: Dict = Depends(get_current_user)):
    job = _enroll_jobs.get(job_id)
    if job is None: raise HTTPException(status_code=404, detail="Job not found")
    if job.is_terminal(): raise HTTPException(status_code=400, detail=f"Job is {job.state}")
    job.trigger_event.set()
    return {"job_id": job_id, "angle_done": job.angle_done, "state": job.state, "message": "Trigger signal sent"}


@router.delete("/enroll/{job_id}", tags=["enroll"])
async def enroll_cancel(job_id: str, user: Dict = Depends(get_current_user)):
    job = _enroll_jobs.get(job_id)
    if job is None: raise HTTPException(status_code=404, detail="Job not found")
    job.cancel_event.set()
    return {"job_id": job_id, "status": "cancellation requested"}


@router.get("/enroll/{job_id}", tags=["enroll"])
async def enroll_job_info(job_id: str, user: Dict = Depends(get_current_user)):
    job = _enroll_jobs.get(job_id)
    if job is None: raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "class_name": job.class_name, "roll_no": job.roll_no, "name": job.name,
            "state": job.state, "angle_done": job.angle_done, "angles_needed": ANGLES_NEEDED,
            "quality": job.quality, "reason": job.reason, "message": job.message, "student_id": job.student_id}
