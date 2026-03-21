"""
FastAPI Wrapper — AI Attendance Management System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Exposes three features as HTTP endpoints WITHOUT modifying any existing
script logic.  All algorithms, thresholds, and DB helpers are imported
directly from live_attendance.py, inspect_index.py, and Enroll_student.py.

Endpoints
─────────
  GET  /health

  GET  /inspect/classes
  GET  /inspect/{class_name}           → JSON data
  GET  /inspect/{class_name}/html      → HTML report (same as opening browser)

  POST /attendance/session/start       → {session_id}
  GET  /attendance/session/{id}/stream → MJPEG live feed (annotated HUD)
  GET  /attendance/session/{id}/status → JSON snapshot
  POST /attendance/session/{id}/stop   → save + close session
  DELETE /attendance/session/{id}      → quit without saving

  POST /enroll/start                   → {job_id}
  GET  /enroll/status/{job_id}         → SSE quality+state events
  POST /enroll/{job_id}/trigger        → trigger angle capture (replaces SPACE key)
  DELETE /enroll/{job_id}              → cancel job

Run:
  cd <project-root>
  uvicorn scripts.api:app --host 0.0.0.0 --port 8000 --reload
"""

# ─────────────────────────────────────────────────────────────────────────────
#  std-lib + FastAPI
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import os
import sys
import uuid
import time
import threading
import queue as _queue
import warnings
import traceback
import json as _json
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Make sure scripts/ is on sys.path when api.py is run as a module ─────────
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _p in (_SCRIPTS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, Response
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
#  Import existing script logic — zero modifications
# ─────────────────────────────────────────────────────────────────────────────

try:
    # live_attendance.py
    from live_attendance import (
        db_load_class_students,
        db_list_classes,
        db_get_todays_session,
        db_get_already_present,
        db_create_session,
        db_close_session,
        db_mark_absent_bulk,
        db_writer_thread,
        build_class_index,
        faiss_batch_match,
        compute_adaptive_thresholds,
        live_quality,
        quality_weighted_avg,
        l2_norm,
        get_bbox,
        has_motion,
        draw_hud,
        beep_present,
        CameraThread,
        blob_to_emb,
        # constants
        CAMERA_INDEX,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        MODEL_NAME,
        EMBEDDING_DIM,
        DET_SCORE_MIN as LA_DET_SCORE_MIN,
        BLUR_MIN_LIVE,
        BASE_SIMILARITY_THRESHOLD,
        SMOOTH_WINDOW,
        QUALITY_WEIGHT,
        FRAME_SKIP,
        MOTION_THRESHOLD,
        CONFIRM_FRAMES,
        DEDUP_SECONDS,
        WRITE_QUEUE_MAX,
        SMOOTH_BUF_MAX,
        PRESENT_LOG_MAX,
    )
except ImportError as _e:
    print(f"[API] Cannot import live_attendance: {_e}")
    sys.exit(1)

try:
    # inspect_index.py
    from inspect_index import load_class_data, build_html, list_available
except ImportError as _e:
    print(f"[API] Cannot import inspect_index: {_e}")
    sys.exit(1)

try:
    # Enroll_student.py
    from Enroll_student import (
        db_connect as enroll_db_connect,
        db_get_or_create_class,
        db_roll_exists,
        db_insert_student,
        db_load_all_students,
        build_indexes_from_mysql,
        faiss_add,
        faiss_save,
        find_match_faiss,
        quality_score,
        get_best_face,
        save_face_images,
        l2_norm as enroll_l2_norm,
        # constants
        SIMILARITY_THRESHOLD,
        ANGLES_NEEDED,
        ANGLE_PROMPTS,
        CAPTURE_INTERVAL,
        NORM_MIN,
        IMAGES_DIR,
        FAISS_DIR,
    )
except ImportError as _e:
    print(f"[API] Cannot import Enroll_student: {_e}")
    sys.exit(1)

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("[API] pip install insightface onnxruntime")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  API-level tuning (OOM / timeout guards — doesn't touch script constants)
# ─────────────────────────────────────────────────────────────────────────────

MAX_CONCURRENT_SESSIONS  = 1          # One physical camera — only 1 session at a time
MAX_CONCURRENT_ENROLLMENTS = 2        # OOM guard: cap active enrollment jobs
MODEL_LOAD_TIMEOUT_S     = 60         # asyncio timeout for InsightFace model load
SSE_IDLE_TIMEOUT_S       = 120        # seconds of no-subscriber before SSE disconnects
MJPEG_FRAME_QUEUE_SIZE   = 2          # per-session: drop old frames, never stall
MJPEG_QUALITY            = 70         # JPEG quality for MJPEG stream (0-100)
ENROLL_JOB_TTL_S         = 300        # auto-cleanup completed/failed jobs after 5 min
WRITE_QUEUE_DRAIN_TIMEOUT = 3.0       # seconds to wait for DB write queue to drain

# ─────────────────────────────────────────────────────────────────────────────
#  Shared model (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────

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
                warnings.filterwarnings("ignore", category=FutureWarning,
                                        module="insightface")
                return fa
            print(f"[API] Loading InsightFace {MODEL_NAME} ...")
            _insight_app = await asyncio.wait_for(
                loop.run_in_executor(None, _load),
                timeout=MODEL_LOAD_TIMEOUT_S,
            )
            print("[API] Model ready.")
    return _insight_app


# ─────────────────────────────────────────────────────────────────────────────
#  Session state (Attendance)
# ─────────────────────────────────────────────────────────────────────────────

class AttendanceSession:
    """Holds all mutable state for one live attendance session."""

    def __init__(self, session_id: str, class_name: str, teacher_name: str,
                 students: List[Dict], class_id: int,
                 already_present: Set[int], db_session_id: str) -> None:
        self.session_id      = session_id
        self.class_name      = class_name
        self.teacher_name    = teacher_name
        self.students        = students
        self.class_id        = class_id
        self.db_session_id   = db_session_id

        self.id_to_student: Dict[int, Dict] = {s["student_id"]: s for s in students}
        self.all_ids: Set[int]              = {s["student_id"] for s in students}
        self.thresholds                     = compute_adaptive_thresholds(students)
        self.index                          = build_class_index(students)

        # Write queue + background DB writer thread
        self.write_queue: _queue.Queue      = _queue.Queue(maxsize=WRITE_QUEUE_MAX)
        self.present_ids: Set[int]          = set(already_present)
        self._writer_thread                 = threading.Thread(
            target=db_writer_thread,
            args=(self.write_queue, self.present_ids),
            daemon=True,
        )
        self._writer_thread.start()

        # Frame tracking
        self.present_log:  List[Dict]        = []
        self.last_marked:  Dict[int, float]  = {}
        self.confirm_buf:  Dict[int, int]    = {}
        self.smooth_buf:   Dict[int, deque]  = {}
        self.last_name:    Dict[int, str]    = {}
        self.last_sim:     Dict[int, float]  = {}
        self.last_boxes:   List              = []
        self.prev_gray:    Optional[np.ndarray] = None
        self.frame_count:  int              = 0
        self.session_start: float           = time.time()
        self.fps:          float            = 0.0
        self.fps_counter:  int             = 0
        self.fps_timer:    float           = time.time()
        self.msg:          str             = "Waiting for students..."
        self.msg_color:    tuple           = (180, 180, 180)
        self.active:       bool            = True

        # Capture the running event loop so the background processing thread
        # can push frames into frame_q via run_coroutine_threadsafe.
        self._loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()

        # MJPEG frame queue (drops old frames — no OOM from slow consumers)
        self.frame_q: asyncio.Queue = asyncio.Queue(maxsize=MJPEG_FRAME_QUEUE_SIZE)

        # Camera + process thread started LAZILY on first /stream connect
        # This avoids MSMF conflicts if the client hasn't opened the stream yet.
        self.cam: Optional[CameraThread] = None
        self._proc_thread: Optional[threading.Thread] = None
        self._cam_started = False
        self._cam_lock    = threading.Lock()

    def start_camera(self) -> None:
        """Start camera + processing thread (called lazily on first /stream connect)."""
        with self._cam_lock:
            if self._cam_started:
                return
            self._cam_started = True
            self.cam = CameraThread(CAMERA_INDEX)
            self.cam.start()
            self._proc_thread = threading.Thread(
                target=self._process_loop, daemon=True)
            self._proc_thread.start()
            print(f"[SESSION:{self.session_id[:8]}] Camera started.")

    def _process_loop(self) -> None:
        """Runs the frame processing loop (identical to live_attendance run())."""
        try:
            while self.active:
                if self.cam is None:
                    time.sleep(0.05)
                    continue
                frame = self.cam.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                self._process_frame(frame)
        except Exception as e:
            print(f"[SESSION:{self.session_id[:8]}] Process loop error: {e}")
        finally:
            if self.cam is not None:
                self.cam.stop()

    def _process_frame(self, frame: np.ndarray) -> None:
        """Process one frame — logic taken verbatim from live_attendance.run()."""
        self.frame_count += 1
        self.fps_counter += 1
        now_t = time.time()
        if now_t - self.fps_timer >= 1.0:
            self.fps         = self.fps_counter / (now_t - self.fps_timer)
            self.fps_counter = 0
            self.fps_timer   = now_t

        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Frame skip (draw cached boxes on skipped frames) ──────────────
        if self.frame_count % FRAME_SKIP != 0:
            for (bx1, by1, bx2, by2, bcol, blabel) in self.last_boxes:
                cv2.rectangle(display, (bx1, by1), (bx2, by2), bcol, 2)
                cv2.putText(display, blabel, (bx1, max(by1-8, 55)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, bcol, 2, cv2.LINE_AA)
            draw_hud(display, self.class_name, self.present_log,
                     len(self.students), self.session_start, self.fps,
                     gray, self.msg, self.msg_color)
            self._push_frame(display)
            self.prev_gray = gray
            return

        motion         = has_motion(gray, self.prev_gray)
        self.prev_gray = gray

        model = _insight_app  # already loaded at startup
        if model is None:
            self._push_frame(display)
            return

        try:
            faces = model.get(frame)
        except Exception as e:
            faces = []
            print(f"[WARN] InsightFace: {e}")

        if not faces:
            self.last_boxes = []
            for k in list(self.confirm_buf.keys()):
                self.confirm_buf[k] -= 1
                if self.confirm_buf[k] <= 0:
                    del self.confirm_buf[k]
        else:
            scored = [(f, *live_quality(f, gray)) for f in faces]
            good   = [(f, s, r) for f, s, r in scored
                      if f.embedding is not None and s >= 55.0]

            if good and motion:
                raw_results = faiss_batch_match(
                    [f.embedding for f, _, _ in good],
                    self.index, self.id_to_student, self.thresholds,
                )
                smoothed_embs: List[np.ndarray] = []
                smoothed_meta: List = []

                for (face, score, reason), (_, sid, _, _) in zip(good, raw_results):
                    if sid == -1:
                        smoothed_embs.append(l2_norm(face.embedding.copy()))
                        smoothed_meta.append((face, score, reason))
                        continue
                    # OOM guard: cap smooth_buf entries
                    if sid not in self.smooth_buf and len(self.smooth_buf) >= SMOOTH_BUF_MAX:
                        del self.smooth_buf[next(iter(self.smooth_buf))]
                    if sid not in self.smooth_buf:
                        self.smooth_buf[sid] = deque(maxlen=SMOOTH_WINDOW)
                    self.smooth_buf[sid].append((face.embedding.copy(), score))
                    buf_embs    = [e for e, _ in self.smooth_buf[sid]]
                    buf_weights = ([w for _, w in self.smooth_buf[sid]]
                                   if QUALITY_WEIGHT else [1.0]*len(self.smooth_buf[sid]))
                    smoothed_embs.append(quality_weighted_avg(buf_embs, buf_weights))
                    smoothed_meta.append((face, score, reason))

                final = faiss_batch_match(
                    smoothed_embs, self.index, self.id_to_student, self.thresholds)

                now = time.time()
                self.last_boxes = []

                for (face, score, reason), (matched, sid, name, avg_sim) \
                        in zip(smoothed_meta, final):
                    x1, y1, x2, y2 = get_bbox(face)

                    if matched and sid != -1:
                        self.last_name[sid] = name
                        self.last_sim[sid]  = avg_sim
                        self.confirm_buf[sid] = self.confirm_buf.get(sid, 0) + 1
                        for other in list(self.confirm_buf):
                            if other != sid:
                                self.confirm_buf[other] -= 1
                                if self.confirm_buf[other] <= 0:
                                    del self.confirm_buf[other]

                        already_marked = sid in self.present_ids
                        box_col = (0, 210, 80) if already_marked else (0, 180, 255)
                        prefix  = "✓ " if already_marked else ""
                        label   = f"{prefix}{name}  {avg_sim:.2f}"
                        cv2.rectangle(display, (x1, y1), (x2, y2), box_col, 2)
                        cv2.putText(display, label, (x1, max(y1-8, 55)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                                    box_col, 2, cv2.LINE_AA)
                        self.last_boxes.append((x1, y1, x2, y2, box_col, label))

                        if not already_marked:
                            bw     = max(x2 - x1, 1)
                            pct    = self.confirm_buf.get(sid, 0) / CONFIRM_FRAMES
                            by     = max(y1 - 18, 58)
                            filled = min(int(pct * bw), bw)
                            cv2.rectangle(display, (x1, by), (x2, by+5), (40,40,40), -1)
                            if filled > 0:
                                cv2.rectangle(display, (x1, by),
                                              (x1+filled, by+5), (0, 210, 80), -1)

                        if (self.confirm_buf.get(sid, 0) >= CONFIRM_FRAMES
                                and not already_marked
                                and (now - self.last_marked.get(sid, 0)) > DEDUP_SECONDS):
                            try:
                                self.write_queue.put_nowait({
                                    "type":       "present",
                                    "student_id": sid,
                                    "class_id":   self.class_id,
                                    "session_id": self.db_session_id,
                                    "confidence": avg_sim,
                                })
                            except _queue.Full:
                                print(f"[WARN] Queue full — {name} retries next pass")
                                continue
                            self.last_marked[sid] = now
                            # OOM guard: cap present_log
                            if len(self.present_log) >= PRESENT_LOG_MAX:
                                self.present_log.pop(0)
                            self.present_log.append({
                                "student_id": sid, "name": name,
                                "sim": avg_sim,
                                "time": datetime.now().strftime("%H:%M:%S"),
                            })
                            self.msg       = f"Present: {name}"
                            self.msg_color = (0, 220, 80)
                            beep_present()
                            ts = datetime.now().strftime("%H:%M:%S")
                            print(f"[PRESENT]  {ts}  |  {name:<26}"
                                  f"  sim={avg_sim:.3f}"
                                  f"  thresh={self.thresholds.get(sid,0.45):.3f}"
                                  f"  id={sid}")
                    else:
                        for k in list(self.confirm_buf):
                            self.confirm_buf[k] -= 1
                            if self.confirm_buf[k] <= 0:
                                del self.confirm_buf[k]
                        label = f"Unknown {avg_sim:.2f}"
                        cv2.rectangle(display, (x1, y1), (x2, y2), (40, 40, 220), 2)
                        cv2.putText(display, label, (x1, max(y1-8, 55)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                                    (40, 80, 220), 1, cv2.LINE_AA)
                        self.last_boxes.append((x1, y1, x2, y2, (40, 40, 220), label))

            elif good and not motion:
                for face, score, reason in good:
                    x1, y1, x2, y2 = get_bbox(face)
                    best_sid = max(self.smooth_buf,
                                   key=lambda s: sum(w for _, w in self.smooth_buf[s])
                                               / len(self.smooth_buf[s])
                                   ) if self.smooth_buf else -1
                    if best_sid != -1:
                        name = self.last_name.get(best_sid, "")
                        sim  = self.last_sim.get(best_sid, 0.0)
                        col  = (0, 210, 80) if best_sid in self.present_ids else (0, 180, 255)
                        cv2.rectangle(display, (x1, y1), (x2, y2), col, 2)
                        prefix = "✓ " if best_sid in self.present_ids else ""
                        cv2.putText(display, f"{prefix}{name}  {sim:.2f}",
                                    (x1, max(y1-8, 55)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2, cv2.LINE_AA)
                    else:
                        cv2.rectangle(display, (x1, y1), (x2, y2), (100,100,100), 1)

            # Low-quality face boxes
            for face, score, reason in scored:
                if score < 55.0:
                    x1, y1, x2, y2 = get_bbox(face)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (60,60,60), 1)
                    cv2.putText(display, f"{reason} {score:.0f}%",
                                (x1, max(y1-8, 55)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                                (120,120,120), 1, cv2.LINE_AA)

        draw_hud(display, self.class_name, self.present_log,
                 len(self.students), self.session_start, self.fps,
                 gray, self.msg, self.msg_color)
        self._push_frame(display)

    def _push_frame(self, frame: np.ndarray) -> None:
        """Encode frame as JPEG and push to the async frame queue (thread-safe).

        Uses loop.call_soon_threadsafe() + a plain callable — the lightest
        possible path from a background thread into the asyncio event loop.
        Drops the oldest frame when the queue is full so the stream never stalls.
        """
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
        if not ok:
            return
        jpg = buf.tobytes()

        def _put():
            # Runs inside the event loop thread — asyncio.Queue ops are safe here
            if self.frame_q.full():
                try:
                    self.frame_q.get_nowait()
                except Exception:
                    pass
            try:
                self.frame_q.put_nowait(jpg)
            except Exception:
                pass

        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(_put)

    def get_status(self) -> Dict[str, Any]:
        elapsed = int(time.time() - self.session_start)
        m, s    = divmod(elapsed, 60)
        absent  = self.all_ids - self.present_ids
        return {
            "session_id":   self.session_id,
            "class_name":   self.class_name,
            "teacher_name": self.teacher_name,
            "elapsed":      f"{m:02d}:{s:02d}",
            "fps":          round(self.fps, 1),
            "total":        len(self.students),
            "present":      len(self.present_ids),
            "absent":       len(absent),
            "present_log":  self.present_log[-20:],   # last 20 entries
        }

    def stop(self) -> None:
        """Signal the processing loop to exit.  Camera cleanup happens
        in _process_loop's finally block (runs in its own thread, never
        blocks the event loop)."""
        self.active = False


# ─────────────────────────────────────────────────────────────────────────────
#  Enrollment job state
# ─────────────────────────────────────────────────────────────────────────────

class EnrollJob:
    """State machine for one enrollment job."""

    STATES = ("waiting",      # idle, watching camera
               "capturing",   # capturing 3 angles
               "checking",    # FAISS duplicate check in progress
               "enrolling",   # writing to MySQL + FAISS
               "done",        # success
               "duplicate",   # already enrolled
               "error",       # unrecoverable error
               "cancelled")   # user cancelled

    def __init__(self, job_id: str, class_name: str,
                 roll_no: str, name: str) -> None:
        self.job_id      = job_id
        self.class_name  = class_name
        self.roll_no     = roll_no
        self.name        = name
        self.state       = "waiting"
        self.angle_done  = 0
        self.embs: List[np.ndarray] = []
        self.crops: List[np.ndarray] = []
        self.message     = "Ready — waiting for face..."
        self.quality     = 0.0
        self.reason      = ""
        self.created_at  = time.time()
        self.finished_at: Optional[float] = None
        self.student_id: Optional[int]    = None
        # NOTE: sse_queue is asyncio.Queue (consumed by async SSE generator)
        # trigger_event / cancel_event are threading.Event (set from executor thread)
        self.sse_queue: asyncio.Queue             = asyncio.Queue(maxsize=50)
        self.trigger_event: threading.Event       = threading.Event()
        self.cancel_event:  threading.Event       = threading.Event()

    def push_event(self, event: str, data: Dict) -> None:
        """
        Push an SSE event (thread-safe; drops oldest if queue full — OOM guard).
        Can be called from any thread; uses run_coroutine_threadsafe to enqueue
        into the asyncio.Queue safely.
        """
        payload = {"event": event, "data": data,
                   "ts": datetime.now().strftime("%H:%M:%S")}
        _loop = _main_loop
        if _loop is None or not _loop.is_running():
            return
        async def _put():
            try:
                self.sse_queue.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    self.sse_queue.get_nowait()
                except Exception:
                    pass
                try:
                    self.sse_queue.put_nowait(payload)
                except Exception:
                    pass
        asyncio.run_coroutine_threadsafe(_put(), _loop)

    def is_terminal(self) -> bool:
        return self.state in ("done", "duplicate", "error", "cancelled")

    def elapsed_ttl(self) -> float:
        ref = self.finished_at or self.created_at
        return time.time() - ref


# ─────────────────────────────────────────────────────────────────────────────
#  Global registries
# ─────────────────────────────────────────────────────────────────────────────

_sessions:    Dict[str, AttendanceSession] = {}
_enroll_jobs: Dict[str, EnrollJob]         = {}
_sessions_lock    = asyncio.Lock()
_enroll_jobs_lock = asyncio.Lock()

# Captured in lifespan so background threads can schedule coroutines safely
_main_loop: Optional[asyncio.AbstractEventLoop] = None

# ─────────────────────────────────────────────────────────────────────────────
#  Lifespan — startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    global _main_loop
    _main_loop = asyncio.get_event_loop()
    print("[API] Starting up …")
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(FAISS_DIR,  exist_ok=True)

    # Pre-load model (fail fast rather than on first request)
    try:
        await _get_model()
    except asyncio.TimeoutError:
        print(f"[API] Model load timed out after {MODEL_LOAD_TIMEOUT_S}s — continuing")
    except Exception as e:
        print(f"[API] Model load failed: {e} — endpoints will retry on first request")

    yield  # ← app runs here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    print("[API] Shutting down …")
    async with _sessions_lock:
        for sid, sess in list(_sessions.items()):
            try:
                sess.stop()
                sess.write_queue.put(None)
            except Exception:
                pass
    async with _enroll_jobs_lock:
        for jid, job in list(_enroll_jobs.items()):
            job.cancel_event.set()
    print("[API] Shutdown complete.")

# ─────────────────────────────────────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Attendance Management System API",
    version="1.0.0",
    description=(
        "FastAPI wrapper for live_attendance, inspect_index, and Enroll_student. "
        "All core algorithms remain in the original scripts."
    ),
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
async def health():
    return {
        "status":       "ok",
        "model_loaded": _insight_app is not None,
        "sessions":     len(_sessions),
        "enroll_jobs":  len(_enroll_jobs),
        "timestamp":    datetime.utcnow().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Browser UI  (replaces the local cv2.imshow window)
# ─────────────────────────────────────────────────────────────────────────────

_ATTENDANCE_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AI Attendance — Live</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0d0f14;color:#e2e8f0;font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
  header{background:linear-gradient(135deg,#1e293b,#0f172a);border-bottom:1px solid #1e3a5f;
         padding:14px 24px;display:flex;align-items:center;gap:14px}
  header h1{font-size:1.25rem;font-weight:700;color:#38bdf8}
  .badge{background:#1e3a5f;color:#7dd3fc;border-radius:999px;padding:2px 10px;font-size:.75rem}
  .wrap{display:grid;grid-template-columns:1fr 340px;gap:18px;padding:20px 24px;max-width:1400px;margin:auto}
  .cam-card{background:#111827;border:1px solid #1e293b;border-radius:12px;overflow:hidden;position:relative}
  .cam-card img{width:100%;display:block;background:#000;min-height:360px}
  .cam-placeholder{width:100%;min-height:420px;display:flex;flex-direction:column;
                   align-items:center;justify-content:center;gap:12px;color:#374151}
  .cam-placeholder svg{width:72px;height:72px;opacity:.4}
  .cam-label{position:absolute;top:10px;left:10px;background:rgba(0,0,0,.65);
             border:1px solid #1e3a5f;border-radius:6px;padding:4px 10px;font-size:.78rem;color:#38bdf8}
  .rec-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#ef4444;
           margin-right:5px;animation:blink 1s step-start infinite}
  @keyframes blink{50%{opacity:0}}
  .panel{display:flex;flex-direction:column;gap:14px}
  .card{background:#111827;border:1px solid #1e293b;border-radius:12px;padding:18px}
  .card h2{font-size:.85rem;font-weight:600;color:#7dd3fc;text-transform:uppercase;
           letter-spacing:.08em;margin-bottom:12px}
  label{display:block;font-size:.78rem;color:#94a3b8;margin-bottom:4px;margin-top:8px}
  input{width:100%;background:#0d1117;border:1px solid #1e293b;border-radius:7px;
        color:#e2e8f0;padding:8px 11px;font-size:.88rem;outline:none}
  input:focus{border-color:#38bdf8}
  .btn{display:block;width:100%;border:none;border-radius:8px;padding:10px;font-size:.9rem;
       font-weight:600;cursor:pointer;transition:opacity .15s;color:#fff}
  .btn:disabled{opacity:.4;cursor:not-allowed}
  .btn-start{background:linear-gradient(135deg,#0ea5e9,#6366f1);margin-top:14px}
  .btn-stop{background:linear-gradient(135deg,#ef4444,#dc2626);margin-top:8px}
  .stat-row{display:flex;justify-content:space-between;font-size:.82rem;padding:5px 0;border-bottom:1px solid #1e293b}
  .stat-row:last-child{border:none}
  .stat-val{font-weight:700;color:#38bdf8}
  .present-val{color:#4ade80} .absent-val{color:#f87171}
  #log{max-height:220px;overflow-y:auto;font-size:.78rem}
  .log-item{display:flex;gap:8px;padding:5px 0;border-bottom:1px solid #1e293b}
  .log-time{color:#475569;min-width:52px} .log-name{flex:1} .log-sim{color:#6366f1;font-family:monospace}
  #alert{border-radius:8px;padding:9px 13px;font-size:.82rem;display:none;margin-top:8px}
  .a-err{background:#450a0a;border:1px solid #ef4444;color:#fca5a5;display:block!important}
  .a-ok{background:#052e16;border:1px solid #22c55e;color:#86efac;display:block!important}
  .enroll-link{display:block;text-align:center;color:#818cf8;font-size:.8rem;text-decoration:none;
               margin-top:4px;padding:8px;border:1px solid #312e81;border-radius:7px}
  .enroll-link:hover{background:#1e1b4b}
</style>
</head>
<body>
<header>
  <h1>&#127919; AI Attendance System</h1>
  <span class="badge" id="hdr-badge">&#9679; Offline</span>
  <span style="margin-left:auto;font-size:.8rem">
    <a href="/docs" style="color:#38bdf8;text-decoration:none">API Docs</a>
  </span>
</header>
<div class="wrap">
  <div class="cam-card">
    <div id="cam-placeholder" class="cam-placeholder">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
        <path d="M15 10l4.553-2.276A1 1 0 0121 8.723v6.554a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/>
      </svg>
      <p>Start a session to see the live feed</p>
    </div>
    <img id="cam-img" style="display:none" alt="Live feed"/>
    <div class="cam-label" id="cam-label" style="display:none">
      <span class="rec-dot"></span><span id="cam-class"></span>
    </div>
  </div>
  <div class="panel">
    <div class="card">
      <h2>Start Session</h2>
      <label>Class</label><input id="cls" placeholder="e.g. btech_cs_1"/>
      <label>Teacher</label><input id="teacher" placeholder="e.g. Dr. Sharma"/>
      <button class="btn btn-start" id="btn-start" onclick="startSession()">&#9654; Start Attendance</button>
      <button class="btn btn-stop" id="btn-stop" style="display:none" onclick="stopSession()">&#9632; Stop &amp; Save</button>
      <div id="alert"></div>
    </div>
    <div class="card" id="status-card" style="display:none">
      <h2>Live Status</h2>
      <div class="stat-row"><span>Elapsed</span><span class="stat-val" id="s-elapsed">&#8212;</span></div>
      <div class="stat-row"><span>FPS</span><span class="stat-val" id="s-fps">&#8212;</span></div>
      <div class="stat-row"><span>Total</span><span class="stat-val" id="s-total">&#8212;</span></div>
      <div class="stat-row"><span>Present</span><span class="stat-val present-val" id="s-present">0</span></div>
      <div class="stat-row"><span>Absent</span><span class="stat-val absent-val" id="s-absent">&#8212;</span></div>
    </div>
    <div class="card" id="log-card" style="display:none">
      <h2>Present Log</h2><div id="log"></div>
    </div>
    <a class="enroll-link" href="/ui/enroll">+ Enroll a New Student</a>
  </div>
</div>
<script>
let sessionId=null,pollTimer=null,seenIds=new Set();
function showAlert(msg,ok=false){const el=document.getElementById('alert');el.textContent=msg;el.className=ok?'a-ok':'a-err';}
async function startSession(){
  const cls=document.getElementById('cls').value.trim();
  const teacher=document.getElementById('teacher').value.trim();
  if(!cls||!teacher){showAlert('Fill in class and teacher name.');return;}
  document.getElementById('btn-start').disabled=true;
  try{
    const r=await fetch('/attendance/session/start',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({class_name:cls,teacher_name:teacher,resume:false})});
    const d=await r.json();
    if(!r.ok){showAlert(d.detail||'Error');document.getElementById('btn-start').disabled=false;return;}
    sessionId=d.session_id;
    document.getElementById('hdr-badge').textContent='\u25cf Live';
    document.getElementById('hdr-badge').style.color='#4ade80';
    document.getElementById('cam-class').textContent=cls;
    document.getElementById('cam-label').style.display='';
    document.getElementById('cam-placeholder').style.display='none';
    const img=document.getElementById('cam-img');
    img.src='/attendance/session/'+sessionId+'/stream?t='+Date.now();
    img.style.display='block';
    document.getElementById('btn-start').style.display='none';
    document.getElementById('btn-stop').style.display='block';
    document.getElementById('status-card').style.display='';
    document.getElementById('log-card').style.display='';
    pollTimer=setInterval(pollStatus,1500);
  }catch(e){showAlert('Network: '+e.message);document.getElementById('btn-start').disabled=false;}
}
async function pollStatus(){
  if(!sessionId)return;
  try{
    const r=await fetch('/attendance/session/'+sessionId+'/status');
    if(!r.ok)return;
    const d=await r.json();
    document.getElementById('s-elapsed').textContent=d.elapsed||'\u2014';
    document.getElementById('s-fps').textContent=d.fps||'\u2014';
    document.getElementById('s-total').textContent=d.total||'\u2014';
    document.getElementById('s-present').textContent=d.present||0;
    document.getElementById('s-absent').textContent=d.absent||'\u2014';
    const log=document.getElementById('log');
    (d.present_log||[]).slice().reverse().forEach(e=>{
      if(seenIds.has(e.student_id+'@'+e.time))return;
      seenIds.add(e.student_id+'@'+e.time);
      const row=document.createElement('div');row.className='log-item';
      row.innerHTML='<span class="log-time">'+e.time+'</span><span class="log-name">'+e.name+'</span><span class="log-sim">'+(e.sim*100).toFixed(1)+'%</span>';
      log.prepend(row);
    });
  }catch(_){}
}
async function stopSession(){
  if(!sessionId)return;
  clearInterval(pollTimer);
  document.getElementById('btn-stop').disabled=true;
  try{
    const r=await fetch('/attendance/session/'+sessionId+'/stop',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({confirm:true})});
    const d=await r.json();
    document.getElementById('cam-img').src='';
    document.getElementById('cam-img').style.display='none';
    document.getElementById('cam-placeholder').style.display='flex';
    document.getElementById('cam-label').style.display='none';
    document.getElementById('btn-start').style.display='block';
    document.getElementById('btn-start').disabled=false;
    document.getElementById('btn-stop').style.display='none';
    document.getElementById('hdr-badge').textContent='\u25cf Offline';
    document.getElementById('hdr-badge').style.color='';
    showAlert('Saved \u2014 Present: '+(d.present||0)+' | Absent: '+(d.absent||0),true);
    sessionId=null;seenIds=new Set();
  }catch(e){showAlert('Stop error: '+e.message);}
  document.getElementById('btn-stop').disabled=false;
}
</script>
</body></html>"""

_ENROLL_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><title>Enroll Student</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0d0f14;color:#e2e8f0;font-family:'Segoe UI',system-ui,sans-serif}
  header{background:linear-gradient(135deg,#1e293b,#0f172a);border-bottom:1px solid #1e3a5f;
         padding:14px 24px}
  header h1{font-size:1.25rem;font-weight:700;color:#a78bfa}
  .wrap{max-width:580px;margin:28px auto;padding:0 20px;display:flex;flex-direction:column;gap:16px}
  .card{background:#111827;border:1px solid #1e293b;border-radius:12px;padding:22px}
  .card h2{font-size:.85rem;color:#a78bfa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px}
  label{font-size:.78rem;color:#94a3b8;display:block;margin-bottom:4px;margin-top:10px}
  input{width:100%;background:#0d1117;border:1px solid #1e293b;border-radius:7px;
        color:#e2e8f0;padding:8px 11px;font-size:.88rem;outline:none}
  input:focus{border-color:#a78bfa}
  .btn{border:none;border-radius:8px;padding:11px;font-size:.9rem;font-weight:600;
       cursor:pointer;width:100%;margin-top:12px;color:#fff}
  .btn:disabled{opacity:.4;cursor:not-allowed}
  .btn-enroll{background:linear-gradient(135deg,#7c3aed,#4f46e5)}
  .btn-trigger{background:linear-gradient(135deg,#059669,#047857)}
  .btn-cancel{background:linear-gradient(135deg,#9f1239,#be123c)}
  #events{max-height:260px;overflow-y:auto;font-size:.78rem;background:#0d1117;
          border:1px solid #1e293b;border-radius:8px;padding:10px;margin-top:12px;display:none}
  .ev{padding:4px 0;border-bottom:1px solid #1e293b;display:flex;gap:8px}
  .ev:last-child{border:none}
  .ev-time{color:#475569;min-width:48px}
  .type-quality_update{color:#38bdf8} .type-angle_captured,.type-enrolled{color:#4ade80}
  .type-duplicate,.type-error{color:#f87171} .type-state_change{color:#fbbf24}
  .qbar-wrap{height:6px;background:#1e293b;border-radius:3px;margin-top:8px;overflow:hidden}
  .qbar{height:6px;background:linear-gradient(90deg,#ef4444,#eab308,#4ade80);
        border-radius:3px;transition:width .25s}
  #qbar-label{font-size:.75rem;color:#64748b;margin-top:4px}
  .alert{border-radius:8px;padding:9px 13px;font-size:.82rem;display:none;margin-top:8px}
  .a-err{background:#450a0a;border:1px solid #ef4444;color:#fca5a5;display:block}
  .a-ok{background:#052e16;border:1px solid #22c55e;color:#86efac;display:block}
  .back{color:#6366f1;font-size:.8rem;text-decoration:none}
  .dots{display:flex;gap:8px;margin-top:12px}
  .dot{width:30px;height:30px;border-radius:50%;background:#1e293b;border:2px solid #334155;
       display:flex;align-items:center;justify-content:center;font-size:.7rem;color:#64748b}
  .dot.done{background:#4ade80;border-color:#4ade80;color:#052e16;font-weight:700}
</style>
</head>
<body>
<header><h1>&#128100; Enroll New Student</h1></header>
<div class="wrap">
  <a class="back" href="/">&#8592; Back to Attendance</a>
  <div class="card">
    <h2>Student Info</h2>
    <label>Class</label><input id="cls" placeholder="e.g. btech_cs_1"/>
    <label>Roll No</label><input id="roll" placeholder="e.g. BT2401"/>
    <label>Full Name</label><input id="name" placeholder="e.g. Priya Sharma"/>
    <button class="btn btn-enroll" id="btn-start" onclick="startEnroll()">Start Enrollment</button>
    <div id="alert" class="alert"></div>
  </div>
  <div class="card" id="ctrl-card" style="display:none">
    <h2>Capture</h2>
    <p id="prompt-txt" style="font-size:.9rem">Waiting for face...</p>
    <div class="qbar-wrap"><div class="qbar" id="qbar" style="width:0%"></div></div>
    <div id="qbar-label">Quality: --</div>
    <div class="dots" id="dots"></div>
    <button class="btn btn-trigger" id="btn-trigger" onclick="trigger()" disabled>&#128247; Capture This Angle</button>
    <button class="btn btn-cancel" id="btn-cancel" onclick="cancelJob()">Cancel</button>
    <div id="events"></div>
  </div>
</div>
<script>
let jobId=null,es=null;
function showAlert(msg,ok=false){const el=document.getElementById('alert');el.textContent=msg;el.className='alert '+(ok?'a-ok':'a-err');}
async function startEnroll(){
  const cls=document.getElementById('cls').value.trim();
  const roll=document.getElementById('roll').value.trim();
  const name=document.getElementById('name').value.trim();
  if(!cls||!roll||!name){showAlert('Fill all fields.');return;}
  document.getElementById('btn-start').disabled=true;
  try{
    const r=await fetch('/enroll/start',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({class_name:cls,roll_no:roll,name:name})});
    const d=await r.json();
    if(!r.ok){showAlert(d.detail||'Error');document.getElementById('btn-start').disabled=false;return;}
    jobId=d.job_id;
    document.getElementById('ctrl-card').style.display='';
    const dots=document.getElementById('dots');
    dots.innerHTML='';
    for(let i=0;i<3;i++){const dot=document.createElement('div');dot.className='dot';dot.id='dot-'+i;dot.textContent=i+1;dots.appendChild(dot);}
    subscribeSSE();
  }catch(e){showAlert('Network: '+e.message);document.getElementById('btn-start').disabled=false;}
}
function subscribeSSE(){
  es=new EventSource('/enroll/status/'+jobId);
  const evBox=document.getElementById('events');evBox.style.display='block';
  es.onmessage=function(e){
    const p=JSON.parse(e.data);
    if(p.event==='stream_end'){es.close();return;}
    const row=document.createElement('div');row.className='ev';
    row.innerHTML='<span class="ev-time">'+p.ts+'</span><span class="type-'+p.event+'">'+p.event+'</span><span>'+(p.data.message||p.data.reason||'').slice(0,60)+'</span>';
    evBox.prepend(row);
    if(p.event==='quality_update'){
      const q=p.data.quality||0;
      document.getElementById('qbar').style.width=Math.min(q,100)+'%';
      document.getElementById('qbar-label').textContent='Quality: '+q.toFixed(0)+'% | Angle '+(p.data.angle_done+1)+'/'+p.data.angles_needed;
      document.getElementById('prompt-txt').textContent=p.data.prompt||'Position face...';
      document.getElementById('btn-trigger').disabled=(q<75);
    }
    if(p.event==='angle_captured'){
      const dot=document.getElementById('dot-'+(p.data.angle-1));
      if(dot){dot.className='dot done';}
    }
    if(p.event==='enrolled'){showAlert('Enrolled! ID='+p.data.student_id,true);es.close();}
    if(p.event==='duplicate'){showAlert('Duplicate: '+p.data.name+' (sim='+p.data.sim+')');es.close();}
    if(p.event==='error'){showAlert(p.data.message||'Error');es.close();}
  };
}
async function trigger(){
  if(!jobId)return;
  document.getElementById('btn-trigger').disabled=true;
  await fetch('/enroll/'+jobId+'/trigger',{method:'POST'});
}
async function cancelJob(){
  if(jobId)await fetch('/enroll/'+jobId,{method:'DELETE'});
  if(es)es.close();
  document.getElementById('ctrl-card').style.display='none';
  document.getElementById('btn-start').disabled=false;
  jobId=null;
}
</script>
</body></html>"""


@app.get("/", tags=["ui"], response_class=HTMLResponse)
async def ui_home():
    """Live attendance dashboard — shows camera feed + controls in the browser."""
    return HTMLResponse(content=_ATTENDANCE_UI)


@app.get("/ui/enroll", tags=["ui"], response_class=HTMLResponse)
async def ui_enroll():
    """Student enrollment UI — camera-guided 3-angle capture."""
    return HTMLResponse(content=_ENROLL_UI)


# ─────────────────────────────────────────────────────────────────────────────
#  /inspect — wraps inspect_index.py
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/inspect/classes", tags=["inspect"])
async def inspect_list_classes():
    """List all classes that have a FAISS index file."""
    loop = asyncio.get_event_loop()
    try:
        classes = await loop.run_in_executor(None, list_available)
    except SystemExit:
        raise HTTPException(status_code=404, detail="No FAISS indexes found")
    return {"classes": classes}


@app.get("/inspect/{class_name}", tags=["inspect"])
async def inspect_class(class_name: str):
    """Return full class inspection data as JSON."""
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
    # Convert numpy arrays to serialisable form
    for s in data.get("students", []):
        s.pop("embeddings", None)  # strip raw blobs from JSON response
    return JSONResponse(content=data)


@app.get("/inspect/{class_name}/html", tags=["inspect"])
async def inspect_class_html(class_name: str):
    """Return the HTML inspection report (identical to opening it in a browser)."""
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


# ─────────────────────────────────────────────────────────────────────────────
#  /attendance — wraps live_attendance.py
# ─────────────────────────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    class_name:   str
    teacher_name: str
    resume:       bool = False   # if True, try to resume today's existing session


@app.post("/attendance/session/start", tags=["attendance"])
async def start_session(req: StartSessionRequest):
    """Start (or resume) a live attendance session for a class."""

    # OOM guard: cap concurrent sessions
    async with _sessions_lock:
        if len(_sessions) >= MAX_CONCURRENT_SESSIONS:
            raise HTTPException(
                status_code=429,
                detail=f"Max concurrent sessions ({MAX_CONCURRENT_SESSIONS}) reached. "
                        "Stop an existing session first.",
            )

    # Ensure model is loaded
    await _get_model()

    loop = asyncio.get_event_loop()

    # DB work in executor (blocking)
    def _setup():
        students = db_load_class_students(req.class_name)
        if not students:
            raise ValueError(f"No students found in class '{req.class_name}'")
        class_id = students[0]["class_id"]
        existing = db_get_todays_session(class_id) if req.resume else None
        if existing and req.resume:
            db_sid         = str(existing["id"])
            already_present = db_get_already_present(db_sid)
        else:
            db_sid          = str(uuid.uuid4())
            already_present = set()
            db_create_session(db_sid, class_id, req.teacher_name)
        return students, class_id, db_sid, already_present

    try:
        students, class_id, db_sid, already_present = await asyncio.wait_for(
            loop.run_in_executor(None, _setup),
            timeout=30.0,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="DB setup timed out")

    session_id = str(uuid.uuid4())
    sess = AttendanceSession(
        session_id      = session_id,
        class_name      = req.class_name,
        teacher_name    = req.teacher_name,
        students        = students,
        class_id        = class_id,
        already_present = already_present,
        db_session_id   = db_sid,
    )

    async with _sessions_lock:
        _sessions[session_id] = sess

    return {
        "session_id":     session_id,
        "class_name":     req.class_name,
        "teacher_name":   req.teacher_name,
        "students":       len(students),
        "db_session_id":  db_sid,
        "resumed":        bool(already_present),
        "stream_url":     f"/attendance/session/{session_id}/stream",
        "status_url":     f"/attendance/session/{session_id}/status",
        "stop_url":       f"/attendance/session/{session_id}/stop",
    }


@app.get("/attendance/session/{session_id}/stream", tags=["attendance"])
async def stream_session(session_id: str):
    """MJPEG live stream of the annotated attendance frame."""
    async with _sessions_lock:
        sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Lazily start camera on first stream connect
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, sess.start_camera)

    async def generate():
        idle_since = time.time()
        while sess.active:
            try:
                jpg = await asyncio.wait_for(sess.frame_q.get(), timeout=2.0)
                idle_since = time.time()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            except asyncio.TimeoutError:
                if time.time() - idle_since > SSE_IDLE_TIMEOUT_S:
                    print(f"[STREAM:{session_id[:8]}] Idle timeout — closing stream")
                    break
                # send a keep-alive empty chunk
                continue

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/attendance/session/{session_id}/status", tags=["attendance"])
async def session_status(session_id: str):
    """JSON snapshot of the current session state."""
    async with _sessions_lock:
        sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess.get_status()


class StopSessionRequest(BaseModel):
    confirm: bool = True   # set False to just preview without saving


@app.post("/attendance/session/{session_id}/stop", tags=["attendance"])
async def stop_session(session_id: str, req: StopSessionRequest = StopSessionRequest()):
    """Stop session, mark absents, close DB session, return summary."""
    async with _sessions_lock:
        sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if not req.confirm:
        # Preview only
        absent_ids = sess.all_ids - sess.present_ids
        return {
            "preview": True,
            "present": len(sess.present_ids),
            "absent":  len(absent_ids),
        }

    # Signal the processing loop to exit (camera cleanup happens in
    # _process_loop's finally block — never blocks the event loop).
    sess.stop()

    # Send sentinel + drain write queue in an executor so the event loop
    # stays responsive and the HTTP response goes out fast.
    loop = asyncio.get_event_loop()

    def _drain_and_finalize():
        # Stop the writer thread
        try:
            sess.write_queue.put_nowait(None)
        except _queue.Full:
            pass
        # Brief wait for remaining items to flush
        deadline = time.time() + WRITE_QUEUE_DRAIN_TIMEOUT
        while not sess.write_queue.empty() and time.time() < deadline:
            time.sleep(0.05)
        absent_ids_local = sess.all_ids - sess.present_ids
        n_absent = db_mark_absent_bulk(
            list(absent_ids_local), sess.class_id, sess.db_session_id)
        db_close_session(sess.db_session_id, len(sess.present_ids), n_absent)
        return n_absent, absent_ids_local

    try:
        n_absent, absent_ids = await asyncio.wait_for(
            loop.run_in_executor(None, _drain_and_finalize),
            timeout=8.0,
        )
    except asyncio.TimeoutError:
        absent_ids = sess.all_ids - sess.present_ids
        n_absent   = len(absent_ids)
        print(f"[WARN] stop: DB finalize timed out — returning partial result")

    async with _sessions_lock:
        _sessions.pop(session_id, None)

    elapsed = int(time.time() - sess.session_start)
    m, s    = divmod(elapsed, 60)

    return {
        "session_id":  session_id,
        "class_name":  sess.class_name,
        "teacher":     sess.teacher_name,
        "duration":    f"{m:02d}:{s:02d}",
        "present":     len(sess.present_ids),
        "absent":      n_absent,
        "present_log": sess.present_log,
        "absent_log": [
            {"student_id": sid,
             "name":       sess.id_to_student[sid]["name"],
             "roll_no":    sess.id_to_student[sid]["roll_no"]}
            for sid in sorted(absent_ids)
            if sid in sess.id_to_student
        ],
    }


@app.delete("/attendance/session/{session_id}", tags=["attendance"])
async def quit_session(session_id: str):
    """Quit session WITHOUT saving (equivalent to pressing Q)."""
    async with _sessions_lock:
        sess = _sessions.pop(session_id, None)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    sess.stop()
    sess.write_queue.put(None)
    return {"session_id": session_id, "status": "quit — not saved"}


# ─────────────────────────────────────────────────────────────────────────────
#  /enroll — wraps Enroll_student.py
# ─────────────────────────────────────────────────────────────────────────────

class StartEnrollRequest(BaseModel):
    class_name: str
    roll_no:    str
    name:       str


@app.post("/enroll/start", tags=["enroll"])
async def enroll_start(req: StartEnrollRequest):
    """
    Start an enrollment job for one student.
    A background task watches the camera, checks quality, and waits for
    POST /enroll/{job_id}/trigger calls (one per angle) to capture embeddings.
    """
    # OOM guard
    async with _enroll_jobs_lock:
        active = sum(1 for j in _enroll_jobs.values() if not j.is_terminal())
        if active >= MAX_CONCURRENT_ENROLLMENTS:
            raise HTTPException(
                status_code=429,
                detail=f"Max concurrent enrollment jobs ({MAX_CONCURRENT_ENROLLMENTS}) reached.",
            )

    # Ensure model loaded
    await _get_model()

    job_id = str(uuid.uuid4())
    job    = EnrollJob(job_id, req.class_name, req.roll_no, req.name)

    async with _enroll_jobs_lock:
        _enroll_jobs[job_id] = job

    # Launch background task
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_enroll_job_sync, job_id, loop)

    return {
        "job_id":      job_id,
        "class_name":  req.class_name,
        "roll_no":     req.roll_no,
        "name":        req.name,
        "status_url":  f"/enroll/status/{job_id}",
        "trigger_url": f"/enroll/{job_id}/trigger",
        "cancel_url":  f"/enroll/{job_id}",
    }


def _run_enroll_job_sync(job_id: str, loop) -> None:
    """
    Synchronous enrollment worker (runs in executor thread).
    Replicates the state machine from Enroll_student.run() without any
    interactive input — quality updates pushed via SSE; angle capture
    gated on trigger_event (set by POST /enroll/{job_id}/trigger).
    """
    job = _enroll_jobs.get(job_id)
    if job is None:
        return

    def push(event, **kw):
        job.push_event(event, kw)

    try:
        # ── Open camera ───────────────────────────────────────────────────
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not cap.isOpened():
            job.state   = "error"
            job.message = "Cannot open camera"
            push("error", message=job.message)
            return

        # ── Validate roll_no doesn't already exist ────────────────────────
        conn = enroll_db_connect()
        try:
            cl_id = db_get_or_create_class(conn, job.class_name)
            if db_roll_exists(conn, cl_id, job.roll_no):
                job.state   = "error"
                job.message = f"Roll '{job.roll_no}' already exists in {job.class_name}"
                push("error", message=job.message)
                return

            # ── Load all students + build FAISS indexes for dup check ─────
            all_students  = db_load_all_students(conn)
            id_to_record  = {r["student_id"]: r for r in all_students}
            class_indexes = build_indexes_from_mysql(all_students)
        finally:
            pass  # keep conn open — needed later for INSERT

        job.state = "waiting"
        push("state_change", state="waiting",
             message="Camera open — waiting for quality face")

        last_scan = time.time() - CAPTURE_INTERVAL

        # ── Main capture loop (replaces interactive while True in run()) ──
        while not job.cancel_event.is_set() and not job.is_terminal():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            now      = time.time()
            countdown = max(0.0, CAPTURE_INTERVAL - (now - last_scan))
            faces    = []
            model    = _insight_app
            if model:
                try:
                    faces = model.get(frame)
                except Exception as e:
                    print(f"[ENROLL:{job_id[:8]}] {e}")

            best_face, best_score, best_reason = get_best_face(faces, frame)
            job.quality = float(best_score)
            job.reason  = best_reason

            push("quality_update",
                 quality=round(best_score, 1),
                 reason=best_reason,
                 countdown=round(countdown, 1),
                 angle_done=job.angle_done,
                 angles_needed=ANGLES_NEEDED,
                 prompt=ANGLE_PROMPTS[job.angle_done]
                         if job.angle_done < len(ANGLE_PROMPTS)
                         else f"Angle {job.angle_done+1}")

            # ── Capture gate ──────────────────────────────────────────────
            if countdown <= 0.0 and best_face is not None and best_score >= 75.0:
                # Wait for user's trigger (replaces SPACE key)
                job.trigger_event.clear()
                job.state = "capturing"
                push("state_change", state="capturing",
                     message=f"Face ready — send POST /enroll/{job_id}/trigger to capture")
                triggered = job.trigger_event.wait(timeout=30.0)  # 30s to press trigger

                if not triggered or job.cancel_event.is_set():
                    last_scan = time.time()
                    job.state = "waiting"
                    continue

                # Re-read frame at capture moment
                ret2, frame2 = cap.read()
                if not ret2:
                    last_scan = time.time()
                    continue
                faces2 = []
                if model:
                    try:
                        faces2 = model.get(frame2)
                    except Exception:
                        pass
                bf2, bs2, br2 = get_best_face(faces2, frame2)

                if bf2 is None or bs2 < 75.0:
                    push("capture_failed",
                         reason=br2 or f"Quality {bs2:.0f}% too low — retry")
                    last_scan = time.time()
                    job.state = "waiting"
                    continue

                emb = bf2.embedding
                if emb is None:
                    push("capture_failed", reason="No embedding — retry")
                    last_scan = time.time()
                    job.state = "waiting"
                    continue

                norm = float(np.linalg.norm(emb))
                if norm < NORM_MIN:
                    push("capture_failed",
                         reason=f"Embedding quality low (norm={norm:.1f}) — full face needed")
                    last_scan = time.time()
                    job.state = "waiting"
                    continue

                job.embs.append(emb.copy())
                # save face crop
                h2, w2 = frame2.shape[:2]
                b2     = bf2.bbox
                x1, y1, x2, y2 = (max(0,int(b2[0])), max(0,int(b2[1])),
                                   min(w2,int(b2[2])), min(h2,int(b2[3])))
                job.crops.append(frame2[y1:y2, x1:x2].copy())

                job.angle_done += 1
                push("angle_captured",
                     angle=job.angle_done,
                     angles_needed=ANGLES_NEEDED,
                     norm=round(norm, 2),
                     quality=round(bs2, 1))

                last_scan = time.time()

                if job.angle_done >= ANGLES_NEEDED:
                    break   # all angles captured

                job.state = "waiting"
                push("state_change", state="waiting",
                     message=f"Angle {job.angle_done}/{ANGLES_NEEDED} done "
                             f"— next: {ANGLE_PROMPTS[job.angle_done] if job.angle_done < len(ANGLE_PROMPTS) else 'Angle '+str(job.angle_done+1)}")

            time.sleep(0.05)   # ~20 fps polling

        cap.release()

        if job.cancel_event.is_set():
            job.state = "cancelled"
            push("state_change", state="cancelled", message="Enrollment cancelled")
            conn.close()
            return

        if len(job.embs) < ANGLES_NEEDED:
            job.state   = "error"
            job.message = "Not enough angles captured"
            push("error", message=job.message)
            conn.close()
            return

        # ── FAISS duplicate check ──────────────────────────────────────────
        job.state = "checking"
        push("state_change", state="checking", message="Running FAISS duplicate check …")

        matched, sid, dup_name, avg_sim, _ = find_match_faiss(
            job.embs, class_indexes, id_to_record)

        if matched:
            job.state   = "duplicate"
            job.message = f"Already enrolled: {dup_name} (sim={avg_sim:.3f})"
            push("duplicate", name=dup_name, sim=round(avg_sim, 4),
                 message=job.message)
            conn.close()
            return

        # ── Insert into MySQL + FAISS ──────────────────────────────────────
        job.state = "enrolling"
        push("state_change", state="enrolling", message="Saving to MySQL + FAISS …")

        cl_id2     = db_get_or_create_class(conn, job.class_name)
        student_id = db_insert_student(
            conn, cl_id2, job.name, job.roll_no, job.embs, "pending")

        save_face_images(student_id, job.crops)
        photo_path = os.path.join(IMAGES_DIR, str(student_id))
        cur = conn.cursor()
        cur.execute("UPDATE students SET photo_path=%s WHERE id=%s",
                    (photo_path, student_id))
        conn.commit()
        cur.close()
        conn.close()

        if job.class_name not in class_indexes:
            import faiss as _faiss
            base = _faiss.IndexFlatIP(512)
            class_indexes[job.class_name] = _faiss.IndexIDMap(base)
        faiss_add(class_indexes[job.class_name], job.embs, student_id)
        faiss_save(class_indexes[job.class_name], job.class_name)

        job.state      = "done"
        job.student_id = student_id
        job.message    = (f"Enrolled {job.name} → id={student_id}, "
                          f"class={job.class_name}, roll={job.roll_no}")
        job.finished_at = time.time()
        push("enrolled",
             student_id=student_id,
             name=job.name,
             class_name=job.class_name,
             roll_no=job.roll_no,
             message=job.message)

    except Exception as e:
        job.state   = "error"
        job.message = str(e)
        job.push_event("error", {"message": str(e),
                                "traceback": traceback.format_exc()})
        print(f"[ENROLL:{job_id[:8]}] Error: {e}\n{traceback.format_exc()}")
    finally:
        job.finished_at = job.finished_at or time.time()


@app.get("/enroll/status/{job_id}", tags=["enroll"])
async def enroll_status_stream(job_id: str):
    """
    SSE stream for enrollment job events.
    Events: quality_update, state_change, angle_captured, capture_failed,
            duplicate, enrolled, error, cancelled
    """
    job = _enroll_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def generate():
        idle_since = time.time()
        while True:
            try:
                payload = await asyncio.wait_for(job.sse_queue.get(), timeout=2.0)
                idle_since = time.time()
                data = _json.dumps(payload)
                yield f"data: {data}\n\n"
                # Stop streaming once terminal
                if job.is_terminal():
                    yield f"data: {json.dumps({'event':'stream_end'})}\n\n"
                    break
            except asyncio.TimeoutError:
                if time.time() - idle_since > SSE_IDLE_TIMEOUT_S:
                    break
                yield ": keepalive\n\n"   # SSE comment keeps connection alive

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/enroll/{job_id}/trigger", tags=["enroll"])
async def enroll_trigger(job_id: str):
    """
    Trigger capture of the next angle (replaces SPACE key from terminal mode).
    Call this once the user is in position and quality_update shows quality >= 75.
    """
    job = _enroll_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.is_terminal():
        raise HTTPException(status_code=400, detail=f"Job is {job.state}")
    job.trigger_event.set()
    return {
        "job_id":      job_id,
        "angle_done":  job.angle_done,
        "state":       job.state,
        "message":     "Trigger signal sent — capture will execute on next frame",
    }


@app.delete("/enroll/{job_id}", tags=["enroll"])
async def enroll_cancel(job_id: str):
    """Cancel an in-progress enrollment job."""
    job = _enroll_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    job.cancel_event.set()
    return {"job_id": job_id, "status": "cancellation requested"}


@app.get("/enroll/{job_id}", tags=["enroll"])
async def enroll_job_info(job_id: str):
    """Get current state of an enrollment job (non-streaming)."""
    job = _enroll_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id":      job_id,
        "class_name":  job.class_name,
        "roll_no":     job.roll_no,
        "name":        job.name,
        "state":       job.state,
        "angle_done":  job.angle_done,
        "angles_needed": ANGLES_NEEDED,
        "quality":     job.quality,
        "reason":      job.reason,
        "message":     job.message,
        "student_id":  job.student_id,
    }
