"""
routers/attendance.py — Live attendance router.
Routes: GET /, POST /attendance/session/start, stream, status, stop, delete.
All HTML is served from frontend/index.html (no embedded strings).
"""
import asyncio, os, queue as _queue, threading, time, uuid
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import cv2, numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from auth_deps import get_current_user, db_get_teacher_class_ids
from fastapi import Depends

from live_attendance import (
    db_load_class_students, db_get_todays_session, db_get_already_present,
    db_create_session, db_close_session, db_mark_absent_bulk, db_writer_thread,
    build_class_index, faiss_batch_match, compute_adaptive_thresholds,
    live_quality, quality_weighted_avg, l2_norm, get_bbox, has_motion,
    draw_hud, beep_present, CameraThread,
    CAMERA_INDEX, WRITE_QUEUE_MAX, SMOOTH_BUF_MAX, PRESENT_LOG_MAX,
    FRAME_SKIP, CONFIRM_FRAMES, DEDUP_SECONDS, SMOOTH_WINDOW, QUALITY_WEIGHT,
)

router = APIRouter()

# ── constants ─────────────────────────────────────────────────────────────────
MAX_CONCURRENT_SESSIONS   = 1
WRITE_QUEUE_DRAIN_TIMEOUT = 3.0
SSE_IDLE_TIMEOUT_S        = 120
MJPEG_FRAME_QUEUE_SIZE    = 1          # keep only latest frame; never queue stale
MJPEG_QUALITY             = 50         # 50 encodes ~2× faster than 70 with acceptable quality
MJPEG_STREAM_WIDTH        = 640        # resize frame before encode to further reduce payload

# ── shared state (no asyncio.Lock at module level!) ───────────────────────────
_insight_app = None
_sessions: Dict[str, Any] = {}
_sessions_lock: Optional[asyncio.Lock] = None  # lazily created inside event loop

def _get_lock() -> asyncio.Lock:
    global _sessions_lock
    if _sessions_lock is None:
        _sessions_lock = asyncio.Lock()
    return _sessions_lock

def set_model(model) -> None:
    global _insight_app; _insight_app = model

def get_sessions():      return _sessions
def get_sessions_lock(): return _get_lock()

# ── frontend path ─────────────────────────────────────────────────────────────
_FRONTEND = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "frontend")
)

def _serve(fname: str) -> HTMLResponse:
    p = os.path.join(_FRONTEND, fname)
    try:
        return HTMLResponse(open(p, encoding="utf-8").read())
    except FileNotFoundError:
        return HTMLResponse(f"<h2>Missing: {fname}</h2>", status_code=404)


# ── AttendanceSession ─────────────────────────────────────────────────────────
class AttendanceSession:
    def __init__(self, session_id, class_name, teacher_name, students,
                 class_id, already_present, db_session_id):
        self.session_id    = session_id
        self.class_name    = class_name
        self.teacher_name  = teacher_name
        self.students      = students
        self.class_id      = class_id
        self.db_session_id = db_session_id
        self.id_to_student = {s["student_id"]: s for s in students}
        self.all_ids: Set[int]       = {s["student_id"] for s in students}
        self.thresholds              = compute_adaptive_thresholds(students)
        self.index                   = build_class_index(students)
        self.write_queue             = _queue.Queue(maxsize=WRITE_QUEUE_MAX)
        self.present_ids: Set[int]   = set(already_present)
        self._writer_thread = threading.Thread(
            target=db_writer_thread, args=(self.write_queue, self.present_ids), daemon=True)
        self._writer_thread.start()
        self.present_log: List[Dict] = []; self.last_marked: Dict[int,float] = {}
        self.confirm_buf: Dict[int,int] = {}; self.smooth_buf: Dict = {}
        self.last_name: Dict[int,str] = {}; self.last_sim: Dict[int,float] = {}
        self.last_boxes: List = []; self.prev_gray = None
        self.frame_count = 0; self.session_start = time.time()
        self.fps = 0.0; self.fps_counter = 0; self.fps_timer = time.time()
        self.msg = "Waiting..."; self.msg_color = (180,180,180)
        self.active = True
        self._loop = asyncio.get_event_loop()
        self.frame_q: asyncio.Queue = asyncio.Queue(maxsize=MJPEG_FRAME_QUEUE_SIZE)
        self.cam = None; self._proc_thread = None
        self._cam_started = False; self._cam_lock = threading.Lock()

    def start_camera(self):
        with self._cam_lock:
            if self._cam_started: return
            self._cam_started = True
            self.cam = CameraThread(CAMERA_INDEX); self.cam.start()
            self._proc_thread = threading.Thread(target=self._process_loop, daemon=True)
            self._proc_thread.start()

    def _process_loop(self):
        try:
            while self.active:
                if self.cam is None: time.sleep(0.05); continue
                frame = self.cam.get_frame()
                if frame is None: time.sleep(0.01); continue
                self._process_frame(frame)
        except Exception as e:
            print(f"[SESSION:{self.session_id[:8]}] {e}")
        finally:
            if self.cam: self.cam.stop()

    def _process_frame(self, frame):
        if self.frame_count == 0:
            self.session_start = time.time()
        self.frame_count += 1; self.fps_counter += 1
        now_t = time.time()
        if now_t - self.fps_timer >= 1.0:
            self.fps = self.fps_counter/(now_t-self.fps_timer)
            self.fps_counter = 0; self.fps_timer = now_t
        display = frame.copy(); gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.frame_count % FRAME_SKIP != 0:
            for (bx1,by1,bx2,by2,bc,bl) in self.last_boxes:
                cv2.rectangle(display,(bx1,by1),(bx2,by2),bc,2)
                cv2.putText(display,bl,(bx1,max(by1-8,55)),cv2.FONT_HERSHEY_SIMPLEX,.52,bc,2,cv2.LINE_AA)
            draw_hud(display,self.class_name,self.present_log,len(self.students),
                     self.session_start,self.fps,gray,self.msg,self.msg_color)
            self._push_frame(display); self.prev_gray=gray; return
        motion = has_motion(gray, self.prev_gray); self.prev_gray = gray
        model = _insight_app
        # Warmup delay: don't process faces for the first 3 seconds while UI loads
        if model is None or (time.time() - self.session_start < 3.0): 
            if time.time() - self.session_start < 3.0:
                self.msg = "Warming up camera..."; self.msg_color = (180,180,180)
            self._push_frame(display)
            return
        
        try: faces = model.get(frame)
        except Exception as e: faces=[]; print(f"[WARN] {e}")
        if not faces:
            self.last_boxes=[]
            for k in list(self.confirm_buf):
                self.confirm_buf[k]-=1
                if self.confirm_buf[k]<=0: del self.confirm_buf[k]
        else:
            scored=[(f,*live_quality(f,gray)) for f in faces]
            good=[(f,s,r) for f,s,r in scored if f.embedding is not None and s>=55.0]
            if good and motion:
                raw=faiss_batch_match([f.embedding for f,_,_ in good],self.index,self.id_to_student,self.thresholds)
                s_embs=[]; s_meta=[]
                for (face,score,reason),(_,sid,_,_) in zip(good,raw):
                    if sid==-1: s_embs.append(l2_norm(face.embedding.copy())); s_meta.append((face,score,reason)); continue
                    if sid not in self.smooth_buf and len(self.smooth_buf)>=SMOOTH_BUF_MAX:
                        del self.smooth_buf[next(iter(self.smooth_buf))]
                    if sid not in self.smooth_buf: self.smooth_buf[sid]=deque(maxlen=SMOOTH_WINDOW)
                    self.smooth_buf[sid].append((face.embedding.copy(),score))
                    be=[e for e,_ in self.smooth_buf[sid]]
                    bw=[w for _,w in self.smooth_buf[sid]] if QUALITY_WEIGHT else [1.0]*len(be)
                    s_embs.append(quality_weighted_avg(be,bw)); s_meta.append((face,score,reason))
                final=faiss_batch_match(s_embs,self.index,self.id_to_student,self.thresholds)
                now=time.time(); self.last_boxes=[]
                for (face,score,reason),(matched,sid,name,avg_sim) in zip(s_meta,final):
                    x1,y1,x2,y2=get_bbox(face)
                    if matched and sid!=-1:
                        self.last_name[sid]=name; self.last_sim[sid]=avg_sim
                        self.confirm_buf[sid]=self.confirm_buf.get(sid,0)+1
                        for other in list(self.confirm_buf):
                            if other!=sid:
                                self.confirm_buf[other]-=1
                                if self.confirm_buf[other]<=0: del self.confirm_buf[other]
                        am=sid in self.present_ids; bc=(0,210,80) if am else (0,180,255)
                        lbl=("✓ " if am else "")+f"{name}  {avg_sim:.2f}"
                        cv2.rectangle(display,(x1,y1),(x2,y2),bc,2)
                        cv2.putText(display,lbl,(x1,max(y1-8,55)),cv2.FONT_HERSHEY_SIMPLEX,.52,bc,2,cv2.LINE_AA)
                        self.last_boxes.append((x1,y1,x2,y2,bc,lbl))
                        if not am:
                            bw2=max(x2-x1,1); pct=self.confirm_buf.get(sid,0)/CONFIRM_FRAMES
                            by=max(y1-18,58); filled=min(int(pct*bw2),bw2)
                            cv2.rectangle(display,(x1,by),(x2,by+5),(40,40,40),-1)
                            if filled>0: cv2.rectangle(display,(x1,by),(x1+filled,by+5),(0,210,80),-1)
                        if (self.confirm_buf.get(sid,0)>=CONFIRM_FRAMES and not am
                                and (now-self.last_marked.get(sid,0))>DEDUP_SECONDS):
                            try: self.write_queue.put_nowait({"type":"present","student_id":sid,"class_id":self.class_id,"session_id":self.db_session_id,"confidence":avg_sim})
                            except _queue.Full: continue
                            self.last_marked[sid]=now
                            if len(self.present_log)>=PRESENT_LOG_MAX: self.present_log.pop(0)
                            self.present_log.append({"student_id":sid,"name":name,"sim":avg_sim,"time":datetime.now().strftime("%H:%M:%S")})
                            self.msg=f"Present: {name}"; self.msg_color=(0,220,80); beep_present()
                    else:
                        for k in list(self.confirm_buf):
                            self.confirm_buf[k]-=1
                            if self.confirm_buf[k]<=0: del self.confirm_buf[k]
                        lbl=f"Unknown {avg_sim:.2f}"
                        cv2.rectangle(display,(x1,y1),(x2,y2),(40,40,220),2)
                        cv2.putText(display,lbl,(x1,max(y1-8,55)),cv2.FONT_HERSHEY_SIMPLEX,.44,(40,80,220),1,cv2.LINE_AA)
                        self.last_boxes.append((x1,y1,x2,y2,(40,40,220),lbl))
            elif good and not motion:
                for face,score,reason in good:
                    x1,y1,x2,y2=get_bbox(face)
                    bsid=max(self.smooth_buf,key=lambda s:sum(w for _,w in self.smooth_buf[s])/len(self.smooth_buf[s])) if self.smooth_buf else -1
                    if bsid!=-1:
                        n=self.last_name.get(bsid,""); sim=self.last_sim.get(bsid,0.0)
                        col=(0,210,80) if bsid in self.present_ids else (0,180,255)
                        cv2.rectangle(display,(x1,y1),(x2,y2),col,2)
                        cv2.putText(display,("✓ " if bsid in self.present_ids else "")+f"{n}  {sim:.2f}",(x1,max(y1-8,55)),cv2.FONT_HERSHEY_SIMPLEX,.52,col,2,cv2.LINE_AA)
                    else: cv2.rectangle(display,(x1,y1),(x2,y2),(100,100,100),1)
            for face,score,reason in scored:
                if score<55.0:
                    x1,y1,x2,y2=get_bbox(face)
                    cv2.rectangle(display,(x1,y1),(x2,y2),(60,60,60),1)
                    cv2.putText(display,f"{reason} {score:.0f}%",(x1,max(y1-8,55)),cv2.FONT_HERSHEY_SIMPLEX,.34,(120,120,120),1,cv2.LINE_AA)
        draw_hud(display,self.class_name,self.present_log,len(self.students),
                 self.session_start,self.fps,gray,self.msg,self.msg_color)
        self._push_frame(display)

    def _push_frame(self, frame):
        # Resize to stream width to reduce encode cost and HTTP payload
        h, w = frame.shape[:2]
        if w > MJPEG_STREAM_WIDTH:
            scale = MJPEG_STREAM_WIDTH / w
            frame = cv2.resize(frame, (MJPEG_STREAM_WIDTH, int(h * scale)), interpolation=cv2.INTER_LINEAR)
        ok,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,MJPEG_QUALITY])
        if not ok: return
        jpg=buf.tobytes()
        async def _put():
            # Always drop stale frames – only keep the very latest
            while not self.frame_q.empty():
                try: self.frame_q.get_nowait()
                except: break
            try: self.frame_q.put_nowait(jpg)
            except: pass
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(_put(), self._loop)

    def get_status(self):
        elapsed=int(time.time()-self.session_start); m,s=divmod(elapsed,60)
        return {"session_id":self.session_id,"class_name":self.class_name,
                "teacher_name":self.teacher_name,"elapsed":f"{m:02d}:{s:02d}",
                "fps":round(self.fps,1),"total":len(self.students),
                "present":len(self.present_ids),"absent":len(self.all_ids-self.present_ids),
                "present_log":self.present_log[-20:]}

    def stop(self): self.active = False


# ── Routes ────────────────────────────────────────────────────────────────────
@router.get("/", tags=["ui"], response_class=HTMLResponse)
async def ui_home():
    return _serve("index.html")


class StartSessionRequest(BaseModel):
    class_name:   str
    teacher_name: str
    resume:       bool = False
    teacher_id:   Optional[int] = None   # set by frontend after login
    school_name:  Optional[str] = None   # populated from user JWT


@router.post("/attendance/session/start", tags=["attendance"])
async def start_session(req: StartSessionRequest, user: Dict = Depends(get_current_user)):
    req.teacher_id = user["id"]
    req.school_name = user.get("school_name", "Default School")
    lock = _get_lock()
    loop = asyncio.get_event_loop()
    def _setup():
        # Prepend school folder to class name for FAISS/DB lookups
        class_path = f"{req.school_name}/{req.class_name}"
        # Note: class_name is currently global in DB, but we filter by school_id
        students = db_load_class_students(req.class_name, user["school_id"]) 
        if not students: raise ValueError(f"No students in '{req.class_name}'")
        class_id = students[0]["class_id"]
        
        # Verify RBAC...
        
        existing = db_get_todays_session(class_id, user["school_id"]) if req.resume else None
        if existing and req.resume:
            db_sid = str(existing["id"]); already_present = db_get_already_present(db_sid)
        else:
            db_sid = str(uuid.uuid4()); already_present = set()
            db_create_session(db_sid, class_id, req.teacher_name, user["school_id"])
        return students, class_id, db_sid, already_present
    try:
        students, class_id, db_sid, already_present = await asyncio.wait_for(
            loop.run_in_executor(None, _setup), timeout=30.0)
    except ValueError as e: raise HTTPException(status_code=409, detail=str(e))
    except asyncio.TimeoutError: raise HTTPException(status_code=504, detail="DB timeout")
    session_id = str(uuid.uuid4())
    sess = AttendanceSession(session_id, req.class_name, req.teacher_name,
                             students, class_id, already_present, db_sid)
    async with lock: _sessions[session_id] = sess
    return {"session_id":session_id,"class_name":req.class_name,"teacher_name":req.teacher_name,
            "students":len(students),"db_session_id":db_sid,"resumed":bool(already_present),
            "stream_url":f"/attendance/session/{session_id}/stream",
            "status_url":f"/attendance/session/{session_id}/status",
            "stop_url":f"/attendance/session/{session_id}/stop"}


@router.get("/attendance/session/{session_id}/stream", tags=["attendance"])
async def stream_session(session_id: str, user: Dict = Depends(get_current_user)):
    lock = _get_lock()
    async with lock: sess = _sessions.get(session_id)
    if sess is None: raise HTTPException(status_code=404, detail="Session not found")
    
    if user["role"] != "admin":
        if sess.class_id not in db_get_teacher_class_ids(user["id"]):
            raise HTTPException(status_code=403, detail="Not assigned to this class")
    await asyncio.get_event_loop().run_in_executor(None, sess.start_camera)
    async def generate():
        idle = time.time()
        while sess.active:
            try:
                jpg = await asyncio.wait_for(sess.frame_q.get(), timeout=2.0); idle=time.time()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+jpg+b"\r\n"
            except asyncio.TimeoutError:
                if time.time()-idle>SSE_IDLE_TIMEOUT_S: break
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@router.get("/attendance/session/{session_id}/status", tags=["attendance"])
async def session_status(session_id: str, user: Dict = Depends(get_current_user)):
    lock = _get_lock()
    async with lock: sess = _sessions.get(session_id)
    if sess is None: raise HTTPException(status_code=404, detail="Session not found")
    
    if user["role"] != "admin":
        if sess.class_id not in db_get_teacher_class_ids(user["id"]):
            raise HTTPException(status_code=403, detail="Not assigned to this class")
    return sess.get_status()


class StopSessionRequest(BaseModel):
    confirm: bool = True


@router.post("/attendance/session/{session_id}/stop", tags=["attendance"])
async def stop_session(session_id: str, req: StopSessionRequest = StopSessionRequest(), user: Dict = Depends(get_current_user)):
    lock = _get_lock()
    async with lock: sess = _sessions.get(session_id)
    if sess is None: raise HTTPException(status_code=404, detail="Session not found")
    
    if user["role"] != "admin":
        if sess.class_id not in db_get_teacher_class_ids(user["id"]):
            raise HTTPException(status_code=403, detail="Not assigned to this class")
    if not req.confirm:
        return {"preview":True,"present":len(sess.present_ids),"absent":len(sess.all_ids-sess.present_ids)}
    sess.stop()
    loop = asyncio.get_event_loop()
    def _drain():
        try: sess.write_queue.put_nowait(None)
        except _queue.Full: pass
        deadline=time.time()+WRITE_QUEUE_DRAIN_TIMEOUT
        while not sess.write_queue.empty() and time.time()<deadline: time.sleep(0.05)
        absent_ids=sess.all_ids-sess.present_ids
        n=db_mark_absent_bulk(list(absent_ids),sess.class_id,sess.db_session_id)
        db_close_session(sess.db_session_id,len(sess.present_ids),n)
        return n, absent_ids
    try: n_absent, absent_ids = await asyncio.wait_for(loop.run_in_executor(None,_drain), timeout=8.0)
    except asyncio.TimeoutError:
        absent_ids=sess.all_ids-sess.present_ids; n_absent=len(absent_ids)
    async with lock: _sessions.pop(session_id, None)
    elapsed=int(time.time()-sess.session_start); m,s=divmod(elapsed,60)
    return {"session_id":session_id,"class_name":sess.class_name,"teacher":sess.teacher_name,
            "duration":f"{m:02d}:{s:02d}","present":len(sess.present_ids),"absent":n_absent,
            "present_log":sess.present_log,
            "absent_log":[{"student_id":sid,"name":sess.id_to_student[sid]["name"],
                           "roll_no":sess.id_to_student[sid]["roll_no"]}
                          for sid in sorted(absent_ids) if sid in sess.id_to_student]}


@router.delete("/attendance/session/{session_id}", tags=["attendance"])
async def quit_session(session_id: str, user: Dict = Depends(get_current_user)):
    lock = _get_lock()
    async with lock: sess = _sessions.pop(session_id, None)
    if sess is None: raise HTTPException(status_code=404, detail="Session not found")
    
    if user["role"] != "admin":
        if sess.class_id not in db_get_teacher_class_ids(user["id"]):
            raise HTTPException(status_code=403, detail="Not assigned to this class")
    sess.stop(); sess.write_queue.put(None)
    return {"session_id":session_id,"status":"quit — not saved"}
