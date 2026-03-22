"""
Live Attendance  —  High Performance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Detection box shown EVERY frame (student always visible)
- FAISS matching runs only when motion detected (saves CPU)
- Temporal smoothing: 5-frame quality-weighted average
- Adaptive per-student threshold
- Batch FAISS for multiple faces
- Bounded write queue, smooth_buf cap, sid=-1 guard (OOM safe)
- Camera thread, async DB writer, motion gate, frame skip
- CTRL+C graceful shutdown, session resume
"""

import cv2
import os
import sys
import uuid
import time
import warnings
import threading
import queue
import signal
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Protocol, runtime_checkable
from collections import deque

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("[ERROR] pip install insightface onnxruntime"); sys.exit(1)

try:
    import mysql.connector
except ImportError:
    print("[ERROR] pip install mysql-connector-python"); sys.exit(1)

try:
    import faiss
except ImportError:
    print("[ERROR] pip install faiss-cpu"); sys.exit(1)

try:
    import db as DB
except ImportError:
    print("[ERROR] db.py not found in scripts/ folder"); sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_INDEX  = 0
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
MODEL_NAME    = "buffalo_sc"
EMBEDDING_DIM = 512

DET_SCORE_MIN  = 0.55
NORM_MIN_LIVE  = 14.0
YAW_MAX_LIVE   = 45.0
PITCH_MAX_LIVE = 35.0
BLUR_MIN_LIVE  = 50.0
BRIGHTNESS_MIN = 50
BRIGHTNESS_MAX = 220

BASE_SIMILARITY_THRESHOLD = 0.45
THRESHOLD_ADAPT_RANGE     = 0.08

SMOOTH_WINDOW  = 5
QUALITY_WEIGHT = True

FRAME_SKIP       = 2
MOTION_THRESHOLD = 800

CONFIRM_FRAMES = 1
DEDUP_SECONDS  = 10

WRITE_QUEUE_MAX = 50
SMOOTH_BUF_MAX  = 20
PRESENT_LOG_MAX = 500


# ─────────────────────────────────────────────────────────────────────────────
#  SOUND
# ─────────────────────────────────────────────────────────────────────────────

def beep_present() -> None:
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1000, 120)
            time.sleep(0.05)
            winsound.Beep(1200, 120)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def blob_to_emb(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def l2_norm(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb)
    return emb / n if n > 0 else emb


def quality_weighted_avg(embs: List[np.ndarray],
                          weights: List[float]) -> np.ndarray:
    if not embs:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    if len(embs) == 1:
        return l2_norm(embs[0])
    mat     = np.stack(embs)
    w       = np.array(weights, dtype=np.float32)
    total_w = w.sum()
    avg     = mat.mean(axis=0) if total_w == 0 \
              else (mat * w[:, None]).sum(axis=0) / total_w
    return l2_norm(avg)


# ─────────────────────────────────────────────────────────────────────────────
#  ADAPTIVE THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────

def compute_adaptive_thresholds(students: List[Dict]) -> Dict[int, float]:
    thresholds: Dict[int, float] = {}
    for s in students:
        embs = s["embeddings"]
        sid  = s["student_id"]
        if len(embs) < 2:
            thresholds[sid] = BASE_SIMILARITY_THRESHOLD
            continue
        normed = [l2_norm(e) for e in embs]
        sims   = [float(np.dot(normed[i], normed[j]))
                  for i in range(len(normed))
                  for j in range(i + 1, len(normed))]
        consistency     = float(np.mean(sims)) if sims else 0.5
        norm_c          = max(0.0, min(1.0, (consistency - 0.3) / 0.6))
        adjustment      = (norm_c - 0.5) * 2 * THRESHOLD_ADAPT_RANGE
        thresholds[sid] = round(BASE_SIMILARITY_THRESHOLD + adjustment, 4)
    return thresholds


# ─────────────────────────────────────────────────────────────────────────────
#  MYSQL LAYER
# ─────────────────────────────────────────────────────────────────────────────

def db_load_class_students(class_name: str, school_id: int = 1) -> List[Dict]:
    conn = DB.get_conn()
    try:
        rows = DB.execute(conn,
            """SELECT s.id AS student_id, s.name, s.roll_no,
                      s.emb_1, s.emb_2, s.emb_3,
                      c.id AS class_id, c.name AS class_name
               FROM   students s
               JOIN   classes  c ON c.id = s.class_id
               WHERE  s.school_id = %s AND c.name = %s
               ORDER  BY s.id""",
            (school_id, class_name), fetch=True)
    finally:
        conn.close()
    students = []
    for row in rows:
        embs = []
        for key in ("emb_1", "emb_2", "emb_3"):
            if row[key]:
                embs.append(blob_to_emb(row[key]))
        if embs:
            students.append({
                "student_id": int(row["student_id"]),
                "name":       str(row["name"]),
                "roll_no":    str(row["roll_no"]),
                "class_id":   int(row["class_id"]),
                "class_name": str(row["class_name"]),
                "embeddings": embs,
            })
    return students


def db_list_classes() -> List[str]:
    conn = DB.get_conn()
    try:
        rows = DB.execute(conn,
            "SELECT name FROM classes WHERE school_id=%s ORDER BY name",
            (DB.SCHOOL_ID,), fetch=True)
        return [r["name"] for r in rows]
    finally:
        conn.close()


def db_get_todays_session(class_id: int) -> Optional[Dict]:
    conn = DB.get_conn()
    try:
        rows = DB.execute(conn,
            """SELECT id, teacher_name, started_at, total_present
               FROM   sessions
               WHERE  class_id=%s AND school_id=%s
                 AND  DATE(started_at) = CURDATE()
                 AND  status = 'active'
               LIMIT 1""",
            (class_id, DB.SCHOOL_ID), fetch=True)
        return dict(rows[0]) if rows else None
    finally:
        conn.close()


def db_get_already_present(session_id: str) -> Set[int]:
    conn = DB.get_conn()
    try:
        rows = DB.execute(conn,
            "SELECT student_id FROM attendance "
            "WHERE session_id=%s AND status='present'",
            (session_id,), fetch=True)
        return {int(r["student_id"]) for r in rows}
    finally:
        conn.close()


def db_create_session(session_id: str, class_id: int,
                      teacher_name: str, school_id: int = 1) -> None:
    conn = DB.get_conn()
    try:
        DB.execute(conn,
            """INSERT INTO sessions
               (id, school_id, class_id, teacher_name, status)
               VALUES (%s, %s, %s, %s, 'active')""",
            (session_id, school_id, class_id, teacher_name))
    finally:
        conn.close()


def db_close_session(session_id: str,
                     n_present: int, n_absent: int) -> None:
    conn = DB.get_conn()
    try:
        DB.execute(conn,
            """UPDATE sessions
               SET ended_at=NOW(), status='completed',
                   total_present=%s, total_absent=%s
               WHERE id=%s""",
            (n_present, n_absent, session_id))
    finally:
        conn.close()


def db_mark_absent_bulk(student_ids: List[int],
                        class_id: int,
                        session_id: str) -> int:
    if not student_ids:
        return 0
    conn  = DB.get_conn()
    count = 0
    try:
        for sid in student_ids:
            try:
                DB.execute(conn,
                    """INSERT INTO attendance
                       (student_id, class_id, school_id, date, status,
                        confidence_score, session_id)
                       VALUES (%s, %s, %s, CURDATE(), 'absent', NULL, %s)""",
                    (sid, class_id, DB.SCHOOL_ID, session_id))
                count += 1
            except mysql.connector.IntegrityError:
                pass
    finally:
        conn.close()
    return count


# ─────────────────────────────────────────────────────────────────────────────
#  ASYNC DB WRITER
# ─────────────────────────────────────────────────────────────────────────────

def db_writer_thread(wq: queue.Queue, present_ids: Set[int]) -> None:
    conn = DB.get_conn()
    while True:
        try:
            job = wq.get(timeout=1.0)
        except queue.Empty:
            continue
        if job is None:
            conn.close()
            wq.task_done()
            break
        if job["type"] == "present":
            for attempt in range(3):
                try:
                    cur = conn.cursor()
                    cur.execute(
                        """INSERT INTO attendance
                           (student_id, class_id, school_id, date,
                            status, confidence_score, session_id)
                           VALUES (%s, %s, %s, CURDATE(), 'present', %s, %s)""",
                        (job["student_id"], job["class_id"], DB.SCHOOL_ID,
                         round(job["confidence"], 4), job["session_id"]),
                    )
                    conn.commit()
                    cur.close()
                    present_ids.add(job["student_id"])
                    break
                except mysql.connector.IntegrityError:
                    present_ids.add(job["student_id"])
                    break
                except mysql.connector.errors.OperationalError:
                    if attempt < 2:
                        conn.reconnect(attempts=3, delay=1)
                    else:
                        print("[DB-WRITER] Failed after 3 attempts")
                except Exception as e:
                    print(f"[DB-WRITER] {e}")
                    break
        wq.task_done()


# ─────────────────────────────────────────────────────────────────────────────
#  CAMERA THREAD
# ─────────────────────────────────────────────────────────────────────────────

class CameraThread(threading.Thread):
    def __init__(self, camera_index: int) -> None:
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.frame: Optional[np.ndarray] = None
        self.lock    = threading.Lock()
        self.running = True
        self._open()

    def _open(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # never hold stale frames in driver

    def run(self) -> None:
        fails = 0
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                fails = 0
                with self.lock:
                    self.frame = frame
            else:
                fails += 1
                if fails > 30:
                    print("[CAM] Attempting recovery...")
                    self.cap.release()
                    time.sleep(2.0)
                    self._open()
                    fails = 0

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self) -> None:
        self.running = False
        self.cap.release()


# ─────────────────────────────────────────────────────────────────────────────
#  FAISS
# ─────────────────────────────────────────────────────────────────────────────

def build_class_index(students: List[Dict]) -> faiss.IndexIDMap:
    base  = faiss.IndexFlatIP(EMBEDDING_DIM)
    index = faiss.IndexIDMap(base)
    for s in students:
        for emb in s["embeddings"]:
            vec = l2_norm(emb).reshape(1, -1).astype(np.float32)
            index.add_with_ids(vec,
                               np.array([s["student_id"]], dtype=np.int64))
    return index


def faiss_batch_match(
    live_embs: List[np.ndarray],
    index: faiss.IndexIDMap,
    id_to_student: Dict[int, Dict],
    thresholds: Dict[int, float],
) -> List[Tuple[bool, int, str, float]]:
    if index.ntotal == 0 or not live_embs:
        return [(False, -1, "", 0.0)] * len(live_embs)
    vecs = np.stack([l2_norm(e).astype(np.float32) for e in live_embs])
    scores_mat, ids_mat = index.search(vecs, index.ntotal)
    results = []
    for scores, ids in zip(scores_mat, ids_mat):
        id_scores: Dict[int, List[float]] = {}
        for score, sid in zip(scores, ids):
            sid = int(sid)
            if sid != -1:
                id_scores.setdefault(sid, []).append(float(score))
        if not id_scores:
            results.append((False, -1, "", 0.0))
            continue
        avg_sc  = {s: sum(v)/len(v) for s, v in id_scores.items()}
        best_id = max(avg_sc, key=lambda x: avg_sc[x])
        best_v  = avg_sc[best_id]
        name    = id_to_student.get(best_id, {}).get("name", "Unknown")
        thresh  = thresholds.get(best_id, BASE_SIMILARITY_THRESHOLD)
        results.append((best_v >= thresh, best_id, name, best_v))
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  FACE QUALITY
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class FaceObject(Protocol):
    bbox:      np.ndarray
    det_score: float
    embedding: Optional[np.ndarray]
    pose:      Optional[np.ndarray]


def live_quality(face: FaceObject,
                 gray: np.ndarray) -> Tuple[float, str]:
    try:
        det   = float(face.det_score)
        emb   = face.embedding
        norm  = float(np.linalg.norm(emb)) if emb is not None else 0.0
        pose  = face.pose
        yaw   = abs(float(pose[1])) if pose is not None and len(pose) > 1 else 0.0
        pitch = abs(float(pose[0])) if pose is not None and len(pose) > 0 else 0.0
    except Exception as e:
        return 0.0, f"Err:{e}"

    if det   < DET_SCORE_MIN:  return det*100,               f"Conf {det:.2f}"
    if norm  < NORM_MIN_LIVE:  return (norm/NORM_MIN_LIVE)*50, f"Norm {norm:.1f}"
    if yaw   > YAW_MAX_LIVE:   return max(10., 100-yaw*2),    f"Yaw {yaw:.0f}"
    if pitch > PITCH_MAX_LIVE: return max(10., 100-pitch*2),  "Pitch"

    mean = float(np.mean(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if mean < BRIGHTNESS_MIN: return 20.0, f"Dark {mean:.0f}"
    if mean > BRIGHTNESS_MAX: return 20.0, f"Bright {mean:.0f}"
    if blur < BLUR_MIN_LIVE:  return 30.0, f"Blur {blur:.0f}"

    det_s  = min(det, 1.0)
    norm_s = min(norm / 25.0, 1.0)
    yaw_s  = max(0.0, 1.0 - yaw / YAW_MAX_LIVE)
    return float(int((det_s*0.4 + norm_s*0.4 + yaw_s*0.2) * 1000) / 10), ""


def get_bbox(face: FaceObject) -> Tuple[int, int, int, int]:
    try:
        b = face.bbox
        return int(b[0]), int(b[1]), int(b[2]), int(b[3])
    except Exception:
        return 0, 0, 0, 0


# ─────────────────────────────────────────────────────────────────────────────
#  MOTION
# ─────────────────────────────────────────────────────────────────────────────

def has_motion(gray: np.ndarray,
               prev_gray: Optional[np.ndarray]) -> bool:
    if prev_gray is None:
        return True
    diff = cv2.absdiff(gray, prev_gray)
    return float(np.sum(diff > 25)) > MOTION_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
#  HUD
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray, class_name: str,
             present_log: List[Dict], total: int,
             session_start: float, fps: float,
             gray: Optional[np.ndarray],
             msg: str, msg_color: tuple) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.putText(frame,
                f"Class: {class_name}   Present: {len(present_log)}/{total}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    elapsed = int(time.time() - session_start)
    m, s    = divmod(elapsed, 60)
    cv2.putText(frame, f"{m:02d}:{s:02d}  {fps:.0f}fps",
                (w - 155, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (180, 220, 255), 1, cv2.LINE_AA)
    g  = gray if gray is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    br = float(np.mean(g))
    lc = (0, 200, 80) if BRIGHTNESS_MIN <= br <= BRIGHTNESS_MAX \
         else (40, 80, 220)
    cv2.putText(frame, f"Light:{br:.0f}",
                (w - 155, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                lc, 1, cv2.LINE_AA)
    if msg:
        cv2.rectangle(frame, (0, h - 44), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, msg, (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, msg_color, 2, cv2.LINE_AA)
    recent = present_log[-5:]
    for i, rec in enumerate(reversed(recent)):
        y = h - 52 - i * 20
        if y < 58:
            break
        alpha = max(0.3, 1.0 - i * 0.18)
        col   = tuple(int(c * alpha) for c in (80, 220, 80))
        cv2.putText(frame,
                    f"✓ {rec['name'][:20]}  {rec['sim']:.2f}",
                    (w - 260, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)
    cv2.putText(frame, "S=Stop  Q=Quit  R=Reload index",
                (10, h - 4) if not msg else (w - 240, h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                (100, 180, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY SCREEN
# ─────────────────────────────────────────────────────────────────────────────

def show_summary(frame: np.ndarray, present_log: List[Dict],
                 absent_log: List[Dict], class_name: str) -> bool:
    overlay = np.zeros_like(frame)
    overlay[:] = (20, 20, 20)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    h, w = frame.shape[:2]
    y    = 40
    cv2.putText(frame, f"Summary  —  {class_name}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (255, 255, 255), 2, cv2.LINE_AA)
    y += 34
    cv2.putText(frame,
                f"Present: {len(present_log)}   Absent: {len(absent_log)}"
                f"   Total: {len(present_log)+len(absent_log)}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54,
                (180, 220, 255), 1, cv2.LINE_AA)
    y += 28
    col_w = (w - 40) // 2
    cv2.putText(frame, "PRESENT", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 220, 80), 1, cv2.LINE_AA)
    y += 18
    for i, rec in enumerate(present_log):
        px = 20 + (i % 2) * col_w
        py = y  + (i // 2) * 19
        if py > h - 120:
            cv2.putText(frame, f"+{len(present_log)-i} more",
                        (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (80, 180, 80), 1, cv2.LINE_AA)
            break
        cv2.putText(frame, f"✓ {rec['name'][:22]}",
                    (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (80, 200, 80), 1, cv2.LINE_AA)
    ay = y + (len(present_log) // 2 + 1) * 19 + 12
    if ay < h - 120:
        cv2.putText(frame, "ABSENT", (20, ay),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 220), 1, cv2.LINE_AA)
        ay += 18
        for i, rec in enumerate(absent_log):
            px = 20 + (i % 2) * col_w
            py = ay + (i // 2) * 19
            if py > h - 80:
                cv2.putText(frame, f"+{len(absent_log)-i} more",
                            (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                            (80, 80, 200), 1, cv2.LINE_AA)
                break
            cv2.putText(frame, f"✗ {rec['name'][:22]}",
                        (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (100, 100, 220), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, h - 58), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "ENTER = save & end session     ESC = back to live",
                (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Will mark {len(absent_log)} student(s) absent",
                (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                (140, 140, 140), 1, cv2.LINE_AA)
    cv2.imshow("Live Attendance", frame)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 13:  return True
        if key == 27:  return False


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    print("=" * 58)
    print("  Live Attendance  —  High Performance")
    print("=" * 58)
    print()

    available = db_list_classes()
    if not available:
        print("[ERROR] No classes found. Run enrollment first.")
        sys.exit(1)

    print("  Available classes:")
    for i, c in enumerate(available):
        print(f"    [{i+1}] {c}")
    print()

    while True:
        raw = input("  Class (name or number) : ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(available):
            class_name = available[int(raw) - 1]; break
        elif raw in available:
            class_name = raw; break
        print("  [!] Not found.")

    while True:
        teacher_name = input("  Teacher name           : ").strip()
        if teacher_name:
            break
    print()

    print(f"[DB] Loading '{class_name}' ...")
    students = db_load_class_students(class_name)
    if not students:
        print(f"[ERROR] No students in '{class_name}'.")
        sys.exit(1)
    print(f"[DB] {len(students)} student(s) loaded.")

    id_to_student:   Dict[int, Dict] = {s["student_id"]: s for s in students}
    all_student_ids: Set[int]        = {s["student_id"] for s in students}
    class_id = students[0]["class_id"]

    thresholds = compute_adaptive_thresholds(students)
    print("[THRESH] Per-student thresholds:")
    for sid, t in thresholds.items():
        print(f"  {id_to_student[sid]['name']:<28} {t:.3f}")

    print(f"\n[FAISS] Building class index ...")
    index = build_class_index(students)
    print(f"[FAISS] {index.ntotal} vectors ready.")

    existing = db_get_todays_session(class_id)
    if existing:
        print(f"\n[SESSION] Active session found for today:")
        print(f"  teacher = {existing['teacher_name']}")
        print(f"  present = {existing['total_present']}")
        choice = input("  Resume it? (y/n) : ").strip().lower()
        if choice == "y":
            session_id      = str(existing["id"])
            already_present = db_get_already_present(session_id)
            print(f"[SESSION] Resumed — {len(already_present)} already marked.")
        else:
            session_id      = str(uuid.uuid4())
            already_present = set()
            db_create_session(session_id, class_id, teacher_name)
            print(f"[SESSION] New: {session_id[:8]}...")
    else:
        session_id      = str(uuid.uuid4())
        already_present = set()
        db_create_session(session_id, class_id, teacher_name)
        print(f"[SESSION] {session_id[:8]}...")

    session_start = time.time()

    write_queue: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_MAX)
    present_ids: Set[int]    = set(already_present)
    writer = threading.Thread(target=db_writer_thread,
                              args=(write_queue, present_ids), daemon=True)
    writer.start()

    models_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"\n[InsightFace] Loading {MODEL_NAME} ...")
    app = FaceAnalysis(name=MODEL_NAME, root=models_root,
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
    print(f"[InsightFace] Ready.\n")

    cam = CameraThread(CAMERA_INDEX)
    cam.start()
    time.sleep(0.5)

    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  LIVE ATTENDANCE  —  {class_name:<34}║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Students : {len(students):<45}║")
    print(f"║  Teacher  : {teacher_name:<45}║")
    print("║  Detection every frame  |  FAISS on motion only     ║")
    print("║  S=Stop   Q=Quit (no save)   R=Reload FAISS         ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    shutdown_flag = threading.Event()

    def _handle_signal(sig, frame_):
        print("\n[SIGNAL] Graceful shutdown.")
        shutdown_flag.set()

    signal.signal(signal.SIGINT, _handle_signal)

    present_log: List[Dict]       = []
    last_marked: Dict[int, float] = {}
    confirm_buf: Dict[int, int]   = {}
    smooth_buf:  Dict[int, deque] = {}
    prev_gray:   Optional[np.ndarray] = None
    frame_count: int   = 0
    fps_counter: int   = 0
    fps_timer:   float = time.time()
    fps:         float = 0.0
    msg:         str   = "Waiting for students..."
    msg_color:   tuple = (180, 180, 180)

    # Track last known name per face position for no-motion display
    last_name:   Dict[int, str]   = {}   # student_id → name
    last_sim:    Dict[int, float] = {}   # student_id → sim

    # Cache last drawn boxes — redrawn on skipped frames to stop blinking
    # List of (x1, y1, x2, y2, color, label)
    last_boxes: List[Tuple] = []

    while not shutdown_flag.is_set():
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        fps_counter += 1
        now_t = time.time()
        if now_t - fps_timer >= 1.0:
            fps         = fps_counter / (now_t - fps_timer)
            fps_counter = 0
            fps_timer   = now_t

        display = frame.copy()

        # Compute gray once — reused by motion, quality, hud
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Frame skip ────────────────────────────────────────────────────
        if frame_count % FRAME_SKIP != 0:
            # Redraw last known boxes — eliminates blinking on skipped frames
            for (bx1, by1, bx2, by2, bcol, blabel) in last_boxes:
                cv2.rectangle(display, (bx1, by1), (bx2, by2), bcol, 2)
                cv2.putText(display, blabel,
                            (bx1, max(by1-8, 55)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            bcol, 2, cv2.LINE_AA)
            draw_hud(display, class_name, present_log,
                     len(students), session_start, fps,
                     gray, msg, msg_color)
            cv2.imshow("Live Attendance", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                frame_count = -1
            prev_gray = gray
            continue

        motion    = has_motion(gray, prev_gray)
        prev_gray = gray

        # ── Detect faces (ALWAYS — so box is shown every frame) ───────────
        try:
            faces = app.get(frame)
        except Exception as e:
            faces = []
            print(f"[WARN] {e}")

        if not faces:
            # No face — decay confirm counts and clear box cache
            last_boxes = []
            for k in list(confirm_buf.keys()):
                confirm_buf[k] -= 1
                if confirm_buf[k] <= 0:
                    del confirm_buf[k]

        else:
            scored = [(f, *live_quality(f, gray)) for f in faces]
            good   = [(f, s, r) for f, s, r in scored
                      if f.embedding is not None and s >= 55.0]

            if good and motion:
                # ── FAISS matching (only when motion detected) ────────────
                raw_results = faiss_batch_match(
                    [f.embedding for f, _, _ in good],
                    index, id_to_student, thresholds)

                smoothed_embs: List[np.ndarray] = []
                smoothed_meta: List[Tuple]      = []

                for (face, score, reason), (_, sid, _, _) \
                        in zip(good, raw_results):
                    # OOM guard: skip sid=-1
                    if sid == -1:
                        smoothed_embs.append(l2_norm(face.embedding.copy()))
                        smoothed_meta.append((face, score, reason))
                        continue
                    # OOM guard: cap smooth_buf
                    if sid not in smooth_buf and len(smooth_buf) >= SMOOTH_BUF_MAX:
                        del smooth_buf[next(iter(smooth_buf))]
                    if sid not in smooth_buf:
                        smooth_buf[sid] = deque(maxlen=SMOOTH_WINDOW)
                    smooth_buf[sid].append((face.embedding.copy(), score))
                    buf_embs    = [e for e, _ in smooth_buf[sid]]
                    buf_weights = ([w for _, w in smooth_buf[sid]]
                                   if QUALITY_WEIGHT else [1.0]*len(smooth_buf[sid]))
                    smoothed_embs.append(quality_weighted_avg(buf_embs, buf_weights))
                    smoothed_meta.append((face, score, reason))

                final = faiss_batch_match(
                    smoothed_embs, index, id_to_student, thresholds)

                now = time.time()

                # Reset box cache each processed frame
                last_boxes = []

                for (face, score, reason), (matched, sid, name, avg_sim) \
                        in zip(smoothed_meta, final):
                    x1, y1, x2, y2 = get_bbox(face)

                    if matched and sid != -1:
                        last_name[sid] = name
                        last_sim[sid]  = avg_sim

                        confirm_buf[sid] = confirm_buf.get(sid, 0) + 1
                        for other in list(confirm_buf.keys()):
                            if other != sid:
                                confirm_buf[other] -= 1
                                if confirm_buf[other] <= 0:
                                    del confirm_buf[other]

                        already_marked = sid in present_ids
                        box_col = (0, 210, 80) if already_marked else (0, 180, 255)
                        prefix  = "✓ " if already_marked else ""
                        label   = f"{prefix}{name}  {avg_sim:.2f}"

                        cv2.rectangle(display, (x1, y1), (x2, y2), box_col, 2)
                        cv2.putText(display, label,
                                    (x1, max(y1-8, 55)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                                    box_col, 2, cv2.LINE_AA)

                        # Cache for skipped frames
                        last_boxes.append((x1, y1, x2, y2, box_col, label))

                        if not already_marked:
                            bw     = max(x2 - x1, 1)
                            pct    = confirm_buf.get(sid, 0) / CONFIRM_FRAMES
                            by     = max(y1 - 18, 58)
                            filled = min(int(pct * bw), bw)
                            cv2.rectangle(display, (x1, by), (x2, by+5),
                                          (40, 40, 40), -1)
                            if filled > 0:
                                cv2.rectangle(display, (x1, by),
                                              (x1+filled, by+5),
                                              (0, 210, 80), -1)

                        if (confirm_buf.get(sid, 0) >= CONFIRM_FRAMES
                                and not already_marked
                                and (now - last_marked.get(sid, 0)) > DEDUP_SECONDS):
                            try:
                                write_queue.put_nowait({
                                    "type":       "present",
                                    "student_id": sid,
                                    "class_id":   class_id,
                                    "session_id": session_id,
                                    "confidence": avg_sim,
                                })
                            except queue.Full:
                                print(f"[WARN] Queue full — {name} retries next pass")
                                continue
                            last_marked[sid] = now
                            if len(present_log) >= PRESENT_LOG_MAX:
                                present_log.pop(0)
                            present_log.append({
                                "student_id": sid, "name": name,
                                "sim": avg_sim,
                                "time": datetime.now().strftime("%H:%M:%S"),
                            })
                            msg       = f"Present: {name}"
                            msg_color = (0, 220, 80)
                            beep_present()
                            ts = datetime.now().strftime("%H:%M:%S")
                            print(f"[PRESENT]  {ts}  |  {name:<26}"
                                  f"  sim={avg_sim:.3f}"
                                  f"  thresh={thresholds.get(sid,0.45):.3f}"
                                  f"  id={sid}")
                    else:
                        for k in list(confirm_buf.keys()):
                            confirm_buf[k] -= 1
                            if confirm_buf[k] <= 0:
                                del confirm_buf[k]
                        label = f"Unknown {avg_sim:.2f}"
                        cv2.rectangle(display, (x1, y1), (x2, y2),
                                      (40, 40, 220), 2)
                        cv2.putText(display, label,
                                    (x1, max(y1-8, 55)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                                    (40, 80, 220), 1, cv2.LINE_AA)
                        last_boxes.append((x1, y1, x2, y2,
                                           (40, 40, 220), label))

            elif good and not motion:
                # ── No motion — draw box using last known identity ────────
                for face, score, reason in good:
                    x1, y1, x2, y2 = get_bbox(face)
                    # Find best student in smooth_buf by avg quality
                    best_sid = max(smooth_buf, key=lambda s:
                                   sum(w for _, w in smooth_buf[s]) / len(smooth_buf[s])
                                   ) if smooth_buf else -1
                    if best_sid != -1:
                        name = last_name.get(best_sid, "")
                        sim  = last_sim.get(best_sid, 0.0)
                        col  = (0, 210, 80) if best_sid in present_ids \
                               else (0, 180, 255)
                        cv2.rectangle(display, (x1, y1), (x2, y2), col, 2)
                        prefix = "✓ " if best_sid in present_ids else ""
                        cv2.putText(display,
                                    f"{prefix}{name}  {sim:.2f}",
                                    (x1, max(y1-8, 55)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                                    col, 2, cv2.LINE_AA)
                    else:
                        cv2.rectangle(display, (x1, y1), (x2, y2),
                                      (100, 100, 100), 1)

            # Draw low-quality face boxes
            for face, score, reason in scored:
                if score < 55.0:
                    x1, y1, x2, y2 = get_bbox(face)
                    cv2.rectangle(display, (x1, y1), (x2, y2),
                                  (60, 60, 60), 1)
                    cv2.putText(display,
                                f"{reason} {score:.0f}%",
                                (x1, max(y1-8, 55)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                                (120, 120, 120), 1, cv2.LINE_AA)

        draw_hud(display, class_name, present_log,
                 len(students), session_start, fps,
                 gray, msg, msg_color)
        cv2.imshow("Live Attendance", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            print("\n[QUIT] No save.")
            break

        if key in (ord("r"), ord("R")):
            print("\n[RELOAD] Reloading from MySQL ...")
            students        = db_load_class_students(class_name)
            index           = build_class_index(students)
            thresholds      = compute_adaptive_thresholds(students)
            id_to_student   = {s["student_id"]: s for s in students}
            all_student_ids = {s["student_id"] for s in students}
            smooth_buf.clear()
            last_name.clear()
            last_sim.clear()
            print(f"[RELOAD] {index.ntotal} vectors.")
            msg       = "Index reloaded"
            msg_color = (180, 220, 255)

        if key in (ord("s"), ord("S")) or frame_count == -1:
            write_queue.join()
            absent_ids = all_student_ids - present_ids
            absent_log = [
                {"student_id": sid,
                 "name":       id_to_student[sid]["name"],
                 "roll_no":    id_to_student[sid]["roll_no"]}
                for sid in sorted(absent_ids)
            ]
            confirmed = show_summary(
                display.copy(), present_log, absent_log, class_name)
            if confirmed:
                n_absent = db_mark_absent_bulk(
                    list(absent_ids), class_id, session_id)
                db_close_session(session_id, len(present_ids), n_absent)
                write_queue.put(None)
                cam.stop()
                cv2.destroyAllWindows()
                print("\n" + "═" * 58)
                print(f"  Session closed  —  {class_name}")
                elapsed = int(time.time() - session_start)
                m2, s2  = divmod(elapsed, 60)
                print(f"  Duration  : {m2:02d}:{s2:02d}")
                print(f"  Present   : {len(present_ids)}")
                print(f"  Absent    : {n_absent}")
                print("\n  PRESENT:")
                for rec in present_log:
                    print(f"    ✓  {rec['name']:<28}"
                          f"  {rec['time']}  sim={rec['sim']:.3f}")
                print("\n  ABSENT:")
                for rec in absent_log:
                    print(f"    ✗  {rec['name']:<28}"
                          f"  roll={rec['roll_no']}")
                print("═" * 58 + "\n")
                return
            else:
                msg         = "Back to live..."
                msg_color   = (180, 180, 180)
                frame_count = 0

    write_queue.put(None)
    cam.stop()
    cv2.destroyAllWindows()
    print("[SHUTDOWN] Session left open — resumable on next run.")


if __name__ == "__main__":
    run()