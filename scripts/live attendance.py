"""
Live Attendance
━━━━━━━━━━━━━━━
Model   : InsightFace buffalo_sc  (swap to AdaFace for production)
Storage : MySQL attendance table  +  sessions table
Matching: FAISS per-class index   +  aggregated avg similarity

Flow:
  1. Terminal  → teacher types class name + teacher name
  2. Camera opens → live detection starts
  3. Student walks past → detect → embed → FAISS match
     → if matched and not yet marked → mark present + beep
  4. S key → summary screen → confirm → save absents → close session

Scalability:
  - Only ONE class FAISS index loaded into RAM per session
  - Dedup guard: each student marked present at most once per session
  - Confidence score stored per attendance record for audit
  - Session record written to sessions table

Live attendance thresholds (more lenient than enrollment):
  - Lower NORM_MIN (student walking past, not posing)
  - Wider pose angles
  - More lenient lighting
  - Lower similarity threshold accounts for natural variation

Install:
  pip install opencv-python numpy insightface onnxruntime
  pip install mysql-connector-python faiss-cpu
"""

import cv2
import os
import sys
import time
import uuid
import warnings
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Set, Tuple, Protocol, runtime_checkable

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("[ERROR] pip install insightface onnxruntime")
    sys.exit(1)

try:
    import mysql.connector
    from mysql.connector import MySQLConnection
except ImportError:
    print("[ERROR] pip install mysql-connector-python")
    sys.exit(1)

try:
    import faiss
except ImportError:
    print("[ERROR] pip install faiss-cpu")
    sys.exit(1)

try:
    from config import db_password
except ImportError:
    db_password = "your_password"


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_INDEX  = 0
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
FAISS_DIR     = "faiss_indexes"

DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": db_password,
    "database": "attendance_system",
}
SCHOOL_ID = 1

MODEL_NAME    = "buffalo_sc"
EMBEDDING_DIM = 512

# ── Live attendance thresholds (more lenient than enrollment) ─────────────────
# Student walks past naturally — no posing, no retries
DET_SCORE_MIN        = 0.55    # slightly lenient — student moving
NORM_MIN_LIVE        = 14.0    # lenient — natural walk-past
YAW_MAX_LIVE         = 45.0    # wider — student may be slightly turned
PITCH_MAX_LIVE       = 35.0    # wider — laptop on desk, student looks down
BLUR_MIN_LIVE        = 60.0    # lenient — student moving
BRIGHTNESS_MIN       = 50      # lenient — various classroom conditions
BRIGHTNESS_MAX       = 220
SIMILARITY_THRESHOLD = 0.45    # buffalo_sc real-world sweet spot

# Dedup — minimum seconds before same student can be re-marked
# Prevents multiple marks if student lingers in front of camera
DEDUP_SECONDS        = 10

# How many consecutive frames must match same student before marking
# Prevents single-frame false positives
CONFIRM_FRAMES       = 3


# ─────────────────────────────────────────────────────────────────────────────
#  SOUND  (cross-platform beep)
# ─────────────────────────────────────────────────────────────────────────────

def beep_present():
    """Short double beep — student marked present."""
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


def beep_unknown():
    """Single low beep — unknown face detected."""
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(400, 200)
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


# ─────────────────────────────────────────────────────────────────────────────
#  MYSQL LAYER
# ─────────────────────────────────────────────────────────────────────────────

def db_connect() -> MySQLConnection:
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        print(f"[ERROR] MySQL: {e}")
        sys.exit(1)


def db_load_class_students(conn: MySQLConnection,
                            class_name: str) -> List[Dict]:
    """Load all students for a specific class."""
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """SELECT s.id AS student_id, s.name, s.roll_no,
                  s.emb_1, s.emb_2, s.emb_3,
                  c.id AS class_id, c.name AS class_name
           FROM   students s
           JOIN   classes  c ON c.id = s.class_id
           WHERE  s.school_id = %s AND c.name = %s
           ORDER  BY s.id""",
        (SCHOOL_ID, class_name),
    )
    rows = cur.fetchall()
    cur.close()

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


def db_get_class_id(conn: MySQLConnection,
                    class_name: str) -> Optional[int]:
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM classes WHERE school_id=%s AND name=%s LIMIT 1",
        (SCHOOL_ID, class_name),
    )
    row = cur.fetchone()
    cur.close()
    return int(row[0]) if row else None


def db_create_session(conn: MySQLConnection,
                      session_id: str,
                      class_id: int,
                      teacher_name: str) -> None:
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO sessions
           (id, school_id, class_id, teacher_name, status)
           VALUES (%s, %s, %s, %s, 'active')""",
        (session_id, SCHOOL_ID, class_id, teacher_name),
    )
    conn.commit()
    cur.close()


def db_mark_present(conn: MySQLConnection,
                    student_id: int,
                    class_id: int,
                    session_id: str,
                    confidence: float) -> bool:
    """
    Mark student present. Returns True if inserted, False if already marked.
    UNIQUE KEY uq_daily (student_id, date) prevents duplicates at DB level.
    """
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO attendance
               (student_id, class_id, school_id, date, status,
                confidence_score, session_id)
               VALUES (%s, %s, %s, CURDATE(), 'present', %s, %s)""",
            (student_id, class_id, SCHOOL_ID, round(confidence, 4), session_id),
        )
        conn.commit()
        cur.close()
        return True
    except mysql.connector.IntegrityError:
        # Already marked (duplicate key on student_id + date)
        return False
    except Exception as e:
        print(f"[WARN] Mark present failed: {e}")
        return False


def db_mark_absent_bulk(conn: MySQLConnection,
                        student_ids: List[int],
                        class_id: int,
                        session_id: str) -> int:
    """Mark list of student ids as absent. Returns count written."""
    if not student_ids:
        return 0
    cur   = conn.cursor()
    count = 0
    for sid in student_ids:
        try:
            cur.execute(
                """INSERT INTO attendance
                   (student_id, class_id, school_id, date, status,
                    confidence_score, session_id)
                   VALUES (%s, %s, %s, CURDATE(), 'absent', NULL, %s)""",
                (sid, class_id, SCHOOL_ID, session_id),
            )
            count += 1
        except mysql.connector.IntegrityError:
            pass  # already marked present, skip
    conn.commit()
    cur.close()
    return count


def db_close_session(conn: MySQLConnection,
                     session_id: str,
                     n_present: int,
                     n_absent: int) -> None:
    cur = conn.cursor()
    cur.execute(
        """UPDATE sessions
           SET ended_at=NOW(), status='completed',
               total_present=%s, total_absent=%s
           WHERE id=%s""",
        (n_present, n_absent, session_id),
    )
    conn.commit()
    cur.close()


# ─────────────────────────────────────────────────────────────────────────────
#  FAISS LAYER
# ─────────────────────────────────────────────────────────────────────────────

def build_class_index(students: List[Dict]) -> faiss.IndexIDMap:
    """Build FAISS index for a single class from MySQL embeddings."""
    base  = faiss.IndexFlatIP(EMBEDDING_DIM)
    index = faiss.IndexIDMap(base)
    for s in students:
        for emb in s["embeddings"]:
            vec = l2_norm(emb).reshape(1, -1).astype(np.float32)
            ids = np.array([s["student_id"]], dtype=np.int64)
            index.add_with_ids(vec, ids)
    return index


def faiss_match(live_emb: np.ndarray,
                index: faiss.IndexIDMap,
                id_to_student: Dict[int, Dict]) -> Tuple[bool, int, str, float]:
    """
    Search live embedding against class index.
    Returns scores for ALL stored vectors, groups by student_id,
    takes average per student, winner = highest average.
    Returns (matched, student_id, name, avg_similarity).
    """
    if index.ntotal == 0:
        return False, -1, "", 0.0

    vec = l2_norm(live_emb).reshape(1, -1).astype(np.float32)
    k   = index.ntotal

    scores, ids = index.search(vec, k)

    # Group scores by student_id
    id_scores: Dict[int, List[float]] = {}
    for score, sid in zip(scores[0], ids[0]):
        sid = int(sid)
        if sid == -1:
            continue
        if sid not in id_scores:
            id_scores[sid] = []
        id_scores[sid].append(float(score))

    if not id_scores:
        return False, -1, "", 0.0

    # Average per student
    avg_scores = {
        sid: sum(sc) / len(sc)
        for sid, sc in id_scores.items()
    }

    best_id  = max(avg_scores, key=lambda s: avg_scores[s])
    best_avg = avg_scores[best_id]
    name     = id_to_student.get(best_id, {}).get("name", "Unknown")
    matched  = best_avg >= SIMILARITY_THRESHOLD

    return matched, best_id, name, best_avg


# ─────────────────────────────────────────────────────────────────────────────
#  FACE QUALITY  (live — lenient)
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class FaceObject(Protocol):
    bbox:      np.ndarray
    det_score: float
    embedding: Optional[np.ndarray]
    pose:      Optional[np.ndarray]


def live_quality(face: FaceObject,
                 frame: np.ndarray) -> Tuple[float, str]:
    """
    Lenient quality check for live walk-past detection.
    Hard blocks only on definitive failures.
    """
    try:
        det  = float(face.det_score)
        emb  = face.embedding
        norm = float(np.linalg.norm(emb)) if emb is not None else 0.0
        pose = face.pose
        yaw  = abs(float(pose[1])) if pose is not None and len(pose) > 1 else 0.0
        pitch = abs(float(pose[0])) if pose is not None and len(pose) > 0 else 0.0
    except Exception as e:
        return 0.0, f"Error: {e}"

    if det < DET_SCORE_MIN:
        return det * 100, f"Low conf ({det:.2f})"
    if norm < NORM_MIN_LIVE:
        return (norm / NORM_MIN_LIVE) * 50, f"Low norm ({norm:.1f})"
    if yaw > YAW_MAX_LIVE:
        return max(10.0, 100 - yaw * 2), f"Too side-on ({yaw:.0f}deg)"
    if pitch > PITCH_MAX_LIVE:
        return max(10.0, 100 - pitch * 2), f"Looking away ({pitch:.0f}deg)"

    # Lighting — soft check only
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    if mean < BRIGHTNESS_MIN:
        return 20.0, f"Too dark ({mean:.0f})"
    if mean > BRIGHTNESS_MAX:
        return 20.0, f"Too bright ({mean:.0f})"

    # Blur — soft check only
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur < BLUR_MIN_LIVE:
        return 30.0, f"Blurry ({blur:.0f})"

    det_s  = min(det, 1.0)
    norm_s = min(norm / 25.0, 1.0)
    yaw_s  = max(0.0, 1.0 - yaw / YAW_MAX_LIVE)
    score  = (det_s * 0.40 + norm_s * 0.40 + yaw_s * 0.20) * 100
    return float(int(score * 10) / 10), ""


def get_best_face_live(faces: list,
                       frame: np.ndarray
                       ) -> Tuple[Optional[FaceObject], float, str]:
    best_face:   Optional[FaceObject] = None
    best_score:  float = 0.0
    best_reason: str   = "No face"
    for face in faces:
        f: FaceObject = face
        s, r = live_quality(f, frame)
        if s > best_score:
            best_score  = s
            best_reason = r
            best_face   = f
    return best_face, best_score, best_reason


def get_bbox(face: FaceObject) -> Tuple[int, int, int, int]:
    try:
        b = face.bbox
        return int(b[0]), int(b[1]), int(b[2]), int(b[3])
    except Exception:
        return 0, 0, 0, 0


# ─────────────────────────────────────────────────────────────────────────────
#  HUD
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray,
             class_name: str,
             present: List[Dict],
             total_students: int,
             session_start: float,
             msg: str,
             msg_color: tuple) -> None:
    h, w = frame.shape[:2]

    # ── Top bar ───────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.putText(frame,
                f"Class: {class_name}   Present: {len(present)}/{total_students}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

    elapsed = int(time.time() - session_start)
    m, s    = divmod(elapsed, 60)
    cv2.putText(frame, f"Session: {m:02d}:{s:02d}",
                (w - 160, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (180, 220, 255), 1, cv2.LINE_AA)

    # ── Lighting indicator ────────────────────────────────────────────────
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    light_ok   = BRIGHTNESS_MIN <= brightness <= BRIGHTNESS_MAX
    lc         = (0, 200, 80) if light_ok else (40, 80, 220)
    cv2.putText(frame, f"Light:{brightness:.0f}",
                (w - 160, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                lc, 1, cv2.LINE_AA)

    # ── Message ───────────────────────────────────────────────────────────
    if msg:
        cv2.rectangle(frame, (0, h - 44), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, msg, (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    msg_color, 2, cv2.LINE_AA)

    # ── Bottom right — recently marked ────────────────────────────────────
    recent = present[-4:] if len(present) > 4 else present
    for i, rec in enumerate(reversed(recent)):
        y = h - 50 - i * 22
        if y < 55:
            break
        alpha = 1.0 - i * 0.2
        col   = tuple(int(c * alpha) for c in (80, 220, 80))
        cv2.putText(frame,
                    f"✓ {rec['name']}  {rec['sim']:.2f}",
                    (w - 260, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    col, 1, cv2.LINE_AA)

    cv2.putText(frame, "S=Stop session  Q=Quit",
                (10, h - 18) if not msg else (10, h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (100, 180, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY SCREEN
# ─────────────────────────────────────────────────────────────────────────────

def show_summary(frame: np.ndarray,
                 present: List[Dict],
                 absent: List[Dict],
                 class_name: str) -> bool:
    """
    Draw summary overlay on frame.
    Returns True = confirmed save, False = go back.
    Blocks until teacher presses ENTER (save) or ESC (back).
    """
    overlay = np.zeros_like(frame)
    overlay[:] = (20, 20, 20)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    h, w = frame.shape[:2]
    y    = 40

    cv2.putText(frame, f"Session summary  —  {class_name}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    y += 36

    cv2.putText(frame,
                f"Present: {len(present)}   Absent: {len(absent)}   "
                f"Total: {len(present) + len(absent)}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (180, 220, 255), 1, cv2.LINE_AA)
    y += 32

    # Present list
    cv2.putText(frame, "PRESENT", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (80, 220, 80), 1, cv2.LINE_AA)
    y += 22
    col_w = (w - 40) // 2
    for i, rec in enumerate(present):
        col = i % 2
        row = i // 2
        px  = 20 + col * col_w
        py  = y + row * 20
        if py > h - 120:
            cv2.putText(frame, f"... +{len(present)-i} more",
                        (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (80, 180, 80), 1, cv2.LINE_AA)
            break
        cv2.putText(frame,
                    f"✓ {rec['name'][:22]}",
                    (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (80, 200, 80), 1, cv2.LINE_AA)

    # Absent list
    absent_y = y + (len(present) // 2 + 1) * 20 + 16
    if absent_y < h - 120:
        cv2.putText(frame, "ABSENT", (20, absent_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (80, 80, 220), 1, cv2.LINE_AA)
        absent_y += 22
        for i, rec in enumerate(absent):
            col = i % 2
            row = i // 2
            px  = 20 + col * col_w
            py  = absent_y + row * 20
            if py > h - 80:
                cv2.putText(frame, f"... +{len(absent)-i} more",
                            (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (80, 80, 200), 1, cv2.LINE_AA)
                break
            cv2.putText(frame,
                        f"✗ {rec['name'][:22]}",
                        (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (100, 100, 220), 1, cv2.LINE_AA)

    # Confirm bar
    cv2.rectangle(frame, (0, h - 60), (w, h), (30, 30, 30), -1)
    cv2.putText(frame,
                "ENTER = save attendance & end session     ESC = go back",
                (20, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"This will mark {len(absent)} student(s) absent in MySQL",
                (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (140, 140, 140), 1, cv2.LINE_AA)

    cv2.imshow("Live Attendance", frame)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 13:    # ENTER
            return True
        if key == 27:    # ESC
            return False


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    print("=" * 56)
    print("  Live Attendance  —  MySQL + FAISS + buffalo_sc")
    print("=" * 56)

    # ── Terminal: teacher inputs ──────────────────────────────────────────
    print()
    while True:
        class_name = input("  Class name (e.g. btech_cs_1) : ").strip()
        if class_name:
            break
        print("  [!] Cannot be empty.")

    while True:
        teacher_name = input("  Teacher name                 : ").strip()
        if teacher_name:
            break
        print("  [!] Cannot be empty.")

    print()

    # ── Connect MySQL ─────────────────────────────────────────────────────
    print("[DB] Connecting to MySQL ...")
    conn = db_connect()
    print("[DB] Connected.")

    # ── Load class students ───────────────────────────────────────────────
    print(f"[DB] Loading students for class '{class_name}' ...")
    students = db_load_class_students(conn, class_name)

    if not students:
        print(f"[ERROR] No students found for class '{class_name}'.")
        print("  Available classes in MySQL:")
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM classes WHERE school_id=%s ORDER BY name",
            (SCHOOL_ID,)
        )
        for (n,) in cur.fetchall():
            idx = None
            fp  = os.path.join(FAISS_DIR, f"{n}.index")
            if os.path.exists(fp):
                idx = faiss.read_index(fp).ntotal
            print(f"    {n:<30} {'index: '+str(idx)+' vectors' if idx else 'no index'}")
        cur.close()
        conn.close()
        sys.exit(1)

    print(f"[DB] {len(students)} student(s) loaded.")

    id_to_student: Dict[int, Dict] = {
        s["student_id"]: s for s in students
    }
    all_student_ids: Set[int] = {s["student_id"] for s in students}
    class_id = students[0]["class_id"]

    # ── Build FAISS index for this class only ─────────────────────────────
    print(f"[FAISS] Building class index ({len(students)} students × 3 vectors) ...")
    index     = build_class_index(students)
    print(f"[FAISS] {index.ntotal} vectors loaded into RAM.\n")

    # ── Create session ────────────────────────────────────────────────────
    session_id    = str(uuid.uuid4())
    session_start = time.time()
    db_create_session(conn, session_id, class_id, teacher_name)
    print(f"[SESSION] id={session_id[:8]}...  "
          f"class={class_name}  teacher={teacher_name}")

    # ── Load InsightFace ──────────────────────────────────────────────────
    models_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    print(f"\n[InsightFace] Loading {MODEL_NAME} ...")
    app = FaceAnalysis(
        name=MODEL_NAME,
        root=models_root,
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    warnings.filterwarnings("ignore", category=FutureWarning,
                            module="insightface")
    print(f"[InsightFace] {MODEL_NAME} ready.\n")

    # ── Camera ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Try CAMERA_INDEX = 1")
        conn.close()
        return

    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  LIVE ATTENDANCE  —  {class_name:<32}║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Students: {len(students):<43}║")
    print(f"║  Teacher : {teacher_name:<43}║")
    print("║  Students walk past camera → auto-marked present    ║")
    print("║  S = stop session   Q = quit without saving         ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── State ─────────────────────────────────────────────────────────────
    present_ids:    Set[int]   = set()   # marked present this session
    present_log:    List[Dict] = []      # ordered list for display
    last_marked:    Dict[int, float] = {}  # student_id → last mark time

    # Frame confirmation buffer — prevents single-frame false positives
    # { student_id: consecutive_frame_count }
    confirm_buffer: Dict[int, int] = {}

    msg:       str   = "Waiting for students..."
    msg_color: tuple = (180, 180, 180)

    # ── Live loop ─────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        display = frame.copy()

        try:
            faces = app.get(frame)
        except Exception as e:
            faces = []
            print(f"[WARN] {e}")

        best_face, best_score, best_reason = \
            get_best_face_live(faces, frame)

        if best_face is not None:
            x1, y1, x2, y2 = get_bbox(best_face)
            emb = best_face.embedding

            if emb is not None and best_score >= 60.0:
                matched, sid, name, avg_sim = faiss_match(
                    emb, index, id_to_student)

                now = time.time()

                if matched:
                    # Update confirmation buffer
                    confirm_buffer[sid] = confirm_buffer.get(sid, 0) + 1

                    # Draw green box + name
                    cv2.rectangle(display,
                                  (x1, y1), (x2, y2),
                                  (0, 210, 80), 2)
                    label = f"{name}  {avg_sim:.2f}"
                    cv2.putText(display, label,
                                (x1, max(y1 - 8, 55)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 210, 80), 2, cv2.LINE_AA)

                    # Mark present if confirmed + not recently marked
                    if (confirm_buffer.get(sid, 0) >= CONFIRM_FRAMES
                            and sid not in present_ids
                            and (now - last_marked.get(sid, 0)) > DEDUP_SECONDS):

                        written = db_mark_present(
                            conn, sid, class_id, session_id, avg_sim)

                        if written:
                            present_ids.add(sid)
                            last_marked[sid] = now
                            present_log.append({
                                "student_id": sid,
                                "name":       name,
                                "sim":        avg_sim,
                                "time":       datetime.now().strftime("%H:%M:%S"),
                            })
                            msg       = f"Present: {name}"
                            msg_color = (0, 220, 80)
                            beep_present()
                            ts = datetime.now().strftime("%H:%M:%S")
                            print(f"[PRESENT]  {ts}  |  {name:<25}  "
                                  f"sim={avg_sim:.3f}  "
                                  f"id={sid}")
                        else:
                            msg       = f"Already marked: {name}"
                            msg_color = (0, 160, 255)

                else:
                    # Unknown face
                    confirm_buffer.clear()
                    cv2.rectangle(display,
                                  (x1, y1), (x2, y2),
                                  (40, 40, 220), 2)
                    cv2.putText(display,
                                f"Unknown  {avg_sim:.2f}",
                                (x1, max(y1 - 8, 55)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                                (40, 80, 220), 1, cv2.LINE_AA)

            else:
                # Low quality face
                confirm_buffer.clear()
                cv2.rectangle(display,
                              (x1, y1), (x2, y2),
                              (60, 60, 60), 1)
                cv2.putText(display,
                            f"{best_reason}  {best_score:.0f}%",
                            (x1, max(y1 - 8, 55)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                            (120, 120, 120), 1, cv2.LINE_AA)
        else:
            confirm_buffer.clear()

        draw_hud(display, class_name, present_log,
                 len(students), session_start, msg, msg_color)

        cv2.imshow("Live Attendance", display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            print("\n[QUIT] Exiting without saving.")
            cap.release()
            cv2.destroyAllWindows()
            conn.close()
            return

        if key in (ord("s"), ord("S")):
            # ── Summary screen ────────────────────────────────────────────
            absent_ids = all_student_ids - present_ids
            absent_log = [
                {"student_id": sid,
                 "name": id_to_student[sid]["name"],
                 "roll_no": id_to_student[sid]["roll_no"]}
                for sid in sorted(absent_ids)
            ]

            confirmed = show_summary(
                display.copy(), present_log, absent_log, class_name)

            if confirmed:
                # Write absents + close session
                n_absent = db_mark_absent_bulk(
                    conn, list(absent_ids), class_id, session_id)
                db_close_session(
                    conn, session_id,
                    len(present_ids), n_absent)

                cap.release()
                cv2.destroyAllWindows()
                conn.close()

                # ── Final terminal summary ────────────────────────────────
                print("\n" + "═" * 56)
                print(f"  Session closed  —  {class_name}")
                print(f"  Session id : {session_id[:8]}...")
                print(f"  Teacher    : {teacher_name}")
                elapsed = int(time.time() - session_start)
                m, s    = divmod(elapsed, 60)
                print(f"  Duration   : {m:02d}:{m:02d}")
                print(f"  Present    : {len(present_ids)}")
                print(f"  Absent     : {n_absent}")
                print(f"  Total      : {len(students)}")
                print()
                print("  PRESENT:")
                for rec in present_log:
                    print(f"    ✓  {rec['name']:<28} "
                          f"{rec['time']}  sim={rec['sim']:.3f}")
                print()
                print("  ABSENT:")
                for rec in absent_log:
                    print(f"    ✗  {rec['name']:<28} "
                          f"roll={rec['roll_no']}")
                print("═" * 56 + "\n")
                return

            else:
                # Teacher pressed ESC — go back to live loop
                msg       = "Back to live attendance..."
                msg_color = (180, 180, 180)
                continue


if __name__ == "__main__":
    run()